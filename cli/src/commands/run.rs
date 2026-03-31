// Copyright 2024 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! This file contains the internal implementation of `run`.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::IsTerminal as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::mpsc;

use crossterm::ExecutableCommand as _;
use crossterm::event::Event;
use crossterm::event::KeyCode;
use crossterm::event::KeyModifiers;
use crossterm::terminal::EnterAlternateScreen;
use crossterm::terminal::LeaveAlternateScreen;
use crossterm::terminal::disable_raw_mode;
use crossterm::terminal::enable_raw_mode;
use futures::TryStreamExt as _;
use itertools::Itertools as _;
use jj_lib::backend::BackendError;
use jj_lib::backend::CommitId;
use jj_lib::commit::Commit;
use jj_lib::commit::CommitIteratorExt as _;
use jj_lib::gitignore::GitIgnoreFile;
use jj_lib::local_working_copy::TreeState;
use jj_lib::local_working_copy::TreeStateError;
use jj_lib::local_working_copy::TreeStateSettings;
use jj_lib::lock::FileLock;
use jj_lib::matchers::EverythingMatcher;
use jj_lib::matchers::NothingMatcher;
use jj_lib::merged_tree::MergedTree;
use jj_lib::object_id::ObjectId as _;
use jj_lib::repo::Repo as _;
use jj_lib::working_copy::CheckoutError;
use jj_lib::working_copy::SnapshotError;
use jj_lib::working_copy::SnapshotOptions;
use ratatui::Terminal;
use ratatui::layout::Constraint;
use ratatui::layout::Direction;
use ratatui::layout::Layout;
use ratatui::prelude::CrosstermBackend;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;
use ratatui::widgets::List;
use ratatui::widgets::ListState;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Wrap;

use crate::cli_util::CommandHelper;
use crate::cli_util::RevisionArg;
use crate::command_error::CommandError;
use crate::ui::Ui;

/// Run a command across a set of revisions.
///
/// The command is executed in a temporary working copy for each revision.
/// If the command modifies any tracked files, the revision is rewritten
/// with the new content. Descendants are rebased accordingly.
///
/// Environment variables set for the command:
///   JJ_CHANGE  The change ID of the current revision
///   JJ_COMMIT  The commit ID of the current revision
///
/// Immutable revisions cannot be rewritten.
///
/// Examples:
///   jj run 'cargo fmt' -r 'trunk()..@'
///   jj run 'cargo clippy' -r @
#[derive(clap::Args, Clone, Debug)]
#[command(verbatim_doc_comment)]
pub struct RunArgs {
    /// The command to run across all selected revisions.
    shell_command: String,

    /// The revisions to change.
    #[arg(
        long = "revision",
        short,
        default_value = "@",
        value_name = "REVSETS",
        alias = "revisions"
    )]
    revisions: Vec<RevisionArg>,

    /// A no-op option to match the interface of `git rebase -x`.
    #[arg(short = 'x', hide = true)]
    _exec: bool,

    /// How many processes should run in parallel (default: number of CPU
    /// cores).
    #[arg(long, short)]
    jobs: Option<usize>,

    /// Remove cached working copies before running. By default, working
    /// copies are reused between invocations so that ignored files (like
    /// build artifacts) persist for incremental builds.
    #[arg(long)]
    clean: bool,

    /// Don't rewrite commits. The command's exit code is reported but any
    /// file changes are ignored. Useful for running tests or builds across
    /// revisions without modifying the repo.
    #[arg(long)]
    readonly: bool,

    /// Keep going even if the command fails on some revisions. Failed
    /// commits are left unmodified. A summary of failures is reported at
    /// the end.
    #[arg(long, short = 'k')]
    keep_going: bool,
}

// --- Error types ---

#[derive(Debug, thiserror::Error)]
enum RunError {
    #[error("Failed to checkout commit: {0}")]
    FailedCheckout(#[from] CheckoutError),
    #[error("Command '{cmd}' failed with {status} for commit {commit}")]
    CommandFailure {
        cmd: String,
        status: std::process::ExitStatus,
        commit: CommitId,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("Failed to create path {}: {source}", path.display())]
    PathCreation { path: PathBuf, source: io::Error },
    #[error("Failed to acquire lock on run directory: {0}")]
    Lock(#[from] jj_lib::lock::FileLockError),
    #[error(transparent)]
    TreeState(#[from] TreeStateError),
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

impl From<RunError> for CommandError {
    fn from(value: RunError) -> Self {
        Self::new(crate::command_error::CommandErrorKind::Cli, Box::new(value))
    }
}

// --- Core execution ---

fn get_shell() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("cmd", "/c")
    } else {
        ("/bin/sh", "-c")
    }
}

/// Result of running a command on a commit, including captured output.
struct RunOutput {
    new_tree: Option<MergedTree>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_command_on_commit(
    tree_state: &mut TreeState,
    shell_command: &str,
    commit: &Commit,
    base_ignores: Arc<GitIgnoreFile>,
    readonly: bool,
) -> Result<RunOutput, RunError> {
    tree_state.check_out(&commit.tree())?;

    let output = {
        let (prog, first_arg) = get_shell();
        Command::new(prog)
            .arg(first_arg)
            .arg(shell_command)
            .current_dir(tree_state.working_copy_path())
            .env("JJ_CHANGE", commit.change_id().hex())
            .env("JJ_COMMIT", commit.id().hex())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?
    };

    if !output.status.success() {
        return Err(RunError::CommandFailure {
            cmd: shell_command.to_owned(),
            status: output.status,
            commit: commit.id().clone(),
            stdout: output.stdout,
            stderr: output.stderr,
        });
    }

    if readonly {
        return Ok(RunOutput { new_tree: None, stdout: output.stdout, stderr: output.stderr });
    }

    let options = SnapshotOptions {
        base_ignores,
        start_tracking_matcher: &EverythingMatcher,
        progress: None,
        max_new_file_size: 64_000_000,
        force_tracking_matcher: &NothingMatcher,
    };
    let (dirty, _) = pollster::FutureExt::block_on(tree_state.snapshot(&options))?;
    Ok(RunOutput {
        new_tree: if dirty { Some(tree_state.current_tree().clone()) } else { None },
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn ensure_dir(path: &PathBuf) -> Result<(), RunError> {
    if !path.exists() {
        fs::create_dir_all(path)
            .map_err(|e| RunError::PathCreation { path: path.clone(), source: e })?;
    }
    Ok(())
}

fn init_or_load_tree_state(
    store: Arc<jj_lib::store::Store>,
    working_copy_dir: &PathBuf,
    state_dir: &PathBuf,
    settings: &TreeStateSettings,
) -> Result<TreeState, RunError> {
    ensure_dir(working_copy_dir)?;
    ensure_dir(state_dir)?;
    let tree_state_path = state_dir.join("tree_state");
    if tree_state_path.exists() {
        Ok(TreeState::load(store, working_copy_dir.clone(), state_dir.clone(), settings)?)
    } else {
        Ok(TreeState::init(store, working_copy_dir.clone(), state_dir.clone(), settings)?)
    }
}

// --- TUI types ---

/// Message from worker → TUI.
enum TuiMessage {
    Started { commit_id: CommitId, worker: usize },
    Passed { commit_id: CommitId, modified: bool, stdout: Vec<u8>, stderr: Vec<u8>, diff_summary: String },
    Failed { commit_id: CommitId, message: String, stdout: Vec<u8>, stderr: Vec<u8> },
    AllDone,
}

#[derive(Clone, PartialEq)]
enum JobStatus {
    Pending,
    Running(usize),
    Passed { modified: bool },
    Failed(String),
}

struct CommitEntry {
    commit_id: CommitId,
    change_id_hex: String,
    description: String,
    status: JobStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    /// Summary of file changes (populated for modified commits).
    diff_summary: String,
}

#[derive(Clone, Copy, PartialEq)]
enum DetailView { None, Stdout, Stderr, Diff }

/// What the user decided in the TUI.
enum TuiOutcome {
    /// Apply changes and quit.
    Confirm,
    /// Discard changes and quit.
    Cancel,
}

struct TuiState {
    commits: Vec<CommitEntry>,
    selected: usize,
    detail: DetailView,
    done: bool,
    outcome: Option<TuiOutcome>,
    passed: usize,
    failed: usize,
}

impl TuiState {
    fn new(commits: &[Commit]) -> Self {
        TuiState {
            commits: commits.iter().map(|c| CommitEntry {
                commit_id: c.id().clone(),
                change_id_hex: c.change_id().hex(),
                description: c.description().lines().next().unwrap_or("").to_string(),
                status: JobStatus::Pending,
                stdout: Vec::new(),
                stderr: Vec::new(),
                diff_summary: String::new(),
            }).collect(),
            selected: 0,
            detail: DetailView::None,
            done: false,
            outcome: None,
            passed: 0,
            failed: 0,
        }
    }

    fn find_mut(&mut self, id: &CommitId) -> Option<&mut CommitEntry> {
        self.commits.iter_mut().find(|c| c.commit_id == *id)
    }

    fn handle_message(&mut self, msg: TuiMessage) {
        match msg {
            TuiMessage::Started { commit_id, worker } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.status = JobStatus::Running(worker);
                }
            }
            TuiMessage::Passed { commit_id, modified, stdout, stderr, diff_summary } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.status = JobStatus::Passed { modified };
                    e.stdout = stdout;
                    e.stderr = stderr;
                    e.diff_summary = diff_summary;
                }
                self.passed += 1;
            }
            TuiMessage::Failed { commit_id, message, stdout, stderr } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.status = JobStatus::Failed(message);
                    e.stdout = stdout;
                    e.stderr = stderr;
                }
                self.failed += 1;
            }
            TuiMessage::AllDone => {
                self.done = true;
            }
        }
    }
}

// --- TUI rendering ---

fn render_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &TuiState,
    shell_command: &str,
) -> io::Result<()> {
    terminal.draw(|frame| {
        let show_detail = state.detail != DetailView::None;
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if show_detail {
                vec![Constraint::Length(1), Constraint::Percentage(50), Constraint::Percentage(50), Constraint::Length(1)]
            } else {
                vec![Constraint::Length(1), Constraint::Fill(1), Constraint::Length(0), Constraint::Length(1)]
            })
            .split(frame.area());

        // Header
        let done = state.passed + state.failed;
        let total = state.commits.len();
        let header = Line::from(vec![
            Span::styled(format!(" jj run '{shell_command}' "), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("[{done}/{total}] ")),
            Span::styled(format!("{} passed", state.passed), Style::default().fg(Color::Green)),
            if state.failed > 0 {
                Span::styled(format!(" {} failed", state.failed), Style::default().fg(Color::Red))
            } else {
                Span::raw("")
            },
        ]);
        frame.render_widget(header, chunks[0]);

        // Commit list
        let items: Vec<Line> = state.commits.iter().enumerate().map(|(i, e)| {
            let short = &e.change_id_hex[..8.min(e.change_id_hex.len())];
            let desc = if e.description.is_empty() { "(no description)" } else { &e.description };
            let (sym, style) = match &e.status {
                JobStatus::Pending => ("○", Style::default().fg(Color::DarkGray)),
                JobStatus::Running(_) => ("◑", Style::default().fg(Color::Yellow)),
                JobStatus::Passed { modified: true } => ("●", Style::default().fg(Color::Green)),
                JobStatus::Passed { modified: false } => ("○", Style::default().fg(Color::Green)),
                JobStatus::Failed(_) => ("✗", Style::default().fg(Color::Red)),
            };
            let sel = if i == state.selected { "▸ " } else { "  " };
            Line::from(vec![
                Span::raw(sel),
                Span::styled(format!("{sym} "), style),
                Span::styled(format!("{short} "), Style::default().fg(Color::Magenta)),
                Span::styled(desc.to_string(), style),
            ])
        }).collect();

        let list = List::new(items).block(Block::default().borders(Borders::NONE));
        let mut list_state = ListState::default().with_selected(Some(state.selected));
        frame.render_stateful_widget(list, chunks[1], &mut list_state);

        // Detail panel
        if show_detail {
            let e = &state.commits[state.selected];
            let (title, content) = match state.detail {
                DetailView::Stdout => ("stdout (o: close, e: stderr, d: diff)", String::from_utf8_lossy(&e.stdout).to_string()),
                DetailView::Stderr => ("stderr (e: close, o: stdout, d: diff)", String::from_utf8_lossy(&e.stderr).to_string()),
                DetailView::Diff => ("diff (d: close, o: stdout, e: stderr)",
                    if e.diff_summary.is_empty() { "(no changes)".to_string() } else { e.diff_summary.clone() }),
                DetailView::None => unreachable!(),
            };
            let p = Paragraph::new(content)
                .block(Block::default().title(title).borders(Borders::TOP))
                .wrap(Wrap { trim: false });
            frame.render_widget(p, chunks[2]);
        }

        // Help bar
        let mut help_spans = vec![
            Span::styled("↑/↓", Style::default().fg(Color::Magenta)), Span::raw(" nav "),
            Span::styled("o", Style::default().fg(Color::Magenta)), Span::raw(" stdout "),
            Span::styled("e", Style::default().fg(Color::Magenta)), Span::raw(" stderr "),
            Span::styled("d", Style::default().fg(Color::Magenta)), Span::raw(" diff "),
        ];
        if state.done {
            help_spans.extend([
                Span::styled("c", Style::default().fg(Color::Green)), Span::raw(" confirm "),
                Span::styled("q", Style::default().fg(Color::Red)), Span::raw(" discard"),
            ]);
        } else {
            help_spans.extend([
                Span::styled("q", Style::default().fg(Color::Magenta)), Span::raw(" cancel"),
            ]);
        }
        let help = Line::from(help_spans);
        frame.render_widget(help, chunks[3]);
    })?;
    Ok(())
}

fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    rx: &mpsc::Receiver<TuiMessage>,
    shell_command: &str,
) {
    loop {
        render_tui(terminal, state, shell_command).ok();

        // Drain worker messages.
        while let Ok(msg) = rx.try_recv() {
            state.handle_message(msg);
        }

        // Poll keyboard with timeout.
        if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = crossterm::event::read() {
                if key.is_release() { continue; }
                match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) | (KeyCode::Esc, _) => {
                        state.outcome = Some(TuiOutcome::Cancel);
                        return;
                    }
                    (KeyCode::Char('q'), KeyModifiers::NONE) => {
                        state.outcome = Some(TuiOutcome::Cancel);
                        return;
                    }
                    (KeyCode::Char('c'), KeyModifiers::NONE) if state.done => {
                        state.outcome = Some(TuiOutcome::Confirm);
                        return;
                    }
                    (KeyCode::Down | KeyCode::Char('j'), KeyModifiers::NONE) => {
                        if state.selected + 1 < state.commits.len() { state.selected += 1; }
                    }
                    (KeyCode::Up | KeyCode::Char('k'), KeyModifiers::NONE) => {
                        if state.selected > 0 { state.selected -= 1; }
                    }
                    (KeyCode::Char('o'), _) => {
                        state.detail = if state.detail == DetailView::Stdout { DetailView::None } else { DetailView::Stdout };
                    }
                    (KeyCode::Char('e'), _) => {
                        state.detail = if state.detail == DetailView::Stderr { DetailView::None } else { DetailView::Stderr };
                    }
                    (KeyCode::Char('d'), _) => {
                        state.detail = if state.detail == DetailView::Diff { DetailView::None } else { DetailView::Diff };
                    }
                    _ => {}
                }
            }
        }

        if let Some(_) = &state.outcome {
            return;
        }
    }
}

// --- Main command ---

pub async fn cmd_run(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &RunArgs,
) -> Result<(), CommandError> {
    let mut workspace_command = command.workspace_helper(ui)?;
    let resolved_commits: Vec<Commit> = workspace_command
        .parse_union_revsets(ui, &args.revisions)?
        .evaluate_to_commits()?
        .try_collect()
        .await?;

    if !args.readonly {
        workspace_command
            .check_rewritable(resolved_commits.iter().ids())
            .await?;
    }

    let jobs = match args.jobs {
        Some(0) | None => std::thread::available_parallelism().map(|t| t.into()).ok(),
        Some(jobs) => Some(jobs),
    }
    .unwrap_or(1usize)
    .min(resolved_commits.len());

    if resolved_commits.is_empty() {
        writeln!(ui.status(), "No revisions to run on.")?;
        return Ok(());
    }

    let base_ignores = workspace_command.base_ignores()?;
    let tree_state_settings =
        TreeStateSettings::try_from_user_settings(workspace_command.settings())?;

    let repo_path = workspace_command.repo_path().to_owned();
    let run_base = repo_path.parent().unwrap().join("run");
    ensure_dir(&run_base)?;
    let _lock = FileLock::lock(run_base.join("lock")).map_err(RunError::Lock)?;

    if args.clean && run_base.exists() {
        for entry in fs::read_dir(&run_base)? {
            let entry = entry?;
            if entry.file_name() != "lock" {
                fs::remove_dir_all(entry.path()).ok();
            }
        }
    }

    let shell_command = args.shell_command.clone();
    let keep_going = args.keep_going;
    let readonly = args.readonly;
    let store = workspace_command.repo().store().clone();

    // Workers send TUI messages via this channel.
    let (tx, rx) = mpsc::channel::<TuiMessage>();
    // Workers store new trees here for post-run rewriting.
    let new_trees: Arc<std::sync::Mutex<HashMap<CommitId, MergedTree>>> =
        Arc::new(std::sync::Mutex::new(HashMap::new()));
    // Workers store errors here for post-run reporting.
    let errors: Arc<std::sync::Mutex<Vec<RunError>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let should_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let queue: Arc<std::sync::Mutex<std::vec::IntoIter<Commit>>> =
        Arc::new(std::sync::Mutex::new(resolved_commits.clone().into_iter()));

    // Spawn worker manager in a background thread.
    let worker_handle = {
        let tx = tx.clone();
        let queue = queue.clone();
        let should_stop = should_stop.clone();
        let new_trees = new_trees.clone();
        let errors = errors.clone();
        let store = store.clone();
        let run_base = run_base.clone();
        let base_ignores = base_ignores.clone();
        let tree_state_settings = tree_state_settings.clone();
        let shell_command = shell_command.clone();
        std::thread::spawn(move || {
            std::thread::scope(|scope| {
                for worker_id in 0..jobs {
                    let queue = &queue;
                    let should_stop = &should_stop;
                    let new_trees = &new_trees;
                    let errors = &errors;
                    let store = &store;
                    let run_base = &run_base;
                    let base_ignores = &base_ignores;
                    let tree_state_settings = &tree_state_settings;
                    let tx = tx.clone();
                    let shell_command = &shell_command;
                    scope.spawn(move || {
                        let worker_dir = run_base.join(format!("worker-{worker_id}"));
                        let wc_dir = worker_dir.join("working_copy");
                        let state_dir = worker_dir.join("state");

                        let ts = init_or_load_tree_state(
                            store.clone(), &wc_dir, &state_dir, tree_state_settings,
                        );
                        let mut ts = match ts {
                            Ok(ts) => ts,
                            Err(e) => {
                                should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                                errors.lock().unwrap().push(e);
                                return;
                            }
                        };

                        loop {
                            if should_stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                            let commit = { queue.lock().unwrap().next() };
                            let Some(commit) = commit else { break };
                            let cid = commit.id().clone();

                            tx.send(TuiMessage::Started { commit_id: cid.clone(), worker: worker_id }).ok();

                            match run_command_on_commit(&mut ts, shell_command, &commit, base_ignores.clone(), readonly) {
                                Ok(out) => {
                                    let modified = out.new_tree.is_some();
                                    let diff_summary = if let Some(ref tree) = out.new_tree {
                                        use futures::StreamExt as _;
                                        pollster::FutureExt::block_on(async {
                                            let mut summary = String::new();
                                            let mut stream = commit.tree().diff_stream(tree, &EverythingMatcher);
                                            while let Some(entry) = stream.next().await {
                                                let path = entry.path.as_ref();
                                                if let Ok(diff) = &entry.values {
                                                    let kind = if diff.before.is_absent() {
                                                        "A"
                                                    } else if diff.after.is_absent() {
                                                        "D"
                                                    } else {
                                                        "M"
                                                    };
                                                    summary.push_str(&format!("{kind} {}\n", path.as_internal_file_string()));
                                                }
                                            }
                                            summary
                                        })
                                    } else {
                                        String::new()
                                    };
                                    if let Some(tree) = out.new_tree {
                                        new_trees.lock().unwrap().insert(cid.clone(), tree);
                                    }
                                    tx.send(TuiMessage::Passed {
                                        commit_id: cid, modified, stdout: out.stdout, stderr: out.stderr, diff_summary,
                                    }).ok();
                                }
                                Err(e) => {
                                    let (stdout, stderr) = match &e {
                                        RunError::CommandFailure { stdout, stderr, .. } => (stdout.clone(), stderr.clone()),
                                        _ => (Vec::new(), Vec::new()),
                                    };
                                    tx.send(TuiMessage::Failed {
                                        commit_id: cid, message: e.to_string(), stdout, stderr,
                                    }).ok();
                                    if !keep_going {
                                        should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    errors.lock().unwrap().push(e);
                                }
                            }
                        }
                        ts.save().ok();
                    });
                }
            });
            tx.send(TuiMessage::AllDone).ok();
        })
    };

    let use_tui = io::stdout().is_terminal();
    let mut confirmed = true; // Non-TUI always confirms.

    if use_tui {
        io::stdout().execute(EnterAlternateScreen)?;
        enable_raw_mode()?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        terminal.clear()?;

        let mut tui_state = TuiState::new(&resolved_commits);
        run_tui_loop(&mut terminal, &mut tui_state, &rx, &shell_command);

        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;

        confirmed = matches!(tui_state.outcome, Some(TuiOutcome::Confirm));
    } else {
        // Non-interactive: just wait for all workers, printing status lines.
        for msg in &rx {
            match &msg {
                TuiMessage::Passed { commit_id, modified, .. } => {
                    let m = if *modified { " (modified)" } else { "" };
                    writeln!(ui.status(), "  ✓ {}{m}", &commit_id.hex()[..12])?;
                }
                TuiMessage::Failed { commit_id, message, .. } => {
                    writeln!(ui.status(), "  ✗ {} {message}", &commit_id.hex()[..12])?;
                }
                TuiMessage::AllDone => break,
                TuiMessage::Started { .. } => {}
            }
        }
    }

    // Signal workers to stop if user quit early, then wait.
    should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    worker_handle.join().ok();

    // Post-run summary.
    let rewritten = Arc::try_unwrap(new_trees).unwrap().into_inner().unwrap();
    let failures = Arc::try_unwrap(errors).unwrap().into_inner().unwrap();

    if !confirmed {
        writeln!(ui.status(), "Discarded results (cancelled).")?;
        return Ok(());
    }

    if readonly {
        let passed = resolved_commits.len() - failures.len();
        writeln!(
            ui.status(),
            "Ran '{}' on {} commit(s) (readonly). {} passed, {} failed.",
            args.shell_command, resolved_commits.len(), passed, failures.len(),
        )?;
        return Ok(());
    }

    if rewritten.is_empty() && failures.is_empty() {
        writeln!(ui.status(), "No commits were rewritten (command did not modify any tracked files).")?;
        return Ok(());
    }

    if !rewritten.is_empty() {
        let mut tx = workspace_command.start_transaction();
        let mut count: u32 = 0;
        tx.repo_mut()
            .transform_descendants(
                resolved_commits.iter().ids().cloned().collect_vec(),
                async |rewriter| {
                    let old_id = rewriter.old_commit().id().clone();
                    if let Some(new_tree) = rewritten.get(&old_id) {
                        count += 1;
                        rewriter.rebase().await?.set_tree(new_tree.clone()).write().await?;
                    } else {
                        rewriter.rebase().await?.write().await?;
                    }
                    Ok(())
                },
            )
            .await?;

        writeln!(ui.status(), "Rewrote {count} commit(s) with '{}'.", args.shell_command)?;

        tx.finish(
            ui,
            format!("run: rewrite {count} commit(s) with '{}'", args.shell_command),
        )
        .await?;
    }

    if !failures.is_empty() {
        if keep_going {
            writeln!(ui.warning_default(), "{} commit(s) failed.", failures.len())?;
        } else {
            return Err(failures.into_iter().next().unwrap().into());
        }
    }

    Ok(())
}
