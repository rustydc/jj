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

//! Implementation of `jj run` — run a command across a set of revisions.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::IsTerminal as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
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
use jj_lib::copies::CopyRecords;
use jj_lib::diff_presentation::LineCompareMode;
use jj_lib::gitignore::GitIgnoreFile;
use jj_lib::local_working_copy::TreeState;
use jj_lib::local_working_copy::TreeStateError;
use jj_lib::local_working_copy::TreeStateSettings;
use jj_lib::lock::FileLock;
use jj_lib::matchers::EverythingMatcher;
use jj_lib::matchers::NothingMatcher;
use jj_lib::merge::Diff;
use jj_lib::merged_tree::MergedTree;
use jj_lib::object_id::ObjectId as _;
use jj_lib::repo::Repo as _;
use jj_lib::working_copy::CheckoutError;
use jj_lib::working_copy::SnapshotError;
use jj_lib::working_copy::SnapshotOptions;
use ratatui::Terminal;
use ratatui::layout::Constraint;
use ratatui::layout::Layout;
use ratatui::layout::Spacing;
use ratatui::prelude::CrosstermBackend;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::symbols::merge::MergeStrategy;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::List;
use ratatui::widgets::ListState;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Scrollbar;
use ratatui::widgets::ScrollbarOrientation;
use ratatui::widgets::ScrollbarState;
use ratatui::widgets::Wrap;

use crate::cli_util::CommandHelper;
use crate::cli_util::RevisionArg;
use crate::cli_util::WorkspaceCommandHelper;
use crate::command_error::CommandError;
use crate::diff_util;
use crate::ui::Ui;

// =============================================================================
// Command definition
// =============================================================================

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

    /// Remove cached working copies before running.
    #[arg(long)]
    clean: bool,

    /// Don't rewrite commits; just run the command and report results.
    #[arg(long)]
    readonly: bool,

    /// Keep going even if the command fails on some revisions.
    #[arg(long, short = 'k')]
    keep_going: bool,
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, thiserror::Error)]
enum RunError {
    #[error("Failed to checkout commit: {0}")]
    FailedCheckout(#[from] CheckoutError),
    #[error("Command '{cmd}' failed with {status} for commit {commit}")]
    CommandFailure {
        cmd: String,
        status: std::process::ExitStatus,
        commit: CommitId,
        new_tree: Option<MergedTree>,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("Failed to create path {}: {source}", path.display())]
    PathCreation { path: PathBuf, source: io::Error },
    #[error("Failed to acquire lock: {0}")]
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

// =============================================================================
// Worker execution
// =============================================================================

fn get_shell() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("cmd", "/c")
    } else {
        ("/bin/sh", "-c")
    }
}

fn init_or_load_tree_state(
    store: Arc<jj_lib::store::Store>,
    working_copy_dir: &PathBuf,
    state_dir: &PathBuf,
    settings: &TreeStateSettings,
) -> Result<TreeState, RunError> {
    fs::create_dir_all(working_copy_dir)
        .map_err(|e| RunError::PathCreation { path: working_copy_dir.clone(), source: e })?;
    fs::create_dir_all(state_dir)
        .map_err(|e| RunError::PathCreation { path: state_dir.clone(), source: e })?;
    if state_dir.join("tree_state").exists() {
        Ok(TreeState::load(store, working_copy_dir.clone(), state_dir.clone(), settings)?)
    } else {
        Ok(TreeState::init(store, working_copy_dir.clone(), state_dir.clone(), settings)?)
    }
}

fn make_snapshot_options(base_ignores: Arc<GitIgnoreFile>) -> SnapshotOptions<'static> {
    SnapshotOptions {
        base_ignores,
        start_tracking_matcher: &EverythingMatcher,
        progress: None,
        max_new_file_size: 64_000_000,
        force_tracking_matcher: &NothingMatcher,
    }
}

/// Spawn a thread that reads from `reader` line-by-line and sends tagged
/// output lines to the TUI channel.
fn spawn_pipe_reader(
    reader: impl io::Read + Send + 'static,
    commit_id: CommitId,
    tx: mpsc::Sender<TuiMessage>,
    make_line: fn(Vec<u8>) -> OutputLine,
) -> std::thread::JoinHandle<()> {
    use std::io::BufRead;
    std::thread::spawn(move || {
        for line in io::BufReader::new(reader).split(b'\n') {
            if let Ok(data) = line {
                tx.send(TuiMessage::Output {
                    commit_id: commit_id.clone(),
                    line: make_line(data),
                })
                .ok();
            }
        }
    })
}

/// Run a shell command on a commit. Streams output to the TUI, periodically
/// snapshots for live diff stats, and returns the new tree (if modified).
fn run_command_on_commit(
    tree_state: &mut TreeState,
    shell_command: &str,
    commit: &Commit,
    base_ignores: Arc<GitIgnoreFile>,
    readonly: bool,
    tui_tx: &mpsc::Sender<TuiMessage>,
    new_trees: &Mutex<HashMap<CommitId, MergedTree>>,
) -> Result<Option<MergedTree>, RunError> {
    tree_state.check_out(&commit.tree())?;

    let cid = commit.id().clone();
    let (prog, first_arg) = get_shell();
    let mut child = Command::new(prog)
        .arg(first_arg)
        .arg(shell_command)
        .current_dir(tree_state.working_copy_path())
        .env("JJ_CHANGE", commit.change_id().hex())
        .env("JJ_COMMIT", commit.id().hex())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let h1 = spawn_pipe_reader(
        child.stdout.take().unwrap(), cid.clone(), tui_tx.clone(), OutputLine::Stdout,
    );
    let h2 = spawn_pipe_reader(
        child.stderr.take().unwrap(), cid.clone(), tui_tx.clone(), OutputLine::Stderr,
    );

    // Periodically snapshot while the command runs, for live diff stats.
    let original_tree = commit.tree();
    let snap_opts = make_snapshot_options(base_ignores);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if !readonly {
            let (dirty, _) = pollster::FutureExt::block_on(tree_state.snapshot(&snap_opts))?;
            if dirty {
                let current = tree_state.current_tree().clone();
                let stat = compute_stat(&original_tree, &current);
                new_trees.lock().unwrap().insert(cid.clone(), current);
                tui_tx
                    .send(TuiMessage::DiffStat { commit_id: cid.clone(), stat })
                    .ok();
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    };

    h1.join().ok();
    h2.join().ok();

    // Final snapshot.
    let new_tree = if !readonly {
        let (dirty, _) = pollster::FutureExt::block_on(tree_state.snapshot(&snap_opts))?;
        if dirty { Some(tree_state.current_tree().clone()) } else { None }
    } else {
        None
    };

    if !status.success() {
        return Err(RunError::CommandFailure {
            cmd: shell_command.to_owned(),
            status,
            commit: commit.id().clone(),
            new_tree,
        });
    }

    Ok(new_tree)
}

// =============================================================================
// Diff / stat helpers
// =============================================================================

/// Compute line-level stat string: `[N:+A-R]` (files, lines added/removed).
fn compute_stat(before: &MergedTree, after: &MergedTree) -> String {
    let store = before.store();
    let options = diff_util::DiffStatOptions {
        line_diff: diff_util::LineDiffOptions {
            compare_mode: LineCompareMode::Exact,
        },
    };
    let result = pollster::FutureExt::block_on(async {
        let copy_records = CopyRecords::default();
        let tree_diff =
            before.diff_stream_with_copies(after, &EverythingMatcher, &copy_records);
        diff_util::DiffStats::calculate(
            store,
            tree_diff,
            &options,
            jj_lib::conflicts::ConflictMarkerStyle::Snapshot,
        )
        .await
    });
    let Ok(stats) = result else {
        return String::new();
    };
    let files = stats.entries().len();
    if files == 0 {
        return String::new();
    }
    let added = stats.count_total_added();
    let removed = stats.count_total_removed();
    let mut s = format!("[{files}:");
    if added > 0 {
        s.push_str(&format!("+{added}"));
    }
    if removed > 0 {
        s.push_str(&format!("-{removed}"));
    }
    if added == 0 && removed == 0 {
        s.push('~');
    }
    s.push(']');
    s
}

/// Render a full diff to bytes using jj's built-in color-words diff, and
/// compute the line stat.
fn compute_diff(
    ui: &Ui,
    workspace_command: &WorkspaceCommandHelper,
    before: &MergedTree,
    after: &MergedTree,
) -> (Vec<u8>, String) {
    let diff_bytes = (|| -> Result<Vec<u8>, CommandError> {
        let options =
            diff_util::ColorWordsDiffOptions::from_settings(workspace_command.settings())?;
        let formats = vec![diff_util::DiffFormat::ColorWords(Box::new(options))];
        let diff_renderer = workspace_command.diff_renderer(formats);
        let mut buf = Vec::new();
        {
            let mut formatter = ui.new_formatter(&mut buf);
            pollster::FutureExt::block_on(diff_renderer.show_diff(
                ui,
                formatter.as_mut(),
                Diff::new(before, after),
                &EverythingMatcher,
                &CopyRecords::default(),
                80,
            ))?;
        }
        Ok(buf)
    })()
    .unwrap_or_else(|e| format!("(error: {e:?})").into_bytes());

    let stat = compute_stat(before, after);
    (diff_bytes, stat)
}

/// Look up a commit's new tree and compute its diff + stat, storing the
/// results in the entry. Returns true if a diff was computed.
fn ensure_diff_rendered(
    entry: &mut CommitEntry,
    commits: &[Commit],
    new_trees: &Mutex<HashMap<CommitId, MergedTree>>,
    ui: &Ui,
    workspace_command: &WorkspaceCommandHelper,
) -> bool {
    let new_tree = new_trees.lock().unwrap().get(&entry.commit_id).cloned();
    if let Some(new_tree) = new_tree {
        if let Some(commit) = commits.iter().find(|c| c.id() == &entry.commit_id) {
            let (diff_bytes, stat) =
                compute_diff(ui, workspace_command, &commit.tree(), &new_tree);
            entry.diff_bytes = Some(diff_bytes);
            entry.diff_stat = Some(stat);
            return true;
        }
    }
    false
}

// =============================================================================
// TUI types
// =============================================================================

/// A line of output tagged with its source.
#[derive(Clone, Debug)]
enum OutputLine {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
}

/// Message from worker → TUI.
enum TuiMessage {
    Started { commit_id: CommitId, worker: usize },
    Output { commit_id: CommitId, line: OutputLine },
    DiffStat { commit_id: CommitId, stat: String },
    Passed { commit_id: CommitId, modified: bool },
    Failed { commit_id: CommitId, message: String },
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
    output: Vec<OutputLine>,
    diff_bytes: Option<Vec<u8>>,
    diff_stat: Option<String>,
}

/// Which bottom panel has focus for scrolling.
#[derive(Clone, Copy, PartialEq)]
enum FocusPanel {
    Output,
    Diff,
}

enum TuiOutcome {
    Confirm,
    Cancel,
}

struct TuiState {
    commits: Vec<CommitEntry>,
    index: HashMap<CommitId, usize>,
    selected: usize,
    focus: FocusPanel,
    output_scroll: u16,
    diff_scroll: u16,
    done: bool,
    outcome: Option<TuiOutcome>,
    passed: usize,
    failed: usize,
}

impl TuiState {
    fn new(commits: &[Commit]) -> Self {
        let entries: Vec<CommitEntry> = commits
            .iter()
            .map(|c| CommitEntry {
                commit_id: c.id().clone(),
                change_id_hex: c.change_id().hex(),
                description: c.description().lines().next().unwrap_or("").to_string(),
                status: JobStatus::Pending,
                output: Vec::new(),
                diff_bytes: None,
                diff_stat: None,
            })
            .collect();
        let index = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.commit_id.clone(), i))
            .collect();
        TuiState {
            commits: entries,
            index,
            selected: 0,
            focus: FocusPanel::Output,
            output_scroll: 0,
            diff_scroll: 0,
            done: false,
            outcome: None,
            passed: 0,
            failed: 0,
        }
    }

    fn find_mut(&mut self, id: &CommitId) -> Option<&mut CommitEntry> {
        self.index.get(id).copied().map(|i| &mut self.commits[i])
    }

    fn handle_message(&mut self, msg: TuiMessage) {
        match msg {
            TuiMessage::Started { commit_id, worker } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.status = JobStatus::Running(worker);
                }
            }
            TuiMessage::DiffStat { commit_id, stat } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.diff_stat = Some(stat);
                }
            }
            TuiMessage::Output { commit_id, line } => {
                let is_selected = self
                    .commits
                    .get(self.selected)
                    .is_some_and(|c| c.commit_id == commit_id);
                if let Some(e) = self.find_mut(&commit_id) {
                    e.output.push(line);
                }
                if is_selected {
                    self.output_scroll = u16::MAX; // auto-follow
                }
            }
            TuiMessage::Passed { commit_id, modified } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.status = JobStatus::Passed { modified };
                }
                self.passed += 1;
            }
            TuiMessage::Failed { commit_id, message } => {
                if let Some(e) = self.find_mut(&commit_id) {
                    e.status = JobStatus::Failed(message);
                }
                self.failed += 1;
            }
            TuiMessage::AllDone => {
                self.done = true;
            }
        }
    }
}

// =============================================================================
// TUI rendering
// =============================================================================

/// Render an OutputLine to ratatui Lines, preserving ANSI colors. Stderr
/// lines without ANSI styling are rendered in red.
fn render_output_line(ol: &OutputLine) -> Vec<Line<'static>> {
    let (data, is_stderr) = match ol {
        OutputLine::Stdout(d) => (d, false),
        OutputLine::Stderr(d) => (d, true),
    };
    if let Ok(text) = ansi_to_tui::IntoText::into_text(data) {
        if is_stderr {
            text.lines
                .into_iter()
                .map(|line| {
                    let has_style =
                        line.spans.iter().any(|s| s.style != Style::default());
                    if has_style {
                        line
                    } else {
                        Line::from(
                            line.spans
                                .into_iter()
                                .map(|s| Span::styled(s.content, s.style.fg(Color::Red)))
                                .collect::<Vec<_>>(),
                        )
                    }
                })
                .collect()
        } else {
            text.lines
        }
    } else {
        let s = String::from_utf8_lossy(data).to_string();
        let style = if is_stderr {
            Style::default().fg(Color::Red)
        } else {
            Style::default()
        };
        vec![Line::styled(s, style)]
    }
}

/// Render a scrollable paragraph panel with a scrollbar.
fn render_panel(
    frame: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    text: ratatui::text::Text<'_>,
    scroll: u16,
    block: Block<'_>,
) -> u16 {
    let inner = block.inner(area);
    let line_count = text.lines.len() as u16;
    let panel_height = inner.height;
    let max_scroll = line_count.saturating_sub(panel_height);
    let scroll = scroll.min(max_scroll);
    let p = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(p, area);

    if line_count > panel_height {
        let sb_area = ratatui::layout::Rect {
            x: area.x + area.width - 1,
            y: inner.y,
            width: 1,
            height: panel_height,
        };
        let mut sb_state = ScrollbarState::new(max_scroll as usize).position(scroll as usize);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            sb_area,
            &mut sb_state,
        );
    }
    max_scroll
}

fn render_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &TuiState,
    shell_command: &str,
) -> io::Result<()> {
    terminal.draw(|frame| {
        // Shrink commit list to fit content + 1 line padding, but at least 3 lines.
        let commit_rows = (state.commits.len() as u16 + 1).max(3);

        // Vertical: commits | bottom panels | help
        let [commits_area, bottom_area, help_area] =
            Layout::vertical([
                Constraint::Length(commit_rows + 2), // +2 for borders
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .spacing(Spacing::Overlap(1))
            .areas(frame.area());

        // Check if the selected commit has a diff to show.
        let selected_has_diff = state.commits.get(state.selected)
            .and_then(|e| e.diff_bytes.as_ref())
            .is_some_and(|b| !b.is_empty());

        // Commit list title includes the run status.
        let items: Vec<Line> = state
            .commits
            .iter()
            .enumerate()
            .map(|(i, e)| {
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
                let mut spans = vec![
                    Span::raw(sel),
                    Span::styled(format!("{sym} "), style),
                    Span::styled(format!("{short} "), Style::default().fg(Color::Magenta)),
                    Span::styled(desc.to_string(), style),
                ];
                if let Some(ref stat) = e.diff_stat {
                    if !stat.is_empty() {
                        spans.push(Span::raw(" "));
                        spans.push(Span::styled(stat.clone(), Style::default().fg(Color::Cyan)));
                    }
                }
                Line::from(spans)
            })
            .collect();

        let done_count = state.passed + state.failed;
        let total = state.commits.len();
        let mut title_spans = vec![
            Span::styled(
                format!(" jj run '{shell_command}' "),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("[{done_count}/{total}] ")),
            Span::styled(format!("{} passed", state.passed), Style::default().fg(Color::Green)),
        ];
        if state.failed > 0 {
            title_spans.push(Span::styled(
                format!(" {} failed", state.failed),
                Style::default().fg(Color::Red),
            ));
        }
        title_spans.push(Span::raw(" "));
        let commits_block = Block::bordered()
            .title(Line::from(title_spans))
            .merge_borders(MergeStrategy::Exact);
        let list = List::new(items).block(commits_block);
        let mut list_state = ListState::default().with_selected(Some(state.selected));
        frame.render_stateful_widget(list, commits_area, &mut list_state);

        // Output (always shown) and diff (only if selected commit has changes).
        let e = &state.commits[state.selected];
        let mut output_lines: Vec<Line> = Vec::new();
        for ol in &e.output {
            output_lines.extend(render_output_line(ol));
        }
        if output_lines.is_empty() {
            output_lines.push(Line::styled("(no output)", Style::default().fg(Color::DarkGray)));
        }

        if selected_has_diff {
            let [output_area, diff_area] = Layout::horizontal([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .spacing(Spacing::Overlap(1))
            .areas(bottom_area);

            let output_title = if state.focus == FocusPanel::Output { " output ◀ " } else { " output " };
            let output_block = Block::bordered()
                .title(output_title)
                .merge_borders(MergeStrategy::Exact);
            render_panel(
                frame, output_area,
                ratatui::text::Text::from(output_lines),
                state.output_scroll,
                output_block,
            );

            let diff_text = if let Some(ref bytes) = e.diff_bytes {
                if let Ok(t) = ansi_to_tui::IntoText::into_text(bytes) {
                    t
                } else {
                    ratatui::text::Text::raw(String::from_utf8_lossy(bytes).to_string())
                }
            } else {
                ratatui::text::Text::raw("")
            };
            let diff_title = if state.focus == FocusPanel::Diff { " diff ◀ " } else { " diff " };
            let diff_block = Block::bordered()
                .title(diff_title)
                .merge_borders(MergeStrategy::Exact);
            render_panel(
                frame, diff_area,
                diff_text,
                state.diff_scroll,
                diff_block,
            );
        } else {
            // No diff — output takes the full width.
            let output_block = Block::bordered()
                .title(" output ")
                .merge_borders(MergeStrategy::Exact);
            render_panel(
                frame, bottom_area,
                ratatui::text::Text::from(output_lines),
                state.output_scroll,
                output_block,
            );
        }

        // Help bar
        let mut help_spans = vec![
            Span::styled("↑/↓", Style::default().fg(Color::Magenta)),
            Span::raw(" nav "),
            Span::styled("Tab", Style::default().fg(Color::Magenta)),
            Span::raw(" focus "),
            Span::styled("PgUp/Dn", Style::default().fg(Color::Magenta)),
            Span::raw(" scroll "),
        ];
        if state.done {
            help_spans.extend([
                Span::styled("c", Style::default().fg(Color::Green)),
                Span::raw(" confirm "),
                Span::styled("q", Style::default().fg(Color::Red)),
                Span::raw(" discard"),
            ]);
        } else {
            help_spans.extend([
                Span::styled("q", Style::default().fg(Color::Magenta)),
                Span::raw(" cancel"),
            ]);
        }
        frame.render_widget(Line::from(help_spans), help_area);
    })?;
    Ok(())
}

// =============================================================================
// TUI event loop
// =============================================================================

fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    rx: &mpsc::Receiver<TuiMessage>,
    shell_command: &str,
    ui: &Ui,
    workspace_command: &WorkspaceCommandHelper,
    commits: &[Commit],
    new_trees: &Mutex<HashMap<CommitId, MergedTree>>,
) {
    loop {
        render_tui(terminal, state, shell_command).ok();

        // Drain worker messages, compute diffs for finished/updated commits.
        while let Ok(msg) = rx.try_recv() {
            let render_diff_for = match &msg {
                TuiMessage::Passed { modified: true, commit_id, .. }
                | TuiMessage::Failed { commit_id, .. } => Some(commit_id.clone()),
                TuiMessage::DiffStat { commit_id, .. }
                    if state.commits.get(state.selected)
                        .is_some_and(|c| c.commit_id == *commit_id) =>
                {
                    Some(commit_id.clone())
                }
                _ => None,
            };

            state.handle_message(msg);

            if let Some(cid) = render_diff_for {
                if let Some(entry) = state.find_mut(&cid) {
                    ensure_diff_rendered(entry, commits, new_trees, ui, workspace_command);
                }
            }
        }

        // Ensure the selected commit's diff is always rendered.
        let entry = &mut state.commits[state.selected];
        if entry.diff_bytes.is_none() {
            ensure_diff_rendered(entry, commits, new_trees, ui, workspace_command);
        }

        // Poll keyboard.
        if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = crossterm::event::read() {
                if key.is_release() {
                    continue;
                }
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
                        if state.selected + 1 < state.commits.len() {
                            state.selected += 1;
                            state.output_scroll = u16::MAX; // auto-follow new selection
                            state.diff_scroll = 0;
                        }
                    }
                    (KeyCode::Up | KeyCode::Char('k'), KeyModifiers::NONE) => {
                        if state.selected > 0 {
                            state.selected -= 1;
                            state.output_scroll = u16::MAX;
                            state.diff_scroll = 0;
                        }
                    }
                    (KeyCode::Tab, _) => {
                        state.focus = match state.focus {
                            FocusPanel::Output => FocusPanel::Diff,
                            FocusPanel::Diff => FocusPanel::Output,
                        };
                    }
                    (KeyCode::PageDown, _) => {
                        match state.focus {
                            FocusPanel::Output => state.output_scroll = state.output_scroll.saturating_add(10),
                            FocusPanel::Diff => state.diff_scroll = state.diff_scroll.saturating_add(10),
                        }
                    }
                    (KeyCode::PageUp, _) => {
                        match state.focus {
                            FocusPanel::Output => state.output_scroll = state.output_scroll.saturating_sub(10),
                            FocusPanel::Diff => state.diff_scroll = state.diff_scroll.saturating_sub(10),
                        }
                    }
                    _ => {}
                }
            }
        }

        if state.outcome.is_some() {
            return;
        }
    }
}

// =============================================================================
// Main command
// =============================================================================

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
    let store = workspace_command.repo().store().clone();

    // Lock the run directory to prevent concurrent jj run.
    let repo_path = workspace_command.repo_path().to_owned();
    let run_base = repo_path.parent().unwrap().join("run");
    fs::create_dir_all(&run_base)?;
    let _lock = FileLock::lock(run_base.join("lock")).map_err(RunError::Lock)?;

    if args.clean {
        for entry in fs::read_dir(&run_base)? {
            let entry = entry?;
            if entry.file_name() != "lock" {
                fs::remove_dir_all(entry.path()).ok();
            }
        }
    }

    // Shared state between workers and TUI.
    let (tx, rx) = mpsc::channel::<TuiMessage>();
    let new_trees: Arc<Mutex<HashMap<CommitId, MergedTree>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let errors: Arc<Mutex<Vec<RunError>>> = Arc::new(Mutex::new(Vec::new()));
    let should_stop = Arc::new(AtomicBool::new(false));
    let queue: Arc<Mutex<std::vec::IntoIter<Commit>>> =
        Arc::new(Mutex::new(resolved_commits.clone().into_iter()));

    let shell_command = args.shell_command.clone();
    let keep_going = args.keep_going;
    let readonly = args.readonly;

    // Spawn workers in a background thread.
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

                        let mut ts = match init_or_load_tree_state(
                            store.clone(),
                            &wc_dir,
                            &state_dir,
                            tree_state_settings,
                        ) {
                            Ok(ts) => ts,
                            Err(e) => {
                                should_stop.store(true, Ordering::Relaxed);
                                errors.lock().unwrap().push(e);
                                return;
                            }
                        };

                        loop {
                            if should_stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let commit = { queue.lock().unwrap().next() };
                            let Some(commit) = commit else { break };
                            let cid = commit.id().clone();

                            tx.send(TuiMessage::Started {
                                commit_id: cid.clone(),
                                worker: worker_id,
                            })
                            .ok();

                            match run_command_on_commit(
                                &mut ts,
                                shell_command,
                                &commit,
                                base_ignores.clone(),
                                readonly,
                                &tx,
                                new_trees,
                            ) {
                                Ok(new_tree) => {
                                    let modified = new_tree.is_some();
                                    if let Some(tree) = new_tree {
                                        new_trees.lock().unwrap().insert(cid.clone(), tree);
                                    }
                                    tx.send(TuiMessage::Passed {
                                        commit_id: cid,
                                        modified,
                                    })
                                    .ok();
                                }
                                Err(e) => {
                                    if let RunError::CommandFailure {
                                        new_tree: Some(ref tree),
                                        ..
                                    } = e
                                    {
                                        new_trees
                                            .lock()
                                            .unwrap()
                                            .insert(cid.clone(), tree.clone());
                                    }
                                    tx.send(TuiMessage::Failed {
                                        commit_id: cid,
                                        message: e.to_string(),
                                    })
                                    .ok();
                                    if !keep_going {
                                        should_stop.store(true, Ordering::Relaxed);
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

    // TUI or text output.
    let use_tui = io::stdout().is_terminal();
    let mut confirmed = true;

    if use_tui {
        io::stdout().execute(EnterAlternateScreen)?;
        enable_raw_mode()?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        terminal.clear()?;

        let mut tui_state = TuiState::new(&resolved_commits);
        run_tui_loop(
            &mut terminal,
            &mut tui_state,
            &rx,
            &shell_command,
            ui,
            &workspace_command,
            &resolved_commits,
            &new_trees,
        );

        disable_raw_mode()?;
        io::stdout().execute(LeaveAlternateScreen)?;

        confirmed = matches!(tui_state.outcome, Some(TuiOutcome::Confirm));
    } else {
        for msg in &rx {
            match &msg {
                TuiMessage::Passed {
                    commit_id,
                    modified,
                    ..
                } => {
                    let m = if *modified { " (modified)" } else { "" };
                    writeln!(ui.status(), "  ✓ {}{m}", &commit_id.hex()[..12])?;
                }
                TuiMessage::Failed {
                    commit_id, message, ..
                } => {
                    writeln!(ui.status(), "  ✗ {} {message}", &commit_id.hex()[..12])?;
                }
                TuiMessage::AllDone => break,
                TuiMessage::Started { .. }
                | TuiMessage::Output { .. }
                | TuiMessage::DiffStat { .. } => {}
            }
        }
    }

    should_stop.store(true, Ordering::Relaxed);
    worker_handle.join().ok();

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
            args.shell_command,
            resolved_commits.len(),
            passed,
            failures.len(),
        )?;
        return Ok(());
    }

    if rewritten.is_empty() && failures.is_empty() {
        writeln!(
            ui.status(),
            "No commits were rewritten (command did not modify any tracked files)."
        )?;
        return Ok(());
    }

    // Only rewrite commits that passed (failures are left in `rewritten` for
    // diff display but should not be committed).
    let failed_ids: std::collections::HashSet<_> = failures
        .iter()
        .filter_map(|e| match e {
            RunError::CommandFailure { commit, .. } => Some(commit.clone()),
            _ => None,
        })
        .collect();

    let rewritten: HashMap<_, _> = rewritten
        .into_iter()
        .filter(|(id, _)| !failed_ids.contains(id))
        .collect();

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
                        rewriter
                            .rebase()
                            .await?
                            .set_tree(new_tree.clone())
                            .write()
                            .await?;
                    } else {
                        rewriter.rebase().await?.write().await?;
                    }
                    Ok(())
                },
            )
            .await?;

        writeln!(
            ui.status(),
            "Rewrote {count} commit(s) with '{}'.",
            args.shell_command
        )?;

        tx.finish(
            ui,
            format!(
                "run: rewrite {count} commit(s) with '{}'",
                args.shell_command
            ),
        )
        .await?;
    }

    if !failures.is_empty() {
        if keep_going {
            writeln!(
                ui.warning_default(),
                "{} commit(s) failed.",
                failures.len()
            )?;
        } else {
            return Err(failures.into_iter().next().unwrap().into());
        }
    }

    Ok(())
}
