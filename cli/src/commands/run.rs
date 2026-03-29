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
use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;

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

#[derive(Debug, thiserror::Error)]
enum RunError {
    #[error("Failed to checkout commit: {0}")]
    FailedCheckout(#[from] CheckoutError),
    #[error("Command '{cmd}' failed with {status} for commit {commit}{}{}",
        if stdout.is_empty() { String::new() } else {
            format!("\n--- stdout ---\n{}", String::from_utf8_lossy(stdout))
        },
        if stderr.is_empty() { String::new() } else {
            format!("\n--- stderr ---\n{}", String::from_utf8_lossy(stderr))
        }
    )]
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

fn get_shell() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("cmd", "/c")
    } else {
        ("/bin/sh", "-c")
    }
}

/// The result of running a command on a single commit.
struct RunResult {
    commit_id: CommitId,
    change_id: String,
    new_tree: Option<MergedTree>,
    /// Captured stdout/stderr (only when running in parallel).
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Run the shell command using an existing TreeState. The TreeState is
/// incrementally updated to the commit's tree (only changed files are
/// written to disk), then the command runs, and the result is snapshotted.
fn run_command_on_commit(
    tree_state: &mut TreeState,
    shell_command: &str,
    commit: &Commit,
    base_ignores: Arc<GitIgnoreFile>,
    capture_output: bool,
    readonly: bool,
) -> Result<RunResult, RunError> {
    // Incremental checkout: only writes files that differ from the current tree.
    tree_state.check_out(&commit.tree())?;

    let (prog, first_arg) = get_shell();
    let mut cmd = Command::new(prog);
    cmd.arg(first_arg)
        .arg(shell_command)
        .current_dir(tree_state.working_copy_path())
        .env("JJ_CHANGE", commit.change_id().hex())
        .env("JJ_COMMIT", commit.id().hex());

    let (status, stdout, stderr) = if capture_output {
        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;
        (output.status, output.stdout, output.stderr)
    } else {
        let status = cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()?;
        (status, Vec::new(), Vec::new())
    };

    if !status.success() {
        return Err(RunError::CommandFailure {
            cmd: shell_command.to_owned(),
            status,
            commit: commit.id().clone(),
            stdout,
            stderr,
        });
    }

    let new_tree = if readonly {
        None
    } else {
        let options = SnapshotOptions {
            base_ignores,
            start_tracking_matcher: &EverythingMatcher,
            progress: None,
            max_new_file_size: 64_000_000, // 64 MB
            force_tracking_matcher: &NothingMatcher,
        };
        let (dirty, _stats) = pollster::FutureExt::block_on(tree_state.snapshot(&options))?;
        if dirty {
            Some(tree_state.current_tree().clone())
        } else {
            None
        }
    };

    Ok(RunResult {
        commit_id: commit.id().clone(),
        change_id: commit.change_id().hex(),
        new_tree,
        stdout,
        stderr,
    })
}

/// Ensure a directory exists, creating it and parents if needed.
fn ensure_dir(path: &PathBuf) -> Result<(), RunError> {
    if !path.exists() {
        fs::create_dir_all(path)
            .map_err(|e| RunError::PathCreation { path: path.clone(), source: e })?;
    }
    Ok(())
}

/// Initialize or load a TreeState for a worker directory. If state exists
/// from a previous run, load it so that check_out() can do an incremental
/// diff. Otherwise, init from scratch.
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
        Ok(TreeState::load(
            store,
            working_copy_dir.clone(),
            state_dir.clone(),
            settings,
        )?)
    } else {
        Ok(TreeState::init(
            store,
            working_copy_dir.clone(),
            state_dir.clone(),
            settings,
        )?)
    }
}

pub async fn cmd_run(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &RunArgs,
) -> Result<(), CommandError> {
    let mut workspace_command = command.workspace_helper(ui)?;
    // Commits are returned in reverse topological order.
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

    let base_ignores = workspace_command.base_ignores()?;
    let tree_state_settings =
        TreeStateSettings::try_from_user_settings(workspace_command.settings())?;

    // Worker directories persist across runs for incremental builds (ignored
    // files like bazel-out/ survive). Lock to prevent concurrent jj run.
    let repo_path = workspace_command.repo_path().to_owned();
    let run_base = repo_path.parent().unwrap().join("run");
    ensure_dir(&run_base)?;
    let _lock = FileLock::lock(run_base.join("lock")).map_err(RunError::Lock)?;

    if args.clean && run_base.exists() {
        // Remove worker dirs but keep the lock file.
        for entry in fs::read_dir(&run_base)? {
            let entry = entry?;
            if entry.file_name() != "lock" {
                fs::remove_dir_all(entry.path()).ok();
            }
        }
    }

    // Run the command on each commit. With -j 1, inherit stdout/stderr for
    // interactive use. With -j >1, capture output and print per-commit on
    // completion to avoid interleaving.
    //
    // Each worker thread gets its own TreeState that persists across commits
    // AND across invocations. check_out() is incremental — only files that
    // differ between the worker's current tree and the target commit's tree
    // are written to disk.
    let parallel = jobs > 1;
    let shell_command = &args.shell_command;
    let keep_going = args.keep_going;
    let store = workspace_command.repo().store().clone();
    let queue = std::sync::Mutex::new(resolved_commits.iter());
    let results = std::sync::Mutex::new(Vec::<Result<RunResult, RunError>>::new());
    // Workers check this flag to stop early on error (for stop/fatal strategies).
    let should_stop = std::sync::atomic::AtomicBool::new(false);

    std::thread::scope(|scope| {
        for worker_id in 0..jobs {
            let queue = &queue;
            let results = &results;
            let should_stop = &should_stop;
            let store = &store;
            let run_base = &run_base;
            let base_ignores = &base_ignores;
            let tree_state_settings = &tree_state_settings;
            scope.spawn(move || {
                let worker_dir = run_base.join(format!("worker-{worker_id}"));
                let working_copy_dir = worker_dir.join("working_copy");
                let state_dir = worker_dir.join("state");

                let tree_state = init_or_load_tree_state(
                    store.clone(),
                    &working_copy_dir,
                    &state_dir,
                    tree_state_settings,
                );
                let mut tree_state = match tree_state {
                    Ok(ts) => ts,
                    Err(e) => {
                        should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        results.lock().unwrap().push(Err(e));
                        return;
                    }
                };

                loop {
                    if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let commit = {
                        let mut q = queue.lock().unwrap();
                        q.next()
                    };
                    let Some(commit) = commit else { break };

                    let result = run_command_on_commit(
                        &mut tree_state,
                        shell_command,
                        commit,
                        base_ignores.clone(),
                        parallel,
                        args.readonly,
                    );

                    let is_err = result.is_err();
                    results.lock().unwrap().push(result);

                    if is_err && !keep_going {
                        should_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                }

                tree_state.save().ok();
            });
        }
    });

    // Collect results, printing captured output.
    let all_results = results.into_inner().unwrap();
    let commit_count = all_results.len();
    let mut rewritten: HashMap<CommitId, MergedTree> = HashMap::new();
    let mut failures: Vec<RunError> = Vec::new();
    for result in all_results {
        match result {
            Ok(run_result) => {
                if parallel
                    && (!run_result.stdout.is_empty() || !run_result.stderr.is_empty())
                {
                    let short_change = &run_result.change_id[..12];
                    writeln!(ui.status(), "--- {} ---", short_change)?;
                    if !run_result.stdout.is_empty() {
                        ui.stdout_formatter().write_all(&run_result.stdout)?;
                    }
                    if !run_result.stderr.is_empty() {
                        ui.stderr_formatter().write_all(&run_result.stderr)?;
                    }
                }
                if let Some(tree) = run_result.new_tree {
                    rewritten.insert(run_result.commit_id, tree);
                }
            }
            Err(e) => {
                if keep_going {
                    writeln!(ui.warning_default(), "{e}")?;
                    failures.push(e);
                } else {
                    // Stop/Fatal: propagate the first error immediately.
                    return Err(e.into());
                }
            }
        }
    }

    if args.readonly {
        writeln!(
            ui.status(),
            "Ran '{}' on {commit_count} commit(s) (readonly).",
            args.shell_command
        )?;
        return Ok(());
    }

    if rewritten.is_empty() {
        writeln!(
            ui.status(),
            "No commits were rewritten (command did not modify any tracked files)."
        )?;
        return Ok(());
    }

    // Rewrite the commits with their new trees.
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

    if !failures.is_empty() {
        writeln!(
            ui.warning_default(),
            "{} commit(s) failed.",
            failures.len()
        )?;
    }

    Ok(())
}
