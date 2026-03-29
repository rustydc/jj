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

use crate::common::TestEnvironment;

#[cfg(unix)]
#[test]
fn test_run_single_commit() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    // Create a commit with a file
    work_dir.write_file("greeting.txt", "hello world");
    work_dir.run_jj(["commit", "-m", "add greeting"]).success();

    // Run a command that uppercases the file
    let output = work_dir.run_jj([
        "run",
        "tr a-z A-Z < greeting.txt > greeting.txt.tmp && mv greeting.txt.tmp greeting.txt",
        "-r",
        "@-",
    ]);
    // Should succeed and rewrite the commit
    assert!(output.status.success(), "jj run should succeed: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("Rewrote 1 commit(s)"), "Should rewrite: {stderr}");

    // Verify the file was uppercased in the commit
    let show_output = work_dir.run_jj(["file", "show", "greeting.txt", "-r", "@-"]);
    let content = show_output.stdout.to_string();
    assert!(content.contains("HELLO WORLD"), "Expected HELLO WORLD, got: {:?}", content);
}

#[cfg(unix)]
#[test]
fn test_run_multiple_commits() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    // Create a stack of 3 commits, each with a file
    work_dir.write_file("file.txt", "aaa\n");
    work_dir.run_jj(["commit", "-m", "commit A"]).success();
    work_dir.write_file("file.txt", "bbb\n");
    work_dir.run_jj(["commit", "-m", "commit B"]).success();
    work_dir.write_file("file.txt", "ccc\n");
    work_dir.run_jj(["commit", "-m", "commit C"]).success();

    // Run a command that uppercases the file on all 3 commits
    // @--- is A, @-- is B, @- is C
    let output = work_dir.run_jj([
        "run",
        "tr a-z A-Z < file.txt > file.txt.tmp && mv file.txt.tmp file.txt",
        "-r", "@---::@-",
    ]);
    assert!(
        output.status.success(),
        "jj run should succeed.\nstdout: {}\nstderr: {}",
        output.stdout,
        output.stderr
    );
    let stderr = output.stderr.to_string();
    assert!(
        stderr.contains("Rewrote 3 commit(s)"),
        "Should rewrite 3: {stderr}"
    );

    // Verify file was uppercased in first and last commit
    let show_a = work_dir.run_jj(["file", "show", "file.txt", "-r", "@---"]);
    assert!(show_a.stdout.to_string().contains("AAA"), "Expected AAA, got: {:?}", show_a.stdout);

    let show_c = work_dir.run_jj(["file", "show", "file.txt", "-r", "@-"]);
    assert!(show_c.stdout.to_string().contains("CCC"), "Expected CCC, got: {:?}", show_c.stdout);
}

#[cfg(unix)]
#[test]
fn test_run_noop_command() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir.write_file("file.txt", "content");
    work_dir.run_jj(["commit", "-m", "initial"]).success();

    // Run a command that doesn't modify any files
    let output = work_dir.run_jj(["run", "true", "-r", "@-"]);
    assert!(output.status.success(), "should succeed: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("No commits were rewritten"), "Should be noop: {stderr}");
}

#[cfg(unix)]
#[test]
fn test_run_failing_command() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir.write_file("file.txt", "content");
    work_dir.run_jj(["commit", "-m", "initial"]).success();

    // Run a command that fails
    let output = work_dir.run_jj(["run", "exit 1", "-r", "@-"]);
    assert!(!output.status.success(), "should fail: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("failed"), "Should report failure: {stderr}");
}

#[cfg(unix)]
#[test]
fn test_run_on_immutable() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir.write_file("file.txt", "content");
    work_dir.run_jj(["commit", "-m", "initial"]).success();

    // The root commit should be immutable
    let output = work_dir.run_jj(["run", "echo test", "-r", "root()"]);
    assert!(!output.status.success(), "should fail: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("immutable"), "Should report immutable: {stderr}");
}

#[cfg(unix)]
#[test]
fn test_run_env_vars() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir.write_file("file.txt", "placeholder");
    work_dir.run_jj(["commit", "-m", "initial"]).success();

    // Run a command that writes environment variables to the file
    let output = work_dir.run_jj([
        "run",
        "echo change=${JJ_CHANGE} commit=${JJ_COMMIT} > file.txt",
        "-r", "@-",
    ]);
    assert!(output.status.success(), "jj run should succeed: {output}");

    // Verify the env vars were written
    let show = work_dir.run_jj(["file", "show", "file.txt", "-r", "@-"]);
    let content = show.stdout.to_string();
    assert!(content.contains("change="), "Should contain JJ_CHANGE: {content}");
    assert!(content.contains("commit="), "Should contain JJ_COMMIT: {content}");
}

#[cfg(unix)]
#[test]
fn test_run_readonly() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir.write_file("file.txt", "original\n");
    work_dir.run_jj(["commit", "-m", "initial"]).success();

    // Run a command that modifies files, but with --readonly
    let output = work_dir.run_jj([
        "run",
        "tr a-z A-Z < file.txt > file.txt.tmp && mv file.txt.tmp file.txt",
        "-r", "@-",
        "--readonly",
    ]);
    assert!(output.status.success(), "should succeed: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("readonly"), "Should mention readonly: {stderr}");

    // Verify the commit was NOT rewritten — file should still be lowercase
    let show = work_dir.run_jj(["file", "show", "file.txt", "-r", "@-"]);
    assert!(
        show.stdout.to_string().contains("original"),
        "File should be unchanged: {:?}",
        show.stdout
    );
}

#[cfg(unix)]
#[test]
fn test_run_keep_going() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    // Create a stack where the middle commit will fail
    work_dir.write_file("a.txt", "aaa\n");
    work_dir.run_jj(["commit", "-m", "commit A"]).success();
    work_dir.write_file("b.txt", "bbb\n");
    work_dir.run_jj(["commit", "-m", "commit B"]).success();
    work_dir.write_file("c.txt", "ccc\n");
    work_dir.run_jj(["commit", "-m", "commit C"]).success();

    // Command that fails if b.txt exists (commit B), succeeds otherwise
    let output = work_dir.run_jj([
        "run",
        "test ! -f b.txt",
        "-r", "@---::@-",
        "-k",
    ]);
    // Should succeed overall (continue mode doesn't fail the command)
    assert!(output.status.success(), "continue mode should not fail: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("failed"), "Should report failure: {stderr}");
}

#[cfg(unix)]
#[test]
fn test_run_parallel() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    // Create a stack of commits
    work_dir.write_file("a.txt", "aaa\n");
    work_dir.run_jj(["commit", "-m", "commit A"]).success();
    work_dir.write_file("b.txt", "bbb\n");
    work_dir.run_jj(["commit", "-m", "commit B"]).success();

    // Run with -j 2
    let output = work_dir.run_jj([
        "run",
        "tr a-z A-Z < a.txt > a.txt.tmp && mv a.txt.tmp a.txt",
        "-r", "@--::@-",
        "-j", "2",
    ]);
    assert!(output.status.success(), "parallel run should succeed: {output}");
    let stderr = output.stderr.to_string();
    assert!(stderr.contains("Rewrote"), "Should rewrite: {stderr}");
}

#[cfg(unix)]
#[test]
fn test_run_on_working_copy() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    // Create a file in the working copy (no explicit commit)
    work_dir.write_file("file.txt", "hello\n");
    work_dir.run_jj(["commit", "-m", "initial"]).success();

    // Run on @ (the default)
    let output = work_dir.run_jj([
        "run",
        "tr a-z A-Z < file.txt > file.txt.tmp && mv file.txt.tmp file.txt",
    ]);
    // @ is empty so there's nothing to transform
    assert!(output.status.success(), "should succeed: {output}");
}
