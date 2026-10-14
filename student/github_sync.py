"""Push an explicit allowlist of report artifacts, never datasets or weights."""
import os
import subprocess
import tempfile
from pathlib import Path


def push_results(repo, token, branch):
    repo = Path(repo)
    if not branch.startswith("kaggle-results-"):
        raise ValueError("Use a unique kaggle-results-* branch")
    env = os.environ.copy()
    env["LAB_GITHUB_TOKEN"] = token
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Askpass stays outside the repo and only returns credentials to git.
    with tempfile.TemporaryDirectory(prefix="lab-auth-") as tmp:
        askpass = Path(tmp) / "askpass.py"
        askpass.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "Username" in sys.argv[1] else os.environ["LAB_GITHUB_TOKEN"])\n')
        askpass.chmod(0o700)
        env["GIT_ASKPASS"] = str(askpass)
        def git(*args, check=True):
            return subprocess.run(["git", "-c", "credential.helper=", *args], cwd=repo,
                                  env=env, check=check, capture_output=True, text=True)
        current = git("branch", "--show-current").stdout.strip()
        if current != branch:
            git("checkout", "-b", branch)
        git("config", "user.name", "Kaggle experiment runner")
        git("config", "user.email", "minhduong814@users.noreply.github.com")
        allowed = []
        for path in (repo / "report").rglob("*"):
            if path.is_file() and path.suffix in {".json", ".md", ".png", ".txt"}:
                if path.stat().st_size > 20 * 1024**2:
                    raise RuntimeError(f"Unexpected large report artifact: {path.name}")
                allowed.append(str(path.relative_to(repo)))
        if not allowed:
            raise RuntimeError("No report artifacts to push")
        git("add", "--", *allowed)
        if git("diff", "--cached", "--quiet", check=False).returncode:
            git("commit", "-m", "Save Kaggle E0–E4 experiment evidence and report artifacts")
        result = git("push", "origin", f"HEAD:refs/heads/{branch}", check=False)
        if result.returncode:
            # Do not print authentication diagnostics into notebook output.
            raise RuntimeError("GitHub push failed. Results remain in Kaggle Output; check token permissions and branch rules.")
        print(f"Results pushed to {branch}")
