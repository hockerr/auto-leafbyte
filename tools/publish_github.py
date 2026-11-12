"""Publish a folder to GitHub through the REST API (no local git required).

Why not ``git push``?  In some sandboxed Windows environments git's TLS backend
cannot acquire credentials, while Python's OpenSSL stack works fine.  This script
uses the GitHub *Git Data* API instead: upload blobs -> tree -> commit -> update
the branch ref.  It can also create the repository for you.

Usage::

    python tools/publish_github.py --token <PAT> --repo auto-leafbyte --public
    python tools/publish_github.py --token <PAT> --repo auto-leafbyte --dry-run

Token
-----
A fine-grained personal access token with, for the target account:
  * "Administration: read and write"  (only if the repository must be created)
  * "Contents: read and write"        (always)
or a classic token with the ``repo`` scope.

The token is only used in request headers; it is never printed, written to disk
or stored anywhere.  Revoke/rotate it after publishing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
USER_AGENT = "auto-leafbyte-publisher/1.0"
MAX_BLOB_BYTES = 40 * 1024 * 1024  # stay well under GitHub's 100 MB hard limit


class GitHubError(RuntimeError):
    pass


class GitHub:
    def __init__(self, token: str) -> None:
        self._token = token

    def request(self, method: str, path: str, payload: dict | None = None, attempts: int = 4):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                API + path,
                data=body,
                method=method,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:600]
                # 5xx / rate limiting are worth retrying, 4xx client errors are not
                if error.code >= 500 or error.code == 429:
                    last_error = GitHubError(f"{method} {path} -> HTTP {error.code}: {detail}")
                else:
                    raise GitHubError(f"{method} {path} -> HTTP {error.code}: {detail}") from None
            except (urllib.error.URLError, ssl.SSLError, TimeoutError, ConnectionError, OSError) as error:
                reason = getattr(error, "reason", error)
                last_error = GitHubError(f"network error on {method} {path}: {reason}")
            if attempt < attempts:
                delay = min(2 ** attempt, 10)
                print(f"  retrying {method} {path} in {delay}s (attempt {attempt}/{attempts})")
                time.sleep(delay)
        raise last_error if last_error else GitHubError(f"{method} {path} failed")


#: Never uploaded (local scratch that the API path does not filter via .gitignore).
SKIP_PARTS = {"__pycache__", ".git", ".pytest_cache", ".venv", "venv"}
SKIP_PREFIXES = ("out/tests/", "out/_test_review/", "data/paper/images/", "data/fixtures/")
SKIP_SUFFIXES = {".pyc", ".pyo"}


def collect_files(directory: Path) -> list[Path]:
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        posix = relative.as_posix()
        if any(part in SKIP_PARTS for part in relative.parts[:-1]):
            continue
        if any(posix.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name.startswith(".") and path.name != ".gitignore":
            continue
        files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub PAT (default: $GITHUB_TOKEN)")
    parser.add_argument("--repo", default="auto-leafbyte", help="repository name to create/push")
    parser.add_argument("--dir", default="release/auto-leafbyte", help="folder to publish")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--description", default="Automatic LeafByte leaf-area measurement with a human-review queue for low-precision samples")
    parser.add_argument("--message", default="Release auto-leafbyte")
    parser.add_argument("--public", dest="private", action="store_false", default=False,
                        help="make the repository public (default: private)")
    parser.add_argument("--replace", action="store_true",
                        help="replace the branch tree entirely (deletes files removed locally)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be uploaded")
    args = parser.parse_args(argv)

    directory = Path(args.dir).resolve()
    if not directory.is_dir():
        print(f"error: {directory} is not a directory (run tools/make_release.py first)")
        return 2
    files = collect_files(directory)
    total = sum(path.stat().st_size for path in files)
    print(f"folder : {directory}")
    print(f"files  : {len(files)}   size: {total / 1e6:.2f} MB   visibility: "
          f"{'private' if args.private else 'public'}")
    for path in files:
        print(f"  {path.relative_to(directory).as_posix():<52}{path.stat().st_size:>10} B")
    if max(path.stat().st_size for path in files) > MAX_BLOB_BYTES:
        print("error: a file exceeds the blob upload limit")
        return 2
    if args.dry_run:
        print("\ndry run - nothing uploaded")
        return 0
    if not args.token:
        print("error: no token. Pass --token <PAT> or set GITHUB_TOKEN.")
        return 2

    client = GitHub(args.token)
    user = client.request("GET", "/user")
    login = user.get("login")
    print(f"\nauthenticated as {login}")

    repository = f"{login}/{args.repo}"
    try:
        repo_info = client.request("GET", f"/repos/{repository}")
        print(f"repository exists: {repo_info['html_url']}")
    except GitHubError as error:
        if not any(code in str(error) for code in ("HTTP 404", "HTTP 409")):
            raise
        print(f"creating repository {repository} ...")
        repo_info = client.request("POST", "/user/repos", {
            "name": args.repo,
            "description": args.description,
            "private": args.private,
            "auto_init": False,
            "has_issues": True,
            "has_wiki": False,
        })
        print(f"created: {repo_info['html_url']}")

    # --- parent commit of the target branch (may not exist yet) -----------
    parent_sha = None
    base_tree = None
    try:
        reference = client.request("GET", f"/repos/{repository}/git/ref/heads/{args.branch}")
        parent_sha = reference["object"]["sha"]
        parent = client.request("GET", f"/repos/{repository}/git/commits/{parent_sha}")
        base_tree = parent["tree"]["sha"]
        print(f"branch {args.branch} exists at {parent_sha[:8]}, appending a commit")
    except GitHubError as error:
        if not any(code in str(error) for code in ("HTTP 404", "HTTP 409")):
            raise
        # GitHub's Git Data API answers 409 ("Git Repository is empty") until the
        # repository has one commit, so seed it with the .gitignore via the
        # Contents API and then continue normally.
        seed_name = ".gitignore"
        seed_file = directory / seed_name
        if not seed_file.exists():
            seed_name = "README.md"
            seed_file = directory / seed_name
        print(f"empty repository: seeding an initial commit with {seed_name} ...")
        client.request("PUT", f"/repos/{repository}/contents/{seed_name}", {
            "message": "chore: initialise repository",
            "content": base64.b64encode(seed_file.read_bytes()).decode("ascii"),
            "branch": args.branch,
        })
        reference = client.request("GET", f"/repos/{repository}/git/ref/heads/{args.branch}")
        parent_sha = reference["object"]["sha"]
        parent = client.request("GET", f"/repos/{repository}/git/commits/{parent_sha}")
        base_tree = parent["tree"]["sha"]
        print(f"seeded at {parent_sha[:8]}")

    # --- upload blobs -----------------------------------------------------
    tree_entries = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(directory).as_posix()
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        blob = client.request("POST", f"/repos/{repository}/git/blobs",
                              {"content": content, "encoding": "base64"})
        tree_entries.append({"path": relative, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if index % 10 == 0 or index == len(files):
            print(f"  uploaded {index}/{len(files)} blobs")

    tree_payload: dict = {"tree": tree_entries}
    if base_tree and not args.replace:
        tree_payload["base_tree"] = base_tree
    tree = client.request("POST", f"/repos/{repository}/git/trees", tree_payload)

    commit_payload: dict = {"message": args.message, "tree": tree["sha"]}
    if parent_sha:
        commit_payload["parents"] = [parent_sha]
    commit = client.request("POST", f"/repos/{repository}/git/commits", commit_payload)

    if parent_sha:
        client.request("PATCH", f"/repos/{repository}/git/refs/heads/{args.branch}",
                       {"sha": commit["sha"], "force": False})
    else:
        client.request("POST", f"/repos/{repository}/git/refs",
                       {"ref": f"refs/heads/{args.branch}", "sha": commit["sha"]})

    print(f"\npublished: https://github.com/{repository}/tree/{args.branch}")
    print(f"commit   : {commit['sha'][:12]}")
    print("\nremember to revoke/rotate the token you just used.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GitHubError as error:
        print(f"\nGitHub API error: {error}", file=sys.stderr)
        sys.exit(1)
