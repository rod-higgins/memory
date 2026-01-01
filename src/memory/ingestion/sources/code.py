"""Code data source adapters (GitHub, GitLab, local git)."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from .base import DataCategory, DataPoint, DataSource


class LocalGitSource(DataSource):
    """Adapter for local git repositories."""

    source_type = "local_git"

    def __init__(
        self,
        path: str,
        include_commits: bool = True,
        include_code: bool = False,
        max_commits: int = 1000,
        **kwargs,
    ):
        """
        Initialize local git source.

        Args:
            path: Path to directory containing git repos
            include_commits: Include commit messages
            include_code: Include code files (increases data significantly)
            max_commits: Maximum commits per repo
        """
        super().__init__(**kwargs)
        self.path = Path(path).expanduser()
        self.include_commits = include_commits
        self.include_code = include_code
        self.max_commits = max_commits

    @property
    def name(self) -> str:
        """Human-readable name of this source."""
        return "Local Git Repositories"

    @property
    def category(self) -> DataCategory:
        """Primary category of data from this source."""
        return DataCategory.CODE

    async def is_available(self) -> bool:
        """Check if the path exists."""
        return self.path.exists()

    async def estimate_count(self) -> int:
        """Estimate number of commits across all repos."""
        if not self.path.exists():
            return 0
        git_dirs = list(self.path.glob("**/.git"))
        return len(git_dirs) * 100  # Rough estimate

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over git repositories."""
        if not self.path.exists():
            return

        # Find all git repos
        git_dirs = list(self.path.glob("**/.git"))

        for git_dir in git_dirs:
            repo_path = git_dir.parent
            repo_name = repo_path.name

            if self.include_commits:
                async for dp in self._iterate_commits(repo_path, repo_name):
                    yield dp

            if self.include_code:
                async for dp in self._iterate_code(repo_path, repo_name):
                    yield dp

    async def _iterate_commits(self, repo_path: Path, repo_name: str) -> AsyncIterator[DataPoint]:
        """Iterate over commits in a repository."""
        try:
            # Get commit log
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "log",
                    f"--max-count={self.max_commits}",
                    "--format=%H|%an|%ae|%at|%s|%b",
                    "--no-merges",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|", 5)
                if len(parts) < 5:
                    continue

                commit_hash, author, email, timestamp_str, subject = parts[:5]
                body = parts[5] if len(parts) > 5 else ""

                try:
                    timestamp = datetime.fromtimestamp(int(timestamp_str))
                except (ValueError, OSError):
                    timestamp = None

                content = f"{subject}\n\n{body}".strip() if body else subject

                yield DataPoint(
                    content=content,
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=f"{repo_name}/{commit_hash[:8]}",
                    timestamp=timestamp,
                    metadata={
                        "repo": repo_name,
                        "commit": commit_hash,
                        "author": author,
                        "email": email,
                        "type": "commit",
                    },
                    tags=["git", "commit", repo_name],
                )

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    async def _iterate_code(self, repo_path: Path, repo_name: str) -> AsyncIterator[DataPoint]:
        """Iterate over code files in a repository."""
        # Common code file extensions
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
            ".r",
        }

        for file_path in repo_path.glob("**/*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in code_extensions:
                continue

            # Skip common non-source directories
            if any(
                x in str(file_path)
                for x in [
                    "node_modules",
                    "vendor",
                    ".git",
                    "__pycache__",
                    "venv",
                    ".venv",
                    "build",
                    "dist",
                ]
            ):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not content.strip():
                continue

            # Limit file size
            if len(content) > 50000:
                content = content[:50000] + "\n... (truncated)"

            stat = file_path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)

            rel_path = file_path.relative_to(repo_path)

            yield DataPoint(
                content=content,
                category=self.category,
                source_type=self.source_type,
                source_identifier=f"{repo_name}/{rel_path}",
                timestamp=modified,
                metadata={
                    "repo": repo_name,
                    "file": str(rel_path),
                    "extension": file_path.suffix,
                    "type": "code",
                },
                tags=["git", "code", file_path.suffix.lstrip(".")],
            )


class GitHubSource(DataSource):
    """Adapter for GitHub repositories and activity."""

    def __init__(
        self,
        token: str | None = None,
        username: str | None = None,
        include_repos: bool = True,
        include_issues: bool = True,
        include_prs: bool = True,
        include_stars: bool = True,
    ):
        """
        Initialize GitHub source.

        Args:
            token: GitHub personal access token
            username: GitHub username (if different from token owner)
            include_repos: Include repository info
            include_issues: Include issues created by user
            include_prs: Include pull requests
            include_stars: Include starred repos
        """
        self.token = token
        self.username = username
        self.include_repos = include_repos
        self.include_issues = include_issues
        self.include_prs = include_prs
        self.include_stars = include_stars

    @property
    def name(self) -> str:
        return f"GitHub ({self.username or 'authenticated user'})"

    @property
    def category(self) -> DataCategory:
        return DataCategory.CODE

    async def is_available(self) -> bool:
        return self.token is not None

    async def estimate_count(self) -> int:
        return 100  # Rough estimate

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over GitHub data."""
        import httpx

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        headers["Accept"] = "application/vnd.github.v3+json"

        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            # Get authenticated user if no username
            username = self.username
            if not username and self.token:
                try:
                    resp = await client.get("https://api.github.com/user")
                    if resp.status_code == 200:
                        username = resp.json().get("login")
                except Exception:
                    pass

            if not username:
                return

            # User's repositories
            if self.include_repos:
                async for dp in self._iterate_repos(client, username):
                    yield dp

            # User's issues
            if self.include_issues:
                async for dp in self._iterate_issues(client, username):
                    yield dp

            # User's PRs
            if self.include_prs:
                async for dp in self._iterate_prs(client, username):
                    yield dp

            # Starred repos
            if self.include_stars:
                async for dp in self._iterate_stars(client, username):
                    yield dp

    async def _iterate_repos(self, client, username: str) -> AsyncIterator[DataPoint]:
        """Iterate over user's repositories."""
        page = 1
        while True:
            try:
                resp = await client.get(
                    f"https://api.github.com/users/{username}/repos",
                    params={"page": page, "per_page": 100, "sort": "updated"},
                )
                if resp.status_code != 200:
                    break

                repos = resp.json()
                if not repos:
                    break

                for repo in repos:
                    description = repo.get("description") or ""
                    name = repo.get("name", "")
                    full_name = repo.get("full_name", "")

                    content = f"Repository: {full_name}\n\n{description}"

                    # Parse dates
                    updated = None
                    if repo.get("updated_at"):
                        try:
                            updated = datetime.fromisoformat(repo["updated_at"].replace("Z", "+00:00"))
                        except ValueError:
                            pass

                    yield DataPoint(
                        content=content,
                        category=self.category,
                        source_type="github",
                        source_id=f"github_repo_{full_name}",
                        original_date=updated,
                        raw_data={
                            "type": "repository",
                            "name": name,
                            "full_name": full_name,
                            "language": repo.get("language"),
                            "stars": repo.get("stargazers_count", 0),
                            "forks": repo.get("forks_count", 0),
                            "repo_topics": repo.get("topics", []),
                            "is_fork": repo.get("fork", False),
                            "is_private": repo.get("private", False),
                        },
                        topics=["github", "repository", repo.get("language") or ""],
                    )

                page += 1

            except Exception:
                break

    async def _iterate_issues(self, client, username: str) -> AsyncIterator[DataPoint]:
        """Iterate over issues created by user."""
        page = 1
        while True:
            try:
                resp = await client.get(
                    "https://api.github.com/search/issues",
                    params={
                        "q": f"author:{username} type:issue",
                        "page": page,
                        "per_page": 100,
                        "sort": "updated",
                    },
                )
                if resp.status_code != 200:
                    break

                data = resp.json()
                items = data.get("items", [])
                if not items:
                    break

                for issue in items:
                    title = issue.get("title", "")
                    body = issue.get("body") or ""
                    repo_url = issue.get("repository_url", "")
                    repo_name = repo_url.split("/")[-2:] if repo_url else []

                    content = f"{title}\n\n{body}"

                    created = None
                    if issue.get("created_at"):
                        try:
                            created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
                        except ValueError:
                            pass

                    yield DataPoint(
                        content=content,
                        category=self.category,
                        source_type="github",
                        source_id=f"github_issue_{issue.get('number')}",
                        original_date=created,
                        raw_data={
                            "type": "issue",
                            "number": issue.get("number"),
                            "state": issue.get("state"),
                            "repo": "/".join(repo_name),
                            "labels": [l["name"] for l in issue.get("labels", [])],
                        },
                        topics=["github", "issue"],
                    )

                page += 1

            except Exception:
                break

    async def _iterate_prs(self, client, username: str) -> AsyncIterator[DataPoint]:
        """Iterate over pull requests by user."""
        page = 1
        while True:
            try:
                resp = await client.get(
                    "https://api.github.com/search/issues",
                    params={
                        "q": f"author:{username} type:pr",
                        "page": page,
                        "per_page": 100,
                        "sort": "updated",
                    },
                )
                if resp.status_code != 200:
                    break

                data = resp.json()
                items = data.get("items", [])
                if not items:
                    break

                for pr in items:
                    title = pr.get("title", "")
                    body = pr.get("body") or ""
                    repo_url = pr.get("repository_url", "")
                    repo_name = repo_url.split("/")[-2:] if repo_url else []

                    content = f"PR: {title}\n\n{body}"

                    created = None
                    if pr.get("created_at"):
                        try:
                            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                        except ValueError:
                            pass

                    yield DataPoint(
                        content=content,
                        category=self.category,
                        source_type="github",
                        source_id=f"github_pr_{pr.get('number')}",
                        original_date=created,
                        raw_data={
                            "type": "pull_request",
                            "number": pr.get("number"),
                            "state": pr.get("state"),
                            "repo": "/".join(repo_name),
                            "merged": pr.get("pull_request", {}).get("merged_at") is not None,
                        },
                        topics=["github", "pull_request"],
                    )

                page += 1

            except Exception:
                break

    async def _iterate_stars(self, client, username: str) -> AsyncIterator[DataPoint]:
        """Iterate over starred repositories."""
        page = 1
        while True:
            try:
                resp = await client.get(
                    f"https://api.github.com/users/{username}/starred",
                    params={"page": page, "per_page": 100},
                )
                if resp.status_code != 200:
                    break

                repos = resp.json()
                if not repos:
                    break

                for repo in repos:
                    description = repo.get("description") or ""
                    full_name = repo.get("full_name", "")

                    content = f"Starred: {full_name}\n\n{description}"

                    yield DataPoint(
                        content=content,
                        category=self.category,
                        source_type="github",
                        source_id=f"github_star_{full_name}",
                        original_date=None,
                        raw_data={
                            "type": "star",
                            "full_name": full_name,
                            "language": repo.get("language"),
                            "stars": repo.get("stargazers_count", 0),
                            "repo_topics": repo.get("topics", []),
                        },
                        topics=["github", "star", repo.get("language") or ""],
                    )

                page += 1

            except Exception:
                break


class GitLabSource(DataSource):
    """Adapter for GitLab repositories and activity."""

    source_type = "gitlab"
    category = DataCategory.CODE

    def __init__(
        self,
        token: str,
        url: str = "https://gitlab.com",
        include_repos: bool = True,
        include_issues: bool = True,
        include_mrs: bool = True,
        **kwargs,
    ):
        """
        Initialize GitLab source.

        Args:
            token: GitLab personal access token
            url: GitLab instance URL
            include_repos: Include repositories
            include_issues: Include issues
            include_mrs: Include merge requests
        """
        super().__init__(**kwargs)
        self.token = token
        self.url = url.rstrip("/")
        self.include_repos = include_repos
        self.include_issues = include_issues
        self.include_mrs = include_mrs

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over GitLab data."""
        import httpx

        headers = {"PRIVATE-TOKEN": self.token}

        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            # Get current user
            try:
                resp = await client.get(f"{self.url}/api/v4/user")
                if resp.status_code != 200:
                    return
                user = resp.json()
                user_id = user.get("id")
            except Exception:
                return

            # User's projects
            if self.include_repos:
                page = 1
                while True:
                    try:
                        resp = await client.get(
                            f"{self.url}/api/v4/users/{user_id}/projects",
                            params={"page": page, "per_page": 100},
                        )
                        if resp.status_code != 200:
                            break

                        projects = resp.json()
                        if not projects:
                            break

                        for proj in projects:
                            description = proj.get("description") or ""
                            name = proj.get("path_with_namespace", "")

                            content = f"Project: {name}\n\n{description}"

                            updated = None
                            if proj.get("last_activity_at"):
                                try:
                                    updated = datetime.fromisoformat(proj["last_activity_at"].replace("Z", "+00:00"))
                                except ValueError:
                                    pass

                            yield DataPoint(
                                content=content,
                                category=self.category,
                                source_type=self.source_type,
                                source_identifier=f"gitlab_project_{proj.get('id')}",
                                timestamp=updated,
                                metadata={
                                    "type": "project",
                                    "name": name,
                                    "stars": proj.get("star_count", 0),
                                    "visibility": proj.get("visibility"),
                                },
                                tags=["gitlab", "project"],
                            )

                        page += 1

                    except Exception:
                        break

            # Issues
            if self.include_issues:
                page = 1
                while True:
                    try:
                        resp = await client.get(
                            f"{self.url}/api/v4/issues",
                            params={
                                "scope": "created_by_me",
                                "page": page,
                                "per_page": 100,
                            },
                        )
                        if resp.status_code != 200:
                            break

                        issues = resp.json()
                        if not issues:
                            break

                        for issue in issues:
                            title = issue.get("title", "")
                            description = issue.get("description") or ""

                            content = f"{title}\n\n{description}"

                            created = None
                            if issue.get("created_at"):
                                try:
                                    created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
                                except ValueError:
                                    pass

                            yield DataPoint(
                                content=content,
                                category=self.category,
                                source_type=self.source_type,
                                source_identifier=f"gitlab_issue_{issue.get('iid')}",
                                timestamp=created,
                                metadata={
                                    "type": "issue",
                                    "iid": issue.get("iid"),
                                    "state": issue.get("state"),
                                    "labels": issue.get("labels", []),
                                },
                                tags=["gitlab", "issue"],
                            )

                        page += 1

                    except Exception:
                        break

            # Merge Requests
            if self.include_mrs:
                page = 1
                while True:
                    try:
                        resp = await client.get(
                            f"{self.url}/api/v4/merge_requests",
                            params={
                                "scope": "created_by_me",
                                "page": page,
                                "per_page": 100,
                            },
                        )
                        if resp.status_code != 200:
                            break

                        mrs = resp.json()
                        if not mrs:
                            break

                        for mr in mrs:
                            title = mr.get("title", "")
                            description = mr.get("description") or ""

                            content = f"MR: {title}\n\n{description}"

                            created = None
                            if mr.get("created_at"):
                                try:
                                    created = datetime.fromisoformat(mr["created_at"].replace("Z", "+00:00"))
                                except ValueError:
                                    pass

                            yield DataPoint(
                                content=content,
                                category=self.category,
                                source_type=self.source_type,
                                source_identifier=f"gitlab_mr_{mr.get('iid')}",
                                timestamp=created,
                                metadata={
                                    "type": "merge_request",
                                    "iid": mr.get("iid"),
                                    "state": mr.get("state"),
                                    "merged": mr.get("merged_at") is not None,
                                },
                                tags=["gitlab", "merge_request"],
                            )

                        page += 1

                    except Exception:
                        break
