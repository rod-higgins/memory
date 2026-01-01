"""Tests for help API endpoints."""

import pytest
from pathlib import Path


class TestHelpAPI:
    """Tests for help documentation API."""

    def test_help_directory_exists(self):
        """Test that help directory exists."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"
        assert help_dir.exists(), "Help directory should exist"
        assert help_dir.is_dir(), "Help path should be a directory"

    def test_help_files_exist(self):
        """Test that expected help files exist."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"
        expected_files = [
            "GETTING_STARTED.md",
            "USER_GUIDE.md",
            "MEMORIES.md",
            "AI_ASSISTANT.md",
            "DATA_SOURCES.md",
            "ADMIN.md",
            "KEYBOARD_SHORTCUTS.md",
            "INDEX.md",
        ]

        for filename in expected_files:
            filepath = help_dir / filename
            assert filepath.exists(), f"Help file {filename} should exist"

    def test_help_files_not_empty(self):
        """Test that help files have content."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"

        for md_file in help_dir.glob("*.md"):
            content = md_file.read_text()
            assert len(content) > 100, f"{md_file.name} should have substantial content"

    def test_help_files_valid_markdown(self):
        """Test that help files are valid markdown."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"

        for md_file in help_dir.glob("*.md"):
            content = md_file.read_text()
            # Check for heading
            assert content.startswith("#"), f"{md_file.name} should start with a heading"
            # Check for multiple sections
            assert content.count("\n##") >= 1, f"{md_file.name} should have multiple sections"

    def test_getting_started_content(self):
        """Test Getting Started guide has essential sections."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"
        content = (help_dir / "GETTING_STARTED.md").read_text()

        assert "# Getting Started" in content
        assert "Log In" in content
        assert "First" in content or "Quick" in content

    def test_user_guide_content(self):
        """Test User Guide has essential sections."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"
        content = (help_dir / "USER_GUIDE.md").read_text()

        assert "# User Guide" in content
        assert "Memory" in content or "Memories" in content
        assert "AI" in content or "Chat" in content

    def test_admin_guide_content(self):
        """Test Admin Guide has essential sections."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"
        content = (help_dir / "ADMIN.md").read_text()

        assert "Admin" in content
        assert "User" in content
        assert "Password" in content or "MFA" in content

    def test_help_cross_references(self):
        """Test that help files have proper cross-references."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"

        # Getting started should reference other guides
        getting_started = (help_dir / "GETTING_STARTED.md").read_text()
        assert "USER_GUIDE.md" in getting_started or "User Guide" in getting_started

    def test_index_lists_all_guides(self):
        """Test that index file references all guides."""
        help_dir = Path(__file__).parent.parent / "docs" / "help"
        index_content = (help_dir / "INDEX.md").read_text()

        # Should reference the main guides
        assert "Getting Started" in index_content
        assert "User Guide" in index_content
        assert "Admin" in index_content


class TestHelpAPIEndpoint:
    """Integration tests for help API endpoints (requires running server)."""

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API tests."""
        return "http://localhost:8765"

    @pytest.mark.skip(reason="Requires running server")
    def test_list_help_pages(self, api_base_url):
        """Test listing help pages."""
        import httpx

        response = httpx.get(f"{api_base_url}/api/help")
        assert response.status_code == 200

        data = response.json()
        assert "pages" in data
        assert len(data["pages"]) >= 7  # At least 7 help pages

    @pytest.mark.skip(reason="Requires running server")
    def test_get_help_page(self, api_base_url):
        """Test getting a specific help page."""
        import httpx

        response = httpx.get(f"{api_base_url}/api/help/getting_started")
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert "title" in data
        assert "content" in data
        assert data["content"].startswith("#")

    @pytest.mark.skip(reason="Requires running server")
    def test_get_help_page_not_found(self, api_base_url):
        """Test getting non-existent help page."""
        import httpx

        response = httpx.get(f"{api_base_url}/api/help/nonexistent")
        assert response.status_code == 404
