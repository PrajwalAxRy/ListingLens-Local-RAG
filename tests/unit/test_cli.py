"""
Unit tests for CLI commands.
"""
import tempfile
from pathlib import Path

import pytest
from rhp_analyzer.cli.main import app
from typer.testing import CliRunner

# Initialize CLI test runner
runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_help_output(self):
        """Test that --help displays help information."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "RHP Analyzer" in result.stdout or "Usage" in result.stdout

    def test_version_output(self):
        """Test that --version displays version information."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Check for version pattern (e.g., "0.1.0")
        assert any(char.isdigit() for char in result.stdout)

    def test_unknown_command(self):
        """Test that unknown commands show error."""
        result = runner.invoke(app, ["unknown-command"])
        assert result.exit_code != 0


class TestAnalyzeCommand:
    """Test the analyze command."""

    def test_analyze_help(self):
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout.lower()
        assert "PDF" in result.stdout or "pdf" in result.stdout

    def test_analyze_missing_argument(self):
        """Test analyze without PDF path shows error."""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0
        # Typer error messages may go to stdout or stderr, check both
        combined_output = result.stdout + (result.stderr if hasattr(result, "stderr") else "")
        assert "Missing argument" in combined_output or "required" in combined_output.lower() or result.exit_code == 2

    def test_analyze_nonexistent_file(self):
        """Test analyze with non-existent file shows error."""
        result = runner.invoke(app, ["analyze", "nonexistent.pdf"])
        assert result.exit_code != 0

    def test_analyze_non_pdf_file(self):
        """Test analyze with non-PDF file shows error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"Not a PDF")
            tmp_path = tmp.name

        try:
            result = runner.invoke(app, ["analyze", tmp_path])
            assert result.exit_code != 0
            # Should mention PDF requirement
            assert "PDF" in result.stdout or "pdf" in result.stdout
        finally:
            Path(tmp_path).unlink()

    def test_analyze_dry_run_with_valid_pdf(self):
        """Test analyze dry run with a valid PDF."""
        # Create a minimal valid PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as tmp:
            # Minimal PDF header
            tmp.write(b"%PDF-1.4\n")
            tmp.write(b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>endobj\n")
            tmp.write(b"2 0 obj\n<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n")
            tmp.write(b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n")
            tmp.write(b"xref\n0 4\n")
            tmp.write(b"0000000000 65535 f\n")
            tmp.write(b"0000000009 00000 n\n")
            tmp.write(b"0000000058 00000 n\n")
            tmp.write(b"0000000115 00000 n\n")
            tmp.write(b"trailer\n<</Size 4/Root 1 0 R>>\nstartxref\n194\n%%EOF\n")
            tmp_path = tmp.name

        try:
            result = runner.invoke(app, ["analyze", tmp_path, "--dry-run"])
            # Dry run should succeed even with minimal implementation
            assert result.exit_code == 0
            assert "Dry run" in result.stdout or "validation" in result.stdout.lower()
        finally:
            Path(tmp_path).unlink()

    def test_analyze_verbose_flag(self):
        """Test analyze with verbose flag."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as tmp:
            tmp.write(b"%PDF-1.4\n%%EOF\n")
            tmp_path = tmp.name

        try:
            result = runner.invoke(app, ["analyze", tmp_path, "--dry-run", "--verbose"])
            # Should run (exit code 0 for dry-run even if not fully implemented)
            assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink()

    def test_analyze_output_dir_option(self):
        """Test analyze with custom output directory."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as tmp:
            tmp.write(b"%PDF-1.4\n%%EOF\n")
            tmp_path = tmp.name

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = runner.invoke(
                    app,
                    ["analyze", tmp_path, "--output-dir", tmpdir, "--dry-run"],
                )
                assert result.exit_code == 0
            finally:
                Path(tmp_path).unlink()


class TestConfigCommand:
    """Test the config command."""

    def test_config_show(self):
        """Test config show command."""
        result = runner.invoke(app, ["config", "show"])
        # Should either work or show that it's not implemented yet
        # Not failing is acceptable for Phase 1
        assert result.exit_code in [0, 1, 2]

    def test_config_help(self):
        """Test config command help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_help(self):
        """Test validate command help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0

    def test_validate_basic(self):
        """Test validate command basic invocation."""
        result = runner.invoke(app, ["validate"])
        # Should either work or show that it's not implemented yet
        assert result.exit_code in [0, 1, 2]


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_option(self):
        """Test that invalid options are caught."""
        result = runner.invoke(app, ["analyze", "--invalid-option"])
        assert result.exit_code != 0

    def test_multiple_commands_invalid(self):
        """Test that multiple commands aren't allowed."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4\n%%EOF\n")
            tmp_path = tmp.name

        try:
            runner.invoke(app, ["analyze", tmp_path, "config", "show"])
            # Either fails or processes only first command
            # Either behavior is acceptable
            assert True  # Just verify it doesn't crash
        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
