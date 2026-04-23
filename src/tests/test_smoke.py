from importlib.metadata import version

from typer.testing import CliRunner

from qctddft.cli import App


runner = CliRunner()


def test_package_version_is_available():
    assert version("qctddft")


def test_cli_reports_version():
    result = runner.invoke(App, ["--version"])
    assert result.exit_code == 0
    assert "qctddft" in result.stdout
