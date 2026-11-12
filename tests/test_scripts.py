"""Bash scripts: static rules, and `pg-backup.sh` / `restore.sh` / `deploy.sh` run against stub binaries."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
BACKUP_SH: Path = ROOT / "k8s" / "base" / "pg-backup.sh"
SCRIPTS: list[Path] = [
    ROOT / "scripts" / "k3d-cluster.sh",
    ROOT / "scripts" / "deploy.sh",
    ROOT / "scripts" / "backup.sh",
    ROOT / "scripts" / "restore.sh",
]
ALL: list[Path] = [*SCRIPTS, BACKUP_SH]
OLD: list[str] = [f"tracker-2026010{n}T030000Z.dump" for n in (1, 2, 3, 4)]
NEW_NAME: re.Pattern[str] = re.compile(r"^tracker-\d{8}T\d{6}Z\.dump$")


def need(binary: str) -> None:
    if shutil.which(binary) is None:
        if os.environ.get("CI"):
            pytest.fail(f"{binary} is not in PATH, but CI is set")
        pytest.skip(f"{binary} is not in PATH")


def stub(bin_dir: Path, name: str, body: str) -> None:
    path: Path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def marker_stub(bin_dir: Path, name: str, marks: Path) -> None:
    stub(bin_dir, name, f'echo "{name} $*" >> "{marks}"')


def run(script: Path, bin_dir: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    full: dict[str, str] = {**os.environ, **env, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["bash", str(script), *args], env=full, capture_output=True, text=True, check=False, cwd=ROOT
    )


def test_scripts_directory_holds_exactly_the_four() -> None:
    assert sorted(p.name for p in (ROOT / "scripts").iterdir()) == [
        "backup.sh",
        "deploy.sh",
        "k3d-cluster.sh",
        "restore.sh",
    ]


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_script_header_and_executable(path: Path) -> None:
    lines: list[str] = path.read_text().splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    rest: list[str] = [ln.strip() for ln in lines[1:] if ln.strip() and not ln.strip().startswith("#")]
    assert rest[0] == "set -euo pipefail"
    assert os.access(path, os.X_OK)


def test_shellcheck_is_clean() -> None:
    need("shellcheck")
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["shellcheck", *map(str, ALL)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout


class Backup:
    """A BACKUP_DIR with four old dumps and notes.txt, and stub pg_dump / pg_restore."""

    def __init__(self, tmp_path: Path, dump_code: int = 0, restore_code: int = 0) -> None:
        self.dir: Path = tmp_path / "backups"
        self.bin: Path = tmp_path / "bin"
        self.calls: Path = tmp_path / "calls"
        self.dir.mkdir()
        self.bin.mkdir()
        self.calls.write_text("")
        name: str
        for name in OLD:
            (self.dir / name).write_text("OLD")
        (self.dir / "notes.txt").write_text("notes")
        stub(
            self.bin,
            "pg_dump",
            f'echo "pg_dump $*" >> "{self.calls}"\n'
            'for a in "$@"; do case "$a" in --file=*) echo DUMP > "${a#--file=}";; esac; done\n'
            f"exit {dump_code}",
        )
        stub(self.bin, "pg_restore", f'echo "pg_restore $*" >> "{self.calls}"\nexit {restore_code}')

    def run(self, keep: str) -> subprocess.CompletedProcess[str]:
        return run(BACKUP_SH, self.bin, {"BACKUP_DIR": str(self.dir), "BACKUP_KEEP": keep})

    def files(self) -> set[str]:
        return {p.name for p in self.dir.iterdir()}

    def lines(self, tool: str) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln.startswith(tool + " ")]


def test_backup_creates_rotates_and_leaves_other_files(tmp_path: Path) -> None:
    b: Backup = Backup(tmp_path)
    result: subprocess.CompletedProcess[str] = b.run("3")
    assert result.returncode == 0, result.stderr
    new: list[str] = [f for f in b.files() if f not in OLD and f != "notes.txt"]
    assert len(new) == 1
    assert NEW_NAME.match(new[0])
    assert (b.dir / new[0]).read_text().strip() == "DUMP"
    assert b.files() == {OLD[2], OLD[3], new[0], "notes.txt"}
    assert not list(b.dir.glob("*.partial"))
    out: list[str] = result.stdout.splitlines()
    assert f"created {new[0]}" in out
    assert f"deleted {OLD[0]}" in out
    assert f"deleted {OLD[1]}" in out
    assert f"deleted {OLD[2]}" not in out


def test_backup_calls_pg_dump_and_pg_restore_once_each(tmp_path: Path) -> None:
    b: Backup = Backup(tmp_path)
    assert b.run("3").returncode == 0
    dumps: list[str] = b.lines("pg_dump")
    assert len(dumps) == 1
    args: list[str] = dumps[0].split()[1:]
    assert args[0] == "--format=custom"
    assert len(args) == 2
    assert args[1].startswith(f"--file={b.dir}/tracker-")
    assert args[1].endswith(".partial")
    restores: list[str] = b.lines("pg_restore")
    assert len(restores) == 1
    assert restores[0].split()[1:3] == ["--list", args[1].removeprefix("--file=")]


def test_backup_keeps_everything_when_fewer_than_keep(tmp_path: Path) -> None:
    b: Backup = Backup(tmp_path)
    result: subprocess.CompletedProcess[str] = b.run("14")
    assert result.returncode == 0, result.stderr
    assert set(OLD) <= b.files()
    assert "deleted" not in result.stdout


def test_backup_pg_dump_failure_changes_nothing(tmp_path: Path) -> None:
    b: Backup = Backup(tmp_path, dump_code=1)
    result: subprocess.CompletedProcess[str] = b.run("3")
    assert result.returncode != 0
    assert len(b.lines("pg_dump")) == 1
    assert b.files() == {*OLD, "notes.txt"}


def test_backup_unreadable_archive_changes_nothing(tmp_path: Path) -> None:
    b: Backup = Backup(tmp_path, restore_code=1)
    result: subprocess.CompletedProcess[str] = b.run("3")
    assert result.returncode != 0
    assert len(b.lines("pg_restore")) == 1
    assert b.files() == {*OLD, "notes.txt"}


@pytest.mark.parametrize("keep", ["0", "abc", "-1"])
def test_backup_rejects_bad_keep(tmp_path: Path, keep: str) -> None:
    b: Backup = Backup(tmp_path)
    result: subprocess.CompletedProcess[str] = b.run(keep)
    assert result.returncode == 2
    assert "BACKUP_KEEP" in result.stderr
    assert b.lines("pg_dump") == []
    assert b.files() == {*OLD, "notes.txt"}


class Stubs:
    """Stub docker, k3d and kubectl that leave a line per call."""

    def __init__(self, tmp_path: Path) -> None:
        self.bin: Path = tmp_path / "bin"
        self.bin.mkdir()
        self.marks: Path = tmp_path / "marks"
        name: str
        for name in ("docker", "k3d", "kubectl"):
            marker_stub(self.bin, name, self.marks)

    def called(self) -> bool:
        return self.marks.exists() and self.marks.read_text() != ""


@pytest.mark.parametrize(
    "args",
    [[], ["latest", "extra"], ["foo"]],
    ids=["no-args", "two-args", "bad-name"],
)
def test_restore_usage_errors(tmp_path: Path, args: list[str]) -> None:
    s: Stubs = Stubs(tmp_path)
    result: subprocess.CompletedProcess[str] = run(ROOT / "scripts" / "restore.sh", s.bin, {}, *args)
    assert result.returncode == 2
    assert result.stderr.startswith("usage: restore.sh")
    assert not s.called()


def test_restore_missing_host_file(tmp_path: Path) -> None:
    s: Stubs = Stubs(tmp_path)
    result: subprocess.CompletedProcess[str] = run(
        ROOT / "scripts" / "restore.sh", s.bin, {}, "/nonexistent/x.dump"
    )
    assert result.returncode == 1
    assert not s.called()


def test_deploy_without_env_file(tmp_path: Path) -> None:
    s: Stubs = Stubs(tmp_path)
    missing: str = str(tmp_path / "no-such.env")
    result: subprocess.CompletedProcess[str] = run(
        ROOT / "scripts" / "deploy.sh", s.bin, {"ENV_FILE": missing}
    )
    assert result.returncode == 1
    assert missing in result.stderr
    assert not s.called()
