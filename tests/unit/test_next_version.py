"""The release workflow's version decision.

This runs on every merge to main and picks a number that then cannot be
reused, so the rules are worth pinning here rather than discovering them from
a release that went out wrong.
"""

import importlib.util
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / ".github" / "scripts" / "next_version.py"
_spec = importlib.util.spec_from_file_location("next_version", _SCRIPT)
assert _spec is not None and _spec.loader is not None
next_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(next_version)


def pyproject(version: str, engine: str = "0.5.7") -> str:
    return f'''
[project]
name = "langchain-infino"
version = "{version}"
dependencies = [
    "langchain-core>=0.3",
    "infino>={engine},<0.6",
]
'''


def test_engine_minor_move_is_a_minor_release() -> None:
    kind, version, _ = next_version.decide(
        pyproject("0.3.0", "0.5.7"), pyproject("0.3.0", "0.6.0"), []
    )
    assert (kind, version) == ("minor", "0.4.0")


def test_engine_major_move_is_a_major_release() -> None:
    kind, version, _ = next_version.decide(
        pyproject("0.3.0", "0.5.7"), pyproject("0.3.0", "1.0.0"), []
    )
    assert (kind, version) == ("major", "1.0.0")


def test_engine_patch_move_is_a_patch_release() -> None:
    kind, version, _ = next_version.decide(
        pyproject("0.3.0", "0.5.7"), pyproject("0.3.0", "0.5.8"), []
    )
    assert (kind, version) == ("patch", "0.3.1")


def test_an_unchanged_engine_still_releases_a_patch() -> None:
    # The push carried something, even if not an engine bump.
    kind, version, _ = next_version.decide(
        pyproject("0.3.0"), pyproject("0.3.0"), []
    )
    assert (kind, version) == ("patch", "0.3.1")


def test_an_unknown_previous_revision_is_treated_as_a_patch() -> None:
    kind, version, reason = next_version.decide(None, pyproject("0.3.0"), [])
    assert (kind, version) == ("patch", "0.3.1")
    assert "no previous revision" in reason


def test_tags_raise_the_base_above_a_stale_pyproject() -> None:
    # pyproject is not rewritten on release, so tags are what move forward.
    kind, version, _ = next_version.decide(
        pyproject("0.3.0"), pyproject("0.3.0"), ["v0.3.0", "v0.4.0"]
    )
    assert (kind, version) == ("patch", "0.4.1")


def test_pyproject_wins_when_it_is_ahead_of_the_tags() -> None:
    # A version bumped by hand in the tree must not be released backwards.
    _, version, _ = next_version.decide(
        pyproject("0.5.0"), pyproject("0.5.0"), ["v0.3.0"]
    )
    assert version == "0.5.1"


def test_a_minor_bump_resets_the_patch_component() -> None:
    _, version, _ = next_version.decide(
        pyproject("0.3.4", "0.5.7"), pyproject("0.3.4", "0.6.0"), ["v0.3.4"]
    )
    assert version == "0.4.0"


def test_a_major_bump_resets_minor_and_patch() -> None:
    _, version, _ = next_version.decide(
        pyproject("0.3.4", "0.5.7"), pyproject("0.3.4", "1.2.3"), ["v0.3.4"]
    )
    assert version == "1.0.0"


def test_unparseable_tags_are_ignored_not_fatal() -> None:
    # The repo may carry tags that are not releases.
    _, version, _ = next_version.decide(
        pyproject("0.3.0"), pyproject("0.3.0"), ["nightly", "v0.3.0", "junk"]
    )
    assert version == "0.3.1"


def test_a_missing_engine_requirement_is_not_fatal() -> None:
    without = '[project]\nversion = "0.3.0"\ndependencies = ["langchain-core>=0.3"]\n'
    kind, version, _ = next_version.decide(without, without, [])
    assert (kind, version) == ("patch", "0.3.1")


def test_the_engine_floor_is_read_from_the_requirement() -> None:
    assert next_version.engine_floor(pyproject("0.3.0", "0.5.7")) == (0, 5, 7)


def test_a_project_without_a_version_is_an_error() -> None:
    with pytest.raises(ValueError, match="no .version = "):
        next_version.project_version('[project]\nname = "x"\n')


def test_a_non_numeric_project_version_is_an_error() -> None:
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        next_version.project_version('[project]\nversion = "0.3.0rc1"\n')


def test_a_change_under_the_package_ships() -> None:
    assert next_version.is_shipping_change(["langchain_infino/vectorstores.py"])


def test_a_change_to_pyproject_ships() -> None:
    # Dependency bounds and metadata reach the user through it.
    assert next_version.is_shipping_change(["pyproject.toml"])


def test_docs_ci_and_tests_do_not_ship() -> None:
    assert not next_version.is_shipping_change(
        [
            "README.md",
            "LICENSE",
            "Makefile",
            ".gitignore",
            ".github/workflows/release.yml",
            ".github/scripts/next_version.py",
            "tests/unit/test_next_version.py",
            "tests/integration/test_compliance.py",
        ]
    )


def test_one_shipping_file_among_many_is_enough() -> None:
    assert next_version.is_shipping_change(
        ["README.md", ".github/workflows/ci.yml", "langchain_infino/cache.py"]
    )


def test_no_changed_files_ships_nothing() -> None:
    assert not next_version.is_shipping_change([])
    assert not next_version.is_shipping_change(["", "  "])


def test_a_lookalike_path_does_not_ship() -> None:
    # Only the package directory itself, not something merely prefixed by it.
    assert not next_version.is_shipping_change(["langchain_infino_extras/x.py"])
    assert not next_version.is_shipping_change(["docs/pyproject.toml"])
