"""Decide the release version for a merge to main.

The size of the bump follows the engine: `langchain-infino` exists to track
`infino`, so a release that carries a new engine minor is itself a minor, and a
new engine major is a major. Everything else is a patch.

Version numbers are read from the tree and from tags rather than from the
index. Tags are what a successful publish leaves behind, so the base is the
highest of the two — which keeps working if a release fails partway, and does
not require the index to be reachable to decide anything.

Regex rather than `tomllib`, because this also runs on the Python floor the
package supports, where `tomllib` does not exist.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional

# PEP 585 generics work at runtime from 3.9, the floor this must run on.
Version = tuple[int, int, int]

_PROJECT_VERSION = re.compile(r'(?m)^version = "([^"]*)"')
_ENGINE_PIN = re.compile(r"infino\s*>=\s*(\d+)\.(\d+)\.(\d+)")
_TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> Optional[Version]:
    """Parse a bare ``MAJOR.MINOR.PATCH``, ignoring any pre-release suffix."""
    match = _TAG.match(text.strip())
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def project_version(pyproject: str) -> Version:
    """The version declared in ``[project]``."""
    match = _PROJECT_VERSION.search(pyproject)
    if match is None:
        raise ValueError("no `version = ...` found in pyproject.toml")
    version = parse_version(match[1])
    if version is None:
        raise ValueError(f"project version is not MAJOR.MINOR.PATCH: {match[1]!r}")
    return version


def engine_floor(pyproject: str) -> Optional[Version]:
    """The lower bound of the ``infino`` requirement, if it declares one.

    ``None`` when absent — which is how a first release, or a revision from
    before the dependency existed, presents itself.
    """
    match = _ENGINE_PIN.search(pyproject)
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def base_version(pyproject: str, tags: list[str]) -> Version:
    """The version to count up from: the highest of the tree and the tags."""
    candidates = [project_version(pyproject)]
    candidates.extend(v for v in (parse_version(t) for t in tags) if v is not None)
    return max(candidates)


def bump_kind(old: Optional[Version], new: Optional[Version]) -> str:
    """How far the engine moved between two revisions.

    An engine that did not move, or cannot be compared because one side has no
    pin, is a patch: the release still carries whatever else changed.
    """
    if old is None or new is None or new == old:
        return "patch"
    if new[0] != old[0]:
        return "major"
    if new[1] != old[1]:
        return "minor"
    return "patch"


def apply_bump(base: Version, kind: str) -> Version:
    major, minor, patch = base
    if kind == "major":
        return (major + 1, 0, 0)
    if kind == "minor":
        return (major, minor + 1, 0)
    if kind == "patch":
        return (major, minor, patch + 1)
    raise ValueError(f"unknown bump kind: {kind!r}")


def render(version: Version) -> str:
    return ".".join(str(part) for part in version)


def decide(
    old_pyproject: Optional[str], new_pyproject: str, tags: list[str]
) -> tuple[str, str, str]:
    """Return ``(bump_kind, next_version, reason)`` for a merge to main."""
    new_floor = engine_floor(new_pyproject)
    old_floor = engine_floor(old_pyproject) if old_pyproject is not None else None

    kind = bump_kind(old_floor, new_floor)
    base = base_version(new_pyproject, tags)
    version = apply_bump(base, kind)

    if old_pyproject is None:
        reason = "no previous revision to compare; treating as a patch"
    elif old_floor == new_floor or old_floor is None or new_floor is None:
        reason = "engine requirement unchanged"
    else:
        reason = (
            f"engine requirement moved {render(old_floor)} -> {render(new_floor)}"
        )
    return kind, render(version), f"{reason}; {render(base)} -> {render(version)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-pyproject", required=True)
    # Absent when the previous revision is unknown (a first push, or history
    # rewritten under us).
    parser.add_argument("--old-pyproject")
    parser.add_argument(
        "--tags", default="", help="whitespace-separated existing release tags"
    )
    args = parser.parse_args()

    with open(args.new_pyproject, encoding="utf-8") as handle:
        new_pyproject = handle.read()
    old_pyproject = None
    if args.old_pyproject:
        with open(args.old_pyproject, encoding="utf-8") as handle:
            old_pyproject = handle.read()

    kind, version, reason = decide(old_pyproject, new_pyproject, args.tags.split())

    print(f"bump={kind} version={version} ({reason})")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"bump={kind}\n")
            handle.write(f"version={version}\n")
            handle.write(f"reason={reason}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
