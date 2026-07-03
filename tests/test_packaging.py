"""Contract tests for the packaging manifest (``pyproject.toml``).

These guard the exact failure modes the 0.4.x audit caught, so they cannot
regress silently:

* the manifest must stay valid TOML — a stray unclosed string in
  ``[tool.setuptools.packages.find]`` previously made ``pip install -e .``
  crash with a ``TOMLDecodeError`` before anything else could run (P0-1);
* the documented optional-dependency extras must keep existing, so the install
  commands in ``README.md`` and the Colab notebooks
  (``voicelegacy[similarity]``, ``[finetune]``, ``[deepfilter]``, ``[all]``)
  never silently break again (P0-2);
* the declared version must match ``voicelegacy.__version__`` (the notebooks
  pin ``@v<version>`` and print ``__version__``; a mismatch is a shipping bug);
* the coverage floor must stay enforced (P0-5).

They read the real repository manifest, not a fixture, on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

import voicelegacy

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Extras documented in README.md and the notebook install cells. If any of
# these disappears, an install command somewhere in the docs/notebooks breaks.
DOCUMENTED_FEATURE_EXTRAS = ("similarity", "finetune", "deepfilter", "eval", "all")


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Parse the real pyproject.toml. Parsing is itself the P0-1 guard."""
    with open(PYPROJECT_PATH, "rb") as fh:
        return tomllib.load(fh)


class TestPyprojectContract:
    def test_manifest_is_valid_toml(self, manifest: dict) -> None:
        assert manifest["project"]["name"] == "voicelegacy"

    def test_version_matches_package_dunder(self, manifest: dict) -> None:
        assert manifest["project"]["version"] == voicelegacy.__version__

    @pytest.mark.parametrize("extra", DOCUMENTED_FEATURE_EXTRAS)
    def test_documented_extra_exists_and_nonempty(self, manifest: dict, extra: str) -> None:
        extras = manifest["project"]["optional-dependencies"]
        assert extra in extras, f"extra '{extra}' missing from [project.optional-dependencies]"
        assert extras[extra], f"extra '{extra}' is declared but empty"

    def test_all_extra_is_union_of_feature_extras(self, manifest: dict) -> None:
        # `all` must pull in the feature extras (kept DRY via a self-reference)
        # so it cannot drift out of sync with them.
        all_spec = " ".join(manifest["project"]["optional-dependencies"]["all"])
        for feature in ("similarity", "finetune", "deepfilter", "eval"):
            assert feature in all_spec, f"'all' extra does not include '{feature}'"

    def test_coverage_floor_is_enforced(self, manifest: dict) -> None:
        addopts = manifest["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov-fail-under=80" in addopts
