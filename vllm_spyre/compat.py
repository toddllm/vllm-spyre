"""Version compatibility checks for vllm-spyre.

This module provides guards that verify the installed vLLM version is
within the supported band. The checks are called from register(), NOT
at import time, to avoid breaking environments that import the package
for inspection without running the full engine.
"""

import importlib.metadata
import logging
import os

logger = logging.getLogger(__name__)

# The vLLM version band that this release of vllm-spyre targets.
# Format: (min_version_inclusive, max_version_exclusive)
# For PR-759 work: we target vllm 0.15.x
VLLM_MIN_VERSION = "0.15.0"
VLLM_MAX_VERSION = "0.16.0"

_compat_checked = False


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of ints for comparison.

    Handles dev/rc/post suffixes by stripping them.
    Examples:
        "0.15.1" -> (0, 15, 1)
        "0.15.1.dev123" -> (0, 15, 1)
        "0.16.0rc1" -> (0, 16, 0)
    """
    # Strip common suffixes
    base = version_str.split("+")[0]  # strip local version
    base = base.split(".dev")[0]
    base = base.split("rc")[0]
    base = base.split("post")[0]
    parts = []
    for part in base.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def check_vllm_version() -> None:
    """Check that the installed vLLM version is within the supported band.

    Called from register(). Idempotent — only runs once.

    Raises:
        RuntimeError: If the vLLM version is outside the supported band
            and VLLM_SPYRE_SKIP_VERSION_CHECK is not set.
    """
    global _compat_checked
    if _compat_checked:
        return

    # Allow users to bypass the check for testing/development
    if os.getenv("VLLM_SPYRE_SKIP_VERSION_CHECK", "0") == "1":
        logger.info(
            "vLLM version check skipped (VLLM_SPYRE_SKIP_VERSION_CHECK=1)"
        )
        _compat_checked = True
        return

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            "vLLM package not found. Version check skipped. "
            "This may cause runtime errors."
        )
        _compat_checked = True
        return

    installed = _parse_version_tuple(vllm_version)
    min_ver = _parse_version_tuple(VLLM_MIN_VERSION)
    max_ver = _parse_version_tuple(VLLM_MAX_VERSION)

    if installed < min_ver or installed >= max_ver:
        msg = (
            f"vllm-spyre requires vllm>={VLLM_MIN_VERSION},<{VLLM_MAX_VERSION} "
            f"but found vllm=={vllm_version}. "
            f"Set VLLM_SPYRE_SKIP_VERSION_CHECK=1 to bypass this check."
        )
        raise RuntimeError(msg)

    logger.debug(
        "vLLM version check passed: %s (band %s–%s)",
        vllm_version, VLLM_MIN_VERSION, VLLM_MAX_VERSION,
    )
    _compat_checked = True
