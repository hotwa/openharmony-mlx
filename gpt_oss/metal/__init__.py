"""
Metal backend for GPT-OSS on Apple Silicon.

This module provides GPU-accelerated inference using Apple's Metal framework.

Usage:
    from gpt_oss.metal import Model, Context, Sampler

Requirements:
    - macOS 13.0+ (Ventura or later)
    - Apple Silicon (M1/M2/M3/M4 series)
    - Xcode with MetalToolchain
"""

from importlib import import_module as _im
from pathlib import Path as _Path
import sys as _sys
import os as _os

__version__ = "0.1.0"

# Package directory
_PACKAGE_DIR = _Path(__file__).parent


def _find_extension() -> _Path:
    """Find the pre-compiled Metal extension module."""
    # Look for _metal*.so files in the package directory
    so_files = list(_PACKAGE_DIR.glob("_metal*.so"))
    if so_files:
        return so_files[0]

    # Look in build directory (development mode)
    build_dir = _PACKAGE_DIR / "build"
    if build_dir.exists():
        so_files = list(build_dir.glob("**/_metal*.so"))
        if so_files:
            return so_files[0]

    return None


def _load_extension():
    """Load the Metal extension or provide helpful error message."""
    ext_path = _find_extension()

    if ext_path is None:
        raise ImportError(
            f"Metal extension not found.\n\n"
            f"Expected location: {_PACKAGE_DIR}/_metal.cpython-*.so\n\n"
            "To fix this:\n"
            "1. For pre-built package: pip install --force-reinstall openharmony-mlx\n"
            "2. For development: cd gpt_oss/metal && mkdir -p build && cd build && \\\n"
            "   cmake .. && make\n"
            "3. Install MetalToolchain: xcodebuild -downloadComponent MetalToolchain"
        )

    # Add the directory to sys.path if needed
    ext_dir = str(ext_path.parent)
    if ext_dir not in _sys.path:
        _sys.path.insert(0, ext_dir)

    try:
        # Import the compiled extension
        _ext = _im(f"gpt_oss.metal._metal")
        return _ext
    except ImportError as e:
        raise ImportError(
            f"Failed to load Metal extension from {ext_path}: {e}\n\n"
            "The extension may have been built for a different Python version.\n"
            "Please rebuild: cd gpt_oss/metal && mkdir -p build && cd build && cmake .. && make"
        ) from e


def _check_metal_availability():
    """Check if Metal is available on this system."""
    import platform

    # Check if running on macOS
    if platform.system() != "Darwin":
        return False, "Metal is only available on macOS"

    # Check if running on Apple Silicon
    machine = platform.machine()
    if machine not in ["arm64", "aarch64"]:
        return False, f"Metal requires Apple Silicon (arm64), got {machine}"

    # Check macOS version
    version = platform.mac_ver()[0]
    major, minor = version.split(".")[:2]
    if int(major) < 13:
        return False, f"Metal requires macOS 13.0+, got {version}"

    return True, "Metal is available"


# Try to load the extension
try:
    _ext = _load_extension()

    # Export public API
    __all__ = [
        "Model",
        "Context",
        "Sampler",
        "gptoss_status_to_str",
        "gptoss_status_success",
        "gptoss_status_no_memory",
        "gptoss_status_io_error",
        "gptoss_status_invalid_argument",
        "gptoss_status_not_implemented",
        "gptoss_status_internal_error",
        "gptoss_status_unsupported_hardware",
    ]

    # Import public symbols from extension
    for _name in _ext.__dict__:
        if not _name.startswith("_"):
            globals()[_name] = _ext.__dict__[_name]
            if _name not in __all__:
                __all__.append(_name)

    del _ext

    # Check Metal availability on import
    _available, _msg = _check_metal_availability()
    if not _available:
        import warnings
        warnings.warn(f"Metal not available: {_msg}", RuntimeWarning, stacklevel=2)

except ImportError as _e:
    # Extension failed to load - still export the API for type checking
    import warnings
    warnings.warn(
        f"Failed to load Metal extension: {_e}\n\n"
        "The Metal backend will not be available until this is resolved.",
        ImportWarning,
        stacklevel=2
    )

    # Export placeholders for static analysis
    class _Placeholder:
        """Placeholder class when Metal extension is unavailable."""
        pass

    Model = _Placeholder
    Context = _Placeholder
    Sampler = _Placeholder
    gptoss_status_to_str = lambda x: "unknown"

    __all__ = [
        "Model",
        "Context",
        "Sampler",
        "gptoss_status_to_str",
        "check_metal_availability",
    ]


def check_metal_availability() -> tuple[bool, str]:
    """Check if Metal is available on this system.

    Returns:
        Tuple of (is_available, message)
    """
    return _check_metal_availability()
