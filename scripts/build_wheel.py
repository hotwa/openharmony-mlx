#!/usr/bin/env python3
"""
Build wheel package for distribution.

This script handles building both the source distribution and wheel
with pre-compiled Metal extension for Apple Silicon.

Usage:
    python scripts/build_wheel.py           # Build wheel locally
    python scripts/build_wheel.py --upload  # Build and upload to PyPI
    python scripts/build_wheel.py --test    # Build and test locally
    python scripts/build_wheel.py --sdist   # Build source distribution only

Requirements for building:
    - macOS 13.0+ (Ventura or later)
    - Apple Silicon (M1/M2/M3/M4)
    - Xcode with MetalToolchain
    - Python 3.12
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Colors for output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"{Colors.OKCYAN}$ {' '.join(cmd)}{Colors.ENDC}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env or os.environ,
        check=False,
        capture_output=capture,
        text=True,
    )


def check_dependencies() -> tuple[bool, dict]:
    """Check if build dependencies are installed."""
    errors = []
    info = {}

    # Check Python version
    if sys.version_info < (3, 12):
        errors.append(f"Python 3.12+ required, got {sys.version}")
    else:
        info["Python"] = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Check macOS
    if sys.platform != "darwin":
        errors.append("This build script only supports macOS")
    else:
        import platform
        info["macOS"] = platform.mac_ver()[0]
        info["Kernel"] = platform.release()

    # Check platform
    if platform.machine() not in ["arm64", "aarch64"]:
        errors.append("Metal builds require Apple Silicon (arm64)")
    else:
        info["Arch"] = platform.machine()

    # Check Xcode command line tools
    result = run_cmd(["xcode-select", "-p"], capture=False)
    if result.returncode != 0:
        errors.append("Xcode command line tools not installed. Run: xcode-select --install")
    else:
        info["Xcode"] = "installed"

    # Check for MetalToolchain
    result = run_cmd(["xcrun", "-find", "metal"], capture=False)
    if result.returncode != 0:
        print(f"{Colors.WARNING}Warning: Metal compiler not found{Colors.ENDC}")
        print("Run: xcodebuild -downloadComponent MetalToolchain")
        info["MetalToolchain"] = "not found"
    else:
        info["MetalToolchain"] = "available"

    if errors:
        print(f"{Colors.FAIL}Errors:{Colors.ENDC}")
        for e in errors:
            print(f"  - {e}")
        return False, info

    return True, info


def install_build_deps() -> bool:
    """Install build dependencies."""
    print(f"\n{Colors.HEADER}Installing build dependencies...{Colors.ENDC}")

    deps = ["build", "twine", "wheel", "pybind11>=2.12"]

    result = run_cmd([sys.executable, "-m", "pip", "install"] + deps)
    if result.returncode != 0:
        print(f"{Colors.FAIL}Failed to install dependencies{Colors.ENDC}")
        return False

    return True


def build_metal_extension() -> bool:
    """Build the Metal extension module."""
    print(f"\n{Colors.HEADER}Building Metal extension...{Colors.ENDC}")

    metal_dir = Path(__file__).parent.parent / "gpt_oss" / "metal"
    build_dir = metal_dir / "build"

    # Clean build directory
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Get pybind11 cmake directory
    result = run_cmd([
        sys.executable, "-c",
        "import pybind11; print(pybind11.get_cmake_dir())"
    ])
    if result.returncode != 0 or not result.stdout:
        print(f"{Colors.FAIL}Failed to find pybind11{Colors.ENDC}")
        return False

    pybind11_dir = result.stdout.strip()

    # Configure with CMake
    result = run_cmd([
        "cmake", "..",
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
    ], cwd=build_dir)

    if result.returncode != 0:
        print(f"{Colors.FAIL}CMake configuration failed{Colors.ENDC}")
        return False

    # Build
    num_cpus = os.cpu_count() or 4
    result = run_cmd(["make", f"-j{num_cpus}"], cwd=build_dir)

    if result.returncode != 0:
        print(f"{Colors.FAIL}Build failed{Colors.ENDC}")
        return False

    # Copy artifacts to package directory
    so_files = list(build_dir.glob("_metal*.so"))
    metallib_files = list(build_dir.glob("default.metallib"))

    if not so_files:
        print(f"{Colors.FAIL}No .so files found in build directory{Colors.ENDC}")
        return False

    # Copy .so file
    for so in so_files:
        dest = metal_dir / so.name
        shutil.copy2(so, dest)
        print(f"  Copied: {so.name}")

    # Copy metallib if found
    for ml in metallib_files:
        dest = metal_dir / ml.name
        shutil.copy2(ml, dest)
        print(f"  Copied: {ml.name}")

    print(f"{Colors.OKGREEN}Metal extension built successfully{Colors.ENDC}")
    return True


def build_wheel(build_dir: Path | None = None) -> bool:
    """Build the wheel package."""
    print(f"\n{Colors.HEADER}Building wheel package...{Colors.ENDC}")

    # Clean old builds
    for d in ["dist", "build", "openharmony_mlx.egg-info"]:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)

    # Build wheel
    result = run_cmd([
        sys.executable, "-m", "build",
        "--wheel",
        "--outdir", str(build_dir or Path("dist"))
    ])

    if result.returncode != 0:
        print(f"{Colors.FAIL}Wheel build failed{Colors.ENDC}")
        return False

    print(f"{Colors.OKGREEN}Wheel built successfully{Colors.ENDC}")
    return True


def build_sdist() -> bool:
    """Build source distribution."""
    print(f"\n{Colors.HEADER}Building source distribution...{Colors.ENDC}")

    # Clean old builds
    for d in ["dist", "build", "openharmony_mlx.egg-info"]:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)

    # Build sdist
    result = run_cmd([
        sys.executable, "-m", "build", "--sdist", "--outdir", "dist"
    ])

    if result.returncode != 0:
        print(f"{Colors.FAIL}Source distribution build failed{Colors.ENDC}")
        return False

    print(f"{Colors.OKGREEN}Source distribution built successfully{Colors.ENDC}")
    return True


def test_wheel(wheel_path: Path) -> bool:
    """Test the built wheel."""
    print(f"\n{Colors.HEADER}Testing wheel: {wheel_path}{Colors.ENDC}")

    # Install the wheel in a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a virtual environment for testing
        venv_dir = Path(tmpdir) / "test_env"
        result = run_cmd([
            sys.executable, "-m", "venv", str(venv_dir)
        ])
        if result.returncode != 0:
            print(f"{Colors.FAIL}Failed to create test environment{Colors.ENDC}")
            return False

        # Install the wheel
        pip = venv_dir / "bin" / "pip"
        result = run_cmd([str(pip), "install", str(wheel_path)])
        if result.returncode != 0:
            print(f"{Colors.FAIL}Failed to install wheel{Colors.ENDC}")
            return False

        # Test import
        python = venv_dir / "bin" / "python"
        result = run_cmd([
            str(python), "-c",
            "import gpt_oss; print(f'Version: {gpt_oss.__version__}')"
        ])
        if result.returncode != 0:
            print(f"{Colors.FAIL}Import test failed{Colors.ENDC}")
            return False

    print(f"{Colors.OKGREEN}Wheel test passed{Colors.ENDC}")
    return True


def upload_to_pypi(repo: str = "pypi") -> bool:
    """Upload packages to PyPI."""
    print(f"\n{Colors.HEADER}Uploading to {repo}...{Colors.ENDC}")

    dist_dir = Path("dist")
    if not dist_dir.exists():
        print(f"{Colors.FAIL}No dist directory found{Colors.ENDC}")
        return False

    packages = list(dist_dir.glob("*"))
    if not packages:
        print(f"{Colors.FAIL}No packages found in dist/{Colors.ENDC}")
        return False

    for p in packages:
        print(f"  {p.name}")

    result = run_cmd([
        sys.executable, "-m", "twine", "upload",
        "--repository", repo,
        str(dist_dir / "*")
    ])

    if result.returncode != 0:
        print(f"{Colors.FAIL}Upload failed{Colors.ENDC}")
        return False

    print(f"{Colors.OKGREEN}Upload successful{Colors.ENDC}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build and publish openharmony-mlx wheel package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                    Build wheel locally
    %(prog)s --upload           Build and upload to PyPI
    %(prog)s --test             Build and run tests
    %(prog)s --sdist            Build source distribution only
    %(prog)s --dev              Build Metal extension for development
        """
    )

    parser.add_argument(
        "--upload", action="store_true",
        help="Upload to PyPI after building"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test the built wheel"
    )
    parser.add_argument(
        "--sdist", action="store_true",
        help="Build source distribution only"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Build Metal extension for development (no wheel)"
    )
    parser.add_argument(
        "--repo", default="pypi",
        help="PyPI repository to upload to (default: pypi)"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output directory for built packages"
    )

    args = parser.parse_args()

    print(f"{Colors.BOLD}OpenHarmony MLX Wheel Builder{Colors.ENDC}")
    print("=" * 50)

    # Check dependencies and show system info
    ok, info = check_dependencies()
    if not ok:
        sys.exit(1)

    print(f"\n{Colors.OKBLUE}System Information:{Colors.ENDC}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  Minimum macOS: 13.0 (Ventura)")
    print(f"  Tested on: macOS {info.get('macOS', 'unknown')}")
    print()

    # Install build deps
    if not install_build_deps():
        sys.exit(1)

    output_dir = args.output or Path("dist")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dev:
        # Development mode: just build the Metal extension
        if not build_metal_extension():
            sys.exit(1)
        print(f"\n{Colors.OKGREEN}Development build complete!{Colors.ENDC}")
        return

    # Build Metal extension first
    if not build_metal_extension():
        print(f"{Colors.WARNING}Metal extension build failed, continuing anyway...{Colors.ENDC}")

    # Build the package
    if args.sdist:
        if not build_sdist():
            sys.exit(1)
    else:
        if not build_wheel(output_dir):
            sys.exit(1)

    # Test the wheel
    if args.test:
        wheels = list(output_dir.glob("*.whl"))
        if wheels:
            if not test_wheel(wheels[0]):
                sys.exit(1)
        else:
            print(f"{Colors.WARNING}No wheel found to test{Colors.ENDC}")

    # List built packages
    print(f"\n{Colors.OKGREEN}Built packages:{Colors.ENDC}")
    for p in output_dir.iterdir():
        print(f"  {p.name}")

    # Upload if requested
    if args.upload:
        if not upload_to_pypi(args.repo):
            sys.exit(1)

    print(f"\n{Colors.OKGREEN}Done!{Colors.ENDC}")


if __name__ == "__main__":
    main()
