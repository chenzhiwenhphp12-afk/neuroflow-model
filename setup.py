"""
NeuroFlow Model - pybind11 Python Bindings
===========================================
Cross-platform setup: Linux (x86/ARM), macOS, Windows
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that uses CMake + pybind11 to build C++ core."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext that invokes CMake for cross-platform compilation."""

    user_options = build_ext.user_options + [
        ('cmake-args=', None, 'Additional CMake arguments'),
        ('parallel=', 'j', 'Number of parallel build jobs'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cmake_args = None
        self.parallel = None

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
            "-DNEUROFLOW_BUILD_PYTHON=ON",
        ]

        # Platform-specific flags
        system = platform.system()
        machine = platform.machine().lower()

        if system == "Windows":
            cmake_args += ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"]
            if "arm" in machine:
                cmake_args += ["-DCMAKE_SYSTEM_PROCESSOR=ARM64"]
        elif system == "Darwin":
            if "arm" in machine:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=arm64"]
                os.environ.setdefault("CMAKE_OSX_ARCHITECTURES", "arm64")
            else:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=x86_64"]
        else:  # Linux
            if "aarch64" in machine or "arm" in machine:
                cmake_args += ["-DCMAKE_SYSTEM_PROCESSOR=aarch64"]
            else:
                cmake_args += ["-DCMAKE_CXX_FLAGS=-mavx2 -mfma"]

        # User cmake args
        if self.cmake_args:
            cmake_args.extend(self.cmake_args.split())

        # Parallel build
        build_args = ["--config", "Release"]
        if self.parallel:
            build_args += [f"-j{self.parallel}"]
        else:
            import multiprocessing
            build_args += [f"-j{multiprocessing.cpu_count()}"]

        # Build directory
        build_temp = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_temp, exist_ok=True)

        # Configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
        )

        # Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )

        print(f"✓ NeuroFlow C++ core built for {system} ({machine})")


# Read long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="neuroflow",
    version="2.1.0",
    author="chenzhiwenhphp12-afk",
    author_email="chenzhiwenhphp12@gmail.com",
    description="Brain-inspired multimodal neural network - 43K params, 0.40ms inference, pure C++17",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenzhiwenhphp12-afk/neuroflow-model",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="neural-network, brain-inspired, multimodal, lightweight, cpp, simd, edge-ai",
    packages=find_packages(include=["neuroflow", "neuroflow.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pybind11>=2.10",
    ],
    extras_require={
        "train": ["torch>=2.0", "unsloth", "datasets"],
        "dev": ["pytest", "pytest-benchmark", "black", "ruff"],
        "all": [
            "torch>=2.0", "unsloth", "datasets",
            "pytest", "pytest-benchmark", "black", "ruff",
        ],
    },
    ext_modules=[
        CMakeExtension(
            "neuroflow._core",
            sourcedir="cpp_core",
        )
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    # Fallback: pure Python mode when C++ can't compile
    entry_points={
        "console_scripts": [
            "neuroflow-bench=neuroflow.cli:benchmark",
        ],
    },
)
