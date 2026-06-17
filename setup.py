import sys
import platform
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

_machine = platform.machine().lower()
_is_windows = sys.platform == "win32"
_is_x86 = _machine in ("x86_64", "amd64", "i386", "i686")

if _is_windows:
    compile_args = ["/fp:fast", "/O2"]
    if _is_x86:
        compile_args.append("/arch:AVX2")  # implies FMA on MSVC; omit on ARM64 Windows
else:
    compile_args = ["-ffast-math", "-O3", "-march=native"]

extensions = [
    Extension(
        name="pact.lib.calculus",
        sources=["pact/lib/calculus.pyx", "pact/lib/ctools.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
