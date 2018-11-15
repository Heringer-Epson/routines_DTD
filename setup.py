import shutil
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("core_funcs",
             ["./lib/core_funcs.pyx"],
             libraries=["m"],
             extra_compile_args = ["-ffast-math"])]

setup(
  name = "core_funcs",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

shutil.move("./core_funcs.so", "./lib/core_funcs.so")
