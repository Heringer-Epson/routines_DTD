import shutil
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- core_funcs.pyx -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

ext_modules=[Extension("core_funcs",
             ["./lib/core_funcs.pyx"],
             libraries=["m"],
             extra_compile_args = ["-ffast-math"])]

setup(
  name = "core_funcs",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

shutil.move("./core_funcs.so", "./lib/core_funcs.so")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- DTD_gen.c -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

ext_modules=[Extension("DTD_gen", ["./lib/DTD_gen.c"])]

setup(
  name = "DTD_gen",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)

shutil.move("./DTD_gen.so", "./lib/DTD_gen.so")

#Previous Makefile
#all: ./lib/DTD_gen.c
#	gcc -shared -o ./lib/DTD_gen.so -fPIC ./lib/DTD_gen.c
#
#clean:
#	rm ./lib/*.pyc
#	rm ./lib/*.so
