#!/bin/bash
#
#    compile_module    -   compiles the python module using swig
#
#    usage:    compile_module
#
#  David González León, Jade Gröli    2022-01-11
swig4.0 -python python_module.i
aarch64-linux-gnu-gcc -fPIC -c python_module.c uinput_api.c python_module_wrap.c -I/usr/include/python3.9
aarch64-linux-gnu-ld -shared -fPIC python_module.o uinput_api.o python_module_wrap.o -o _python_module.so
cp _python_module.so ../_python_module.so
cp python_module.py ../python_module.py