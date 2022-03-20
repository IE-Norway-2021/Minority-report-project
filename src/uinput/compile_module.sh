#!/bin/bash
#
#    compile_module    -   compiles the python module using swig
#
#    usage:    compile_module
#
#    option : use -jn to compile for the jetson nano and not the pi
#
#  David González León, Jade Gröli    2022-01-11
# TODO use swig4.0 et python3.9 pour l'include
if [[ $1 = "-jn" ]]; then  # compiling for the jetson nano
    swig -python python_module.i
    aarch64-linux-gnu-gcc -fPIC -c python_module.c uinput_api.c python_module_wrap.c -I/usr/include/python3.6
else 
    swig4.0 -python python_module.i
    aarch64-linux-gnu-gcc -fPIC -c python_module.c uinput_api.c python_module_wrap.c -I/usr/include/python3.9
fi
aarch64-linux-gnu-ld -shared -fPIC python_module.o uinput_api.o python_module_wrap.o -o _python_module.so
cp _python_module.so ../_python_module.so
cp python_module.py ../python_module.py