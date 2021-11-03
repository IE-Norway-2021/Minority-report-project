swig3.0 -python python_module.i
arm-linux-gnueabihf-gcc -c python_module.c python_module_wrap.c -I/usr/include/python3.7m
arm-linux-gnueabihf-ld -shared python_module.o python_module_wrap.o -o _python_module.so
cp _python_module.so ../_python_module.so
cp python_module.py ../python_module.py