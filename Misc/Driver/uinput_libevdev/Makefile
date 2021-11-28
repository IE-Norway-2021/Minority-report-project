TOOL_PREFIX	:= arm-linux-gnueabihf
LD= $(TOOL_PREFIX)-ld
CC= $(TOOL_PREFIX)-gcc
OBJCOPY= $(TOOL_PREFIX)-objcopy
LIBEVDEV= $(shell pkg-config --cflags --libs libevdev)

CFLAGS = -Wall -Wextra -pedantic

.PHONY: all
all: module_python

module_python:
	$(shell swig3.0 $(pkg-config --cflags --libs libevdev) -python python_module.i)
	$(CC) -c -g $(LIBEVDEV) $(CFLAGS) python_module.c python_module_wrap.c -I/usr/include/python3.7m
	$(LD) -shared python_module.o python_module_wrap.o -o _python_module.so
	$(shell cp _python_module.so ../_python_module.so)
	$(shell cp python_module.py ../python_module.py)

%.o: %.c
	$(CC) $(LIBEVDEV) -pthread -o $@ -I. -c $< -marm

.PHONE: clean
#Clean
clean :
	rm -rf *.o *.bin
	rm -rf python_module_wrap.c _python_module.so python_module.py ../_python_module.so ../python_module.py
	rm -rf __pycache__ ../__pycache__