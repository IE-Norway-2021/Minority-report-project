TOOL_PREFIX	:= aarch64-linux-gnu
LD= $(TOOL_PREFIX)-ld
CC= $(TOOL_PREFIX)-gcc
OBJCOPY= $(TOOL_PREFIX)-objcopy
LIBEVDEV= $(shell pkg-config --cflags --libs libevdev)

CFLAGS = -Wall -Wextra -Werror -pedantic

FILES_C=main_test.o uinput_api.o 

PROGRAM_C=test_api 


.PHONY: all
all: $(PROGRAM_C)

$(PROGRAM_C): $(FILES_C)
	$(CC) -g $(CFLAGS) -o $@ $^ $(LIBEVDEV)

uinput_api.o : uinput_api.h
main.o : uinput_api.h

module_pi: module_python_pi

module_python_pi: 
	./compile_module.sh

module_jn: module_python_jn

module_python_jn:
	./compile_module.sh -jn

%.o: %.c
	$(CC) -o $@ -I. -c $< $(LIBEVDEV)

.PHONE: clean
#Clean
clean :
	rm -rf *.o *.bin
	rm -f $(PROGRAM_C)
	rm -rf python_module_wrap.c _python_module.so python_module.py ../_python_module.so ../python_module.py
	rm -rf __pycache__ ../__pycache__