TOOL_PREFIX	:= arm-linux-gnueabihf
LD= $(TOOL_PREFIX)-ld
CC= $(TOOL_PREFIX)-gcc
OBJCOPY= $(TOOL_PREFIX)-objcopy
LIBEVDEV= $(shell pkg-config --cflags --libs libevdev)

CFLAGS = -Wall -Wextra -Werror -pedantic

FILES_C=main.o uinput_api.o 

PROGRAM_C=test_api 


.PHONY: all
all: $(PROGRAM_C)

$(PROGRAM_C): $(FILES_C)
	$(CC) $(LIBEVDEV) -pthread -g $(CFLAGS) -o $@ $^

uinput_api.o : uinput_api.h
main.o : uinput_api.h

module: module_python

module_python: 
	./compile_module.sh

%.o: %.c
	$(CC) $(LIBEVDEV) -pthread -o $@ -I. -c $< -marm

.PHONE: clean
#Clean
clean :
	rm -rf *.o *.bin
	rm -f $(PROGRAM_C)
	rm -rf python_module_wrap.c _python_module.so python_module.py ../_python_module.so ../python_module.py
	rm -rf __pycache__