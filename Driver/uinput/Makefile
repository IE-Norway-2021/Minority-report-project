TOOL_PREFIX	:= arm-linux-gnueabihf
LD= $(TOOL_PREFIX)-ld
CC= $(TOOL_PREFIX)-gcc
CPP= $(TOOL_PREFIX)-g++ 
OBJCOPY= $(TOOL_PREFIX)-objcopy

CFLAGS = -Wall -Wextra -Werror -pedantic

FILES_C=main.o uinput_api.o 
FILES_CPP= uinput_api.o python_module.o

PROGRAM_C=main 
PROGRAM_CPP=python_module


.PHONY: all
all: $(PROGRAM_C)

$(PROGRAM_C): $(FILES_C)
	$(CC) -g $(CFLAGS) -o $@ $^

uinput_api.o : uinput_api.h
main.o : uinput_api.h

module: $(PROGRAM_CPP)

$(PROGRAM_CPP): $(FILES_CPP)
	$(CPP)  -g $(CFLAGS) -o $@ $^


%.o: %.c
	$(CC) -o $@ -I. -c $< -marm

%.o: %.cpp
	$(CPP) -I /usr/include/python3.7m/ -o $@ -c $<

.PHONE: clean
#Clean
clean :
	rm -rf *.o *.bin
	rm -f $(PROGRAM_C)
	rm -f $(PROGRAM_CPP)