TOOL_PREFIX	:= arm-linux-gnueabihf
LD= $(TOOL_PREFIX)-ld
CC= $(TOOL_PREFIX)-gcc
OBJCOPY= $(TOOL_PREFIX)-objcopy

CFLAGS = -Wall -Wextra -Werror -pedantic

FILES=main.o uinput_api.o 
PROGRAM=main


all: $(PROGRAM)

$(PROGRAM): $(FILES)
	$(CC) -g $(CFLAGS) -o $@ $^

uinput_api.o : uinput_api.h
main.o : uinput_api.h

%.o: %.c
	$(CC) -o $@ -I. -c $< -marm


#Clean
clean :
	rm -rf *.o *.bin
	rm -f $(PROGRAM)