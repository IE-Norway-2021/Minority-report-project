CFLAGS = -Wall -Wextra -pedantic
CFLAGS_ALL = $(CFLAGS)

LDFLAGS =
LDFLAGS_ALL = $(LDFLAGS)

PROGRAM = main

OBJS = uinput_api.o main.o

.PHONY: all clean

all: $(PROGRAM)

%.o : %.c 
	$(CC) $(CFLAGS_ALL) -c -o $@ $<

$(PROGRAM) : % : $(OBJS) 
	$(CC) $(LDFLAGS_ALL) $(CFLAGS_ALL) -o $@ $<

clean:
	rm -f $(PROGRAM)
	rm -f *.o