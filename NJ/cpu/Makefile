CC = g++
CFLAGS = -O3 -g -c -Wall -std=c++11
LDFLAGS =
SRCS = nj.cpp
OBJS = $(SRCS:.cpp=.o)

all: nj

nj: nj.o
	$(CC) $(LDFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr *.o nj

.PHONY: all clean 
