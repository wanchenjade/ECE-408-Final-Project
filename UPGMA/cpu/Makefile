CC = g++
CFLAGS = -O3 -c -Wall -std=c++11
LDFLAGS =
SRCS = upgma.cpp
OBJS = $(SRCS:.cpp=.o)

all: upgma

upgma: upgma.o
	$(CC) $(LDFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr *.o upgma 

.PHONY: all clean 
