CC = nvcc
CFLAGS = -O3 -c -std=c++11 -arch=sm_50 -I/usr/local/cuda/include 
LDFLAGS = -lcudart -L/usr/local/cuda/lib64
SRCS = nj.cu
OBJS = $(SRCS:.cu=.o)

all: nj

nj: nj.o
	$(CC) -o $@ $< $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr *.o nj 

.PHONY: all clean 
