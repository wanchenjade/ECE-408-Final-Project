CC = nvcc
CFLAGS = -c -std=c++11 -arch=sm_50 -O3 -I/usr/local/cuda/include 
LDFLAGS = -lcudart -L/usr/local/cuda/lib64
SRCS = upgma.cu
OBJS = $(SRCS:.cu=.o)

all: upgma

upgma: upgma.o
	$(CC) -o $@ $< $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -fr *.o upgma 

.PHONY: all clean 
