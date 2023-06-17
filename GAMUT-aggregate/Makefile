CUDA = nvcc
ARCH = sm_75

.PHONY : all clean

all : test.out

test.out : launch.o main.o kernel.o alloc.o timer.o heap.o
	$(CUDA) -arch=$(ARCH) $^ -o $@

main.o : main.cu
	$(CUDA) -arch=$(ARCH) -dc $< -o $@

timer.o : gpu_timer.cc
	$(CUDA) -arch=$(ARCH) -c $< -o $@

alloc.o : m_alloc.cu
	$(CUDA) -arch=$(ARCH) -c $< -o $@

heap.o : heap.cu
	$(CUDA) -arch=$(ARCH) -dc $< -o $@

clean :
	rm -rf *.o *.out

#iterator.o : iterator.cu
#	$(CUDA) -arch=$(ARCH) -dc $< -o $@

#operator.o : operator.cu
#	$(CUDA) -arch=$(ARCH) -dc $< -o $@

launch.o : launch.cu utils.h macros.h
	$(CUDA) -arch=$(ARCH) -c $< -o $@

kernel.o : kernel.cu operator.h iterator.h utils.h macros.h tile_helper.h
	$(CUDA) -arch=$(ARCH) -dc $< -o $@