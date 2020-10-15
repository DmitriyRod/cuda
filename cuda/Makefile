#Makefile 
#define variables
objects= main.o kernels.o 
NVCC= nvcc               #cuda c compiler
opt= -O2 -g -G -ccbin g++-5
ARCH= -arch=sm_21        #cuda compute capability
LIBS=  
execname= main


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu
main.o: main.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cu


#clean Makefile
clean:
	rm $(objects)
#end of Makefile

