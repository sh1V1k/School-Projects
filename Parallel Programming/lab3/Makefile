OBJ=lab3

all:
# nvcc -std=c++11 -I../libwb -rdc=true -c -arch=sm_70 lab3.cu -o lab3.o
	nvcc -std=c++11 -I../libwb -rdc=true -c $(OBJ).cu -o $(OBJ).o 
	nvcc -std=c++11 -o $(OBJ) $(OBJ).o ../libwb/lib/libwb.so

clean:
	rm -f $(OBJ).o

cleanall: clean
	rm -f $(OBJ) *.out *.err

run:
	sbatch job.slurm

