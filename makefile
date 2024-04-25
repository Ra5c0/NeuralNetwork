neural: neural.o
	gcc -o neural neural.o

neural.o: neuralNetwork.c
	gcc -o neural.o -c neuralNetwork.c
