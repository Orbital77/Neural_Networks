nn: nn_main.o nn_utils.o
	gcc nn_main.o nn_utils.o -o nn -lm

nn_main.o: nn_main.c
	gcc -c nn_main.c

nn_utils.o: nn_utils.c nn_utils.h
	gcc -c nn_utils.c

