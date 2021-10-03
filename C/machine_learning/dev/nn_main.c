//INLCUDES//

#include <stdio.h>

#include "nn_utils.h"

//INIT//

int main(int argc, char **argv){
	if (argc == 4){
		Network network;
		Network *N = &network;

		load_network(argv[1], argv[2], N, 6);

		activate_network(N);

		calculate_loss(N);

		display_network(N);

		free_memory_network(N, 6);

		return 0;
	}
	else {
                printf("Error: Invalid amount of arguments %d\n", argc);
                printf("Usage: ./nn.out data_file input_file output_file\n");
                return 1;
	}
}
