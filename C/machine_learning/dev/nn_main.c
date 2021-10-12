//INLCUDES//

#include <stdio.h>

#include "nn_utils.h"

//INIT//

int main(int argc, char **argv){
	if (argc == 4){
/*
		Network *N;

		N = load_network(argv[1], argv[2], 6);

		activate_network(N);

		calculate_loss(N);

		calculate_accuracy(N);

		display_network(N);

		backprop_network(N);

		stochastic_gradient_descent(N, 1.0);

		activate_network(N);

		calculate_loss(N);

		calculate_accuracy(N);

		display_network(N);

                backprop_network(N);

                stochastic_gradient_descent(N, 1.0);

                activate_network(N);

                calculate_loss(N);

                calculate_accuracy(N);

                display_network(N);

		for (int i = 0; i < 1; i++)
		{
			backprop_network(N);

			stochastic_gradient_descent(N, 1.0);

			activate_network(N);

			calculate_loss(N);

			calculate_accuracy(N);
		}

		display_network(N);
*/

		Network *N;
		int network[7] = {4, 16, 32, 32, 16, 8, 4};

		N = create_network(argv[2], 7, network);

		train_network(N, argv[2], 100000);
/*
		activate_network(N);

		calculate_loss(N);

		calculate_accuracy(N);

		display_network(N);

		for (int i = 0; i < 100000; i++)
		{
			backprop_network(N);

			stochastic_gradient_descent(N, 0.005);

			activate_network(N);

			calculate_loss(N);

			calculate_accuracy(N);
		}

		display_network(N);

//		save_network(argv[3], N);
*/
		free_memory_network(N, 7);

		return 0;
	}
	else {
                printf("Error: Invalid amount of arguments %d\n", argc);
                printf("Usage: ./nn.out data_file input_file output_file\n");
                return 1;
	}
}
