//INCLUDES//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nn_utils.h"

//DEFINES//

#define Bound_1 128
#define Bound_2 2
#define Bound_3 255
#define Bound_4 25

#define Alpha 0.001
#define Delta 0.000001

//ARRAY//

void clip(Network *N, double min, double max)
{
	for (int row = 0; row < N->layer[N->layers - 2]->so[0]; row++)
	{
		for (int col = 0; col < N->layer[N->layers - 2]->so[1]; col++)
		{
			if (N->layer[N->layers - 2]->output[row][col] < min)
			{
				N->layer[N->layers - 2]->output[row][col] = min;
			}
			else if (N->layer[N->layers - 2]->output[row][col] > max)
			{
				N->layer[N->layers - 2]->output[row][col] = max;
			}
		}
	}
}

void diagonal(double ***input_array, int shape[], double ***result)
{
	for (int row = 0; row < shape[0]; row++)
	{
		for (int col = 0; col < shape[1]; col++)
		{
			(*result)[row][col] = 0;
		}
	}

	if (shape[0] == 1)
	{
		for (int col = 0; col < shape[1] - 1; col++)
		{
			(*result)[col][col] = (*input_array)[0][col];
		}
	}
	else if (shape[0] > 1)
	{
		for (int col = 0; col < shape[1] - 1; col++)
		{
			printf("%f\n", (*input_array)[col][col]);
			(*result)[col][col] = (*input_array)[col][col];
		}
	}
}

int get_one_hot_index(int **input, int cols)
{
	int temp = 0;

	int temp_2 = 0;

	for (int i = 0; i < cols; i++)
	{
		if ((*input)[i] > temp_2)
		{
			temp = i;
			temp_2 = (*input)[i];
		}
	}

	return temp;
}

void get_max_values(Network *N)
{
	int temp;

	double temp_2;

	for (int row = 0; row < N->layer[N->layers - 2]->so[0]; row++)
	{
		temp = 0;

		temp_2 = 0;

		for (int col = 0; col < N->layer[N->layers - 2]->so[1]; col++)
		{
			if (N->layer[N->layers - 2]->output[row][col] > temp_2)
			{
				temp = col;
				temp_2 = N->layer[N->layers - 2]->output[row][col];
			}
		}

		N->output[row] = temp;
	}
}

//MATH//

double f(double x, char *function)
{
	if(strcmp(function, "relu") == 0)
	{
		if (x < 0) return 0;
		else return x;
	}
	else if (strcmp(function, "sigmoid") == 0)
	{
		return 1 / (1 + exp(-x));
	}
	else
	{
		printf("Invalid function input in function 'f'\n");
	}
}

double approximate_derivative(double x, char *function)
{
        double dx = (f(x + Delta, function) - (f(x - Delta, function))) / (2*Delta);

        return dx;
}

double d_rand(double min, double max)
{
	double range = (max - min);
	double div = RAND_MAX / range;
	return min + (rand() / div);
}

double d_softmax(double x, double y)
{
	if (y != x) y = 0;
	return x - y;
}

//BIAS//

void bias_1(double ***input, double **bias, int n_neurons)
{
	for (int i = 0; i < n_neurons; i++)
	{
		(*input)[0][i] += (*bias)[i];
	}
}

void bias_2(double ***input, double **bias, int shape_input[], int n_neurons)
{
	for (int row = 0; row < shape_input[0]; row++)
	{
		for (int col = 0; col < n_neurons; col++)
		{
			(*input)[row][col] += (*bias)[col];
		}
	}
}

void add_bias(double ***input, double **bias, int shape_input[], int n_neurons)
{
	if (shape_input[0] == 1)
	{
		bias_1(input, bias, n_neurons);
	}
	else
	{
		bias_2(input, bias, shape_input, n_neurons);
	}
}

//TRANSPOSE//

void transpose_input(N_Layer *layer)
{
	double **temp = malloc(sizeof(double*) * layer->si[1]);

	for (int i = 0; i < layer->si[1]; i++)
	{
		temp[i] = malloc(sizeof(double) * layer->si[0]);
	}

	for (int i = 0; i < layer->si[0]; i++)
	{
		for (int j = 0; j < layer->si[1]; j++)
		{
			temp[j][i] = layer->input[i][j];
		}
		free(layer->input[i]);
		layer->input[i] = NULL;
	}

	free(layer->input);
	layer->input = NULL;

	allocate_memory_input(&layer->input, layer->si[1], layer->si[0]);

	for (int i = 0; i < layer->si[0]; i++)
	{
		for (int j = 0; j < layer->si[1]; j++)
		{
			layer->input[j][i] = temp[j][i];
		}
	}

	for (int i = 0; i < layer->si[1]; i++)
	{
		free(temp[i]);
		temp[i] = NULL;
	}

	free(temp);
	temp = NULL;

	int temp_2 = 0, temp_3 = 0;

	temp_2 = layer->si[0];
	temp_3 = layer->si[1];

	layer->si[0] = temp_3;
	layer->si[1] = temp_2;
}

void transpose_input_first(V_Layer *layer)
{
	double **temp = malloc(sizeof(double*) * layer->si[1]);

	for (int i = 0; i < layer->si[1]; i++)
	{
		temp[i] = malloc(sizeof(double) * layer->si[0]);
	}

	for (int i = 0; i < layer->si[0]; i++)
	{
		for (int j = 0; j < layer->si[1]; j++)
		{
			temp[j][i] = layer->input[i][j];
		}
		free(layer->input[i]);
		layer->input[i] = NULL;
	}

	free(layer->input);
	layer->input = NULL;

	allocate_memory_input(&layer->input, layer->si[1], layer->si[0]);

	for (int i = 0; i < layer->si[0]; i++)
	{
		for (int j = 0; j < layer->si[1]; j++)
		{
			layer->input[j][i] = temp[j][i];
		}
	}

	for (int i = 0; i < layer->si[1]; i++)
	{
		free(temp[i]);
		temp[i] = NULL;
	}

	free(temp);
	temp = NULL;

	int temp_2 = 0, temp_3 = 0;

	temp_2 = layer->si[0];
	temp_3 = layer->si[1];

	layer->si[0] = temp_3;
	layer->si[1] = temp_2;
}

void transpose_weights(N_Layer *layer)
{
	double **temp = malloc(sizeof(double*) * layer->sw[1]);

        for (int i = 0; i < layer->sw[1]; i++)
        {
                temp[i] = malloc(sizeof(double) * layer->sw[0]);
        }

	for (int i = 0; i < layer->sw[0]; i++)
	{
		for (int j = 0; j < layer->sw[1]; j++)
		{
			temp[j][i] = layer->weights[i][j];
		}
	}

	for (int i = 0; i < layer->sw[0]; i++)
	{
		free(layer->weights[i]);
		layer->weights[i] = NULL;
	}

	free(layer->weights);
	layer->weights = NULL;

	allocate_memory_weights(&layer->weights, layer->sw[1], layer->sw[0]);

	for (int i = 0; i < layer->sw[0]; i++)
	{
		for (int j = 0; j < layer->sw[1]; j++)
		{
			layer->weights[j][i] = temp[j][i];
		}
	}

	for (int i = 0; i < layer->sw[1]; i++)
	{
		free(temp[i]);
		temp[i] = NULL;
	}

	free(temp);
	temp = NULL;

	int temp_2 = 0, temp_3 = 0;

	temp_2 = layer->sw[0];
	temp_3 = layer->sw[1];

	layer->sw[0] = temp_3;
	layer->sw[1] = temp_2;
}

void transpose_weights_first(V_Layer *layer)
{
	double **temp = malloc(sizeof(double*) * layer->sw[1]);

        for (int i = 0; i < layer->sw[1]; i++)
        {
                temp[i] = malloc(sizeof(double) * layer->sw[0]);
        }

	for (int i = 0; i < layer->sw[0]; i++)
	{
		for (int j = 0; j < layer->sw[1]; j++)
		{
			temp[j][i] = layer->weights[i][j];
		}
	}

	for (int i = 0; i < layer->sw[0]; i++)
	{
		free(layer->weights[i]);
		layer->weights[i] = NULL;
	}

	free(layer->weights);
	layer->weights = NULL;

	allocate_memory_weights(&layer->weights, layer->sw[1], layer->sw[0]);

	for (int i = 0; i < layer->sw[0]; i++)
	{
		for (int j = 0; j < layer->sw[1]; j++)
		{
			layer->weights[j][i] = temp[j][i];
		}
	}

	for (int i = 0; i < layer->sw[1]; i++)
	{
		free(temp[i]);
		temp[i] = NULL;
	}

	free(temp);
	temp = NULL;

	int temp_2 = 0, temp_3 = 0;

	temp_2 = layer->sw[0];
	temp_3 = layer->sw[1];

	layer->sw[0] = temp_3;
	layer->sw[1] = temp_2;
}

//DOT PRODUCT//

void dot_product_1_1(double ***a, double ***b, double ***r, int shape)
{
	for (int col = 0; col < shape; col++)
	{
		(*r)[0][0] += (*a)[0][col] * (*b)[0][col];
	}
}

void dot_product_1_2(double ***a, double ***b, double ***r, int shape_b[])
{
	for (int row = 0; row < shape_b[0]; row++)
	{
		for (int col = 0; col < shape_b[1]; col++)
		{
			(*r)[0][row] += (*a)[0][col] * (*b)[row][col];
		}
	}
}

void dot_product_2_2(double ***a, double ***b, double ***r, int shape_a[], int shape_b[])
{
	if (shape_a[1] != shape_b[0])
	{
		printf("Warning: shape error in function 'dot_product_2_2'\n");
	}
	else
	{
		for (int row = 0; row < shape_a[0]; row++)
		{
			for (int col = 0; col < shape_b[1]; col++)
			{
				(*r)[row][col] = 0;

				for (int iteration = 0; iteration < shape_a[1]; iteration++)
				{
					(*r)[row][col] += (*a)[row][iteration] * (*b)[iteration][col];
				}
			}
		}
	}
}

void dot(N_Layer *layer, double ***a, double ***b, double ***r, int shape_a[], int shape_b[])
{
	int choice;

	if (shape_a[1] != shape_b[0]) printf("Shape error in function 'dot': dim a 1 != dim b 0");

	if (shape_a[0] == 1 && shape_b[0] == 1) choice = 1;
	if (shape_a[0] == 1 && shape_b[0] > 1) choice = 2;
	if (shape_a[0] > 1 && shape_b[0] > 1) choice = 3;

	switch(choice)
	{
		case 1:
			dot_product_1_1(a, b, r, shape_b[1]);
			break;
		case 2:
			dot_product_1_2(a, b, r, shape_b);
			break;
		case 3:
			dot_product_2_2(a, b, r, shape_a, shape_b);
			break;
		default:
			printf("Invalid input in function 'dot'\n");
			break;
	}
}

void dot_first(V_Layer *layer, double ***a, double ***b, double ***r, int shape_a[], int shape_b[])
{
        int choice;

        if (shape_a[0] == 1 && shape_b[0] == 1) choice = 1;
        if (shape_a[0] == 1 && shape_b[0] > 1) choice = 2;
        if (shape_a[0] > 1 && shape_b[0] > 1) choice = 3;

        switch(choice)
        {
                case 1:
                        dot_product_1_1(a, b, r, shape_b[1]);
                        break;
                case 2:
                        dot_product_1_2(a, b, r, shape_b);
                        break;
                case 3:
                        dot_product_2_2(a, b, r, shape_a, shape_b);
                        break;
                default:
                        printf("Invalid input in function 'dot'\n");
                        break;
        }
}

//DISPLAY//

void display_1_1(double ***result)
{
	printf("%f\n", (*result)[0][0]);
}

void display_1_2(double ***result, int cols)
{
	for (int i = 0; i < cols; i++)
	{
		printf("%f\t", (*result)[0][i]);
	}
	printf("\n");
}

void display_2_2(double ***result, int shape[])
{
	for (int row = 0; row < shape[0]; row++)
	{
		for (int col = 0; col < shape[1]; col++)
		{
			printf("%f\t", (*result)[row][col]);
		}
		printf("\n\n");
	}
}

void display_visual_layer(V_Layer *layer)
{
	printf("DEPTH: %d\n\n", layer->depth);
	printf("NEURONS: %d\n\n", layer->neurons);
	printf("BATCH_SIZE: %d\n\n", layer->batch_size);
	printf("INPUT:\n");

	for (int row = 0; row < layer->si[0]; row++)
	{
		for (int col = 0; col < layer->si[1]; col++)
		{
			printf("%f ", layer->input[row][col]);
		}
		printf("\n");
	}

	printf("\nOUTPUT:\n");

	for (int row = 0; row < layer->so[0]; row++)
	{
		for (int col = 0; col < layer->so[1]; col++)
		{
			printf("%f ", layer->output[row][col]);
		}
		printf("\n");
	}

	printf("\nWEIGHTS:\n");

	for (int row = 0; row < layer->sw[0]; row++)
	{
		for (int col = 0; col < layer->sw[1]; col++)
		{
			printf("%f ", layer->weights[row][col]);
		}
		printf("\n");
	}

	printf("\nBIASES:\n");

	for (int col = 0; col < layer->neurons; col++)
	{
		printf("%f ", layer->biases[col]);
	}

	printf("\n\nSI: %d %d\n\n", layer->si[0], layer->si[1]);
	printf("SO: %d %d\n\n", layer->so[0], layer->so[1]);
	printf("SW: %d %d\n", layer->sw[0], layer->sw[1]);
}

void display_layer(N_Layer *layer)
{
        printf("DEPTH: %d\n\n", layer->depth);
        printf("NEURONS: %d\n\n", layer->neurons);
        printf("P_NEURONS: %d\n\n", layer->p_neurons);
        printf("INPUT:\n");

        for (int row = 0; row < layer->si[0]; row++)
        {
                for (int col = 0; col < layer->si[1]; col++)
                {
                        printf("%f ", layer->input[row][col]);
                }
                printf("\n");
        }

        printf("\nOUTPUT:\n");

        for (int row = 0; row < layer->so[0]; row++)
        {
                for (int col = 0; col < layer->so[1]; col++)
                {
                        printf("%f ", layer->output[row][col]);
                }
                printf("\n");
        }

        printf("\nWEIGHTS:\n");

        for (int row = 0; row < layer->sw[0]; row++)
        {
                for (int col = 0; col < layer->sw[1]; col++)
                {
                        printf("%f ", layer->weights[row][col]);
                }
                printf("\n");
        }

        printf("\nBIASES:\n");

        for (int col = 0; col < layer->neurons; col++)
        {
                printf("%f ", layer->biases[col]);
        }

        printf("\n\nSI: %d %d\n\n", layer->si[0], layer->si[1]);
        printf("SO: %d %d\n\n", layer->so[0], layer->so[1]);
        printf("SW: %d %d\n", layer->sw[0], layer->sw[1]);
}

void display_network(Network *network)
{
	printf("LAYERS: %d\n\n", network->layers);
        printf("DEPTH: %d\n\n", network->visual->depth);
        printf("NEURONS: %d\n\n", network->visual->neurons);
        printf("BATCH_SIZE: %d\n\n", network->visual->batch_size);
        printf("INPUT:\n");

        for (int row = 0; row < network->visual->si[0]; row++)
        {
                for (int col = 0; col < network->visual->si[1]; col++)
                {
                        printf("%f ", network->visual->input[row][col]);
                }
                printf("\n");
        }

        printf("\nOUTPUT:\n");

        for (int row = 0; row < network->visual->so[0]; row++)
        {
                for (int col = 0; col < network->visual->so[1]; col++)
                {
                        printf("%f ", network->visual->output[row][col]);
                }
                printf("\n");
        }

        printf("\nWEIGHTS:\n");

        for (int row = 0; row < network->visual->sw[0]; row++)
        {
                for (int col = 0; col < network->visual->sw[1]; col++)
                {
                        printf("%f ", network->visual->weights[row][col]);
                }
                printf("\n");
        }

        printf("\nBIASES:\n");

        for (int col = 0; col < network->visual->neurons; col++)
        {
                printf("%f ", network->visual->biases[col]);
        }

	printf("\n\nMETHOD: %s", network->visual->method);

        printf("\n\nSI: %d %d\n\n", network->visual->si[0], network->visual->si[1]);
        printf("SO: %d %d\n\n", network->visual->so[0], network->visual->so[1]);
        printf("SW: %d %d\n", network->visual->sw[0], network->visual->sw[1]);

	for (int layer = 0; layer < (network->layers - 1); layer++)
	{
		printf("\nDEPTH: %d\n\n", network->layer[layer]->depth);
		printf("NEURONS: %d\n\n", network->layer[layer]->neurons);
		printf("P_NEURONS: %d\n\n", network->layer[layer]->p_neurons);
		printf("INPUT:\n");

		for (int row = 0; row < network->layer[layer]->si[0]; row++)
		{
			for (int col = 0; col < network->layer[layer]->si[1]; col++)
			{
				printf("%f ", network->layer[layer]->input[row][col]);
			}
			printf("\n");
		}

		printf("\nOUTPUT:\n");

		for (int row = 0; row < network->layer[layer]->so[0]; row++)
		{
			for (int col = 0; col < network->layer[layer]->so[1]; col++)
			{
				printf("%f ", network->layer[layer]->output[row][col]);
			}
			printf("\n");
		}

		printf("\nWEIGHTS:\n");

		for (int row = 0; row < network->layer[layer]->sw[0]; row++)
		{
			for (int col = 0; col < network->layer[layer]->sw[1]; col++)
			{
				printf("%f ", network->layer[layer]->weights[row][col]);
			}
			printf("\n");
		}

		printf("\nBIASES:\n");

		for (int col = 0; col < network->layer[layer]->neurons; col++)
		{
			printf("%f ", network->layer[layer]->biases[col]);
		}

		printf("\n\nMETHOD: %s", network->layer[layer]->method);

		printf("\n\nSI: %d %d\n\n", network->layer[layer]->si[0], network->layer[layer]->si[1]);
		printf("SO: %d %d\n\n", network->layer[layer]->so[0], network->layer[layer]->so[1]);
		printf("SW: %d %d\n", network->layer[layer]->sw[0], network->layer[layer]->sw[1]);
	}

	printf("\nNETWORK LOSS:\n");

	printf("%f\n", network->loss);

	printf("\nNETWORK ACCURACY\n");

	printf("%f\n", network->accuracy);
}

//MEMORY//

void allocate_memory_biases(double **biases, int cols)
{
	*biases = calloc(cols, sizeof(double));
}

void allocate_memory_dactivation(double ***dactivation, int rows, int cols)
{
	*dactivation = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*dactivation)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_delta(double ***delta, int rows, int cols)
{
	*delta = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*delta)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_dbiases(double **dbiases, int cols)
{
	*dbiases = calloc(cols, sizeof(double));
}

void allocate_memory_dinput(double ***dinput, int rows, int cols)
{
	*dinput = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*dinput)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_dweights(double ***dweights, int rows, int cols)
{
	*dweights = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*dweights)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_errors(double ***errors, int rows, int cols)
{
	*errors = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*errors)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_expected(double **expected, int cols)
{
	*expected = calloc(cols, sizeof(double));
}

void allocate_memory_hot(int ***hot, int rows, int cols)
{
	*hot = malloc(sizeof(int*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*hot)[i] = calloc(cols, sizeof(int));
	}
}

void allocate_memory_input(double ***input, int rows, int cols)
{
	*input = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*input)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_network(Network *network, int n_layers, int iteration)
{
	if (iteration == 0)
	{
		network->visual = malloc(sizeof(V_Layer));
		network->layer = malloc(sizeof(N_Layer*) * (n_layers - 1));
		network->target = malloc(sizeof(One_Hot));
	}
	else if (iteration > 0)
	{
		network->layer[iteration - 1] = malloc(sizeof(N_Layer));
	}
}

void allocate_memory_network_output(Network *network, int cols)
{
	network->output = malloc(sizeof(double) * cols);
}

void allocate_memory_op(double ***op, int rows, int cols)
{
	*op = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*op)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_output(double ***output, int rows, int cols)
{
	*output = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*output)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_weights(double ***weights, int rows, int cols)
{
	*weights = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*weights)[i] = calloc(cols, sizeof(double));
	}
}

void allocate_memory_layer_set(N_Layer ***layer, int n_layers)
{
	*layer = malloc(sizeof(N_Layer*) * (n_layers - 1));
	for (int i = 0; i < (n_layers - 1); i++)
	{
		(*layer)[i] = malloc(sizeof(N_Layer));
	}
}

void free_memory_biases(double **biases)
{
	free(*biases);
	*biases = NULL;
}

void free_memory_hot(int ***hot, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		free((*hot)[i]);
		(*hot)[i] = NULL;
	}
	free(*hot);
	*hot = NULL;
}

void free_memory_input(double ***input, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		free((*input)[i]);
		(*input)[i] = NULL;
	}
	free(*input);
	*input = NULL;
}

void free_memory_network(Network *network, int n_layers)
{
	for (int i = 0; i < network->visual->si[0]; i++)
	{
		free(network->visual->input[i]);
		network->visual->input[i] = NULL;
	}

	free(network->visual->input);
	network->visual->input = NULL;

	for (int i = 0; i < network->visual->so[0]; i++)
	{
		free(network->visual->output[i]);
		network->visual->output[i] = NULL;

		free(network->visual->dactivation[i]);
		network->visual->dactivation[i] = NULL;
	}

	free(network->visual->dactivation);
	network->visual->dactivation = NULL;

	free(network->visual->output);
	network->visual->output = NULL;

	for (int i = 0; i < network->visual->sw[0]; i++)
	{
		free(network->visual->weights[i]);
		network->visual->weights[i] = NULL;
	}

	free(network->visual->weights);
	network->visual->weights = NULL;

	for (int i = 0; i < network->visual->sdi[0]; i++)
	{
		free(network->visual->dinput[i]);
		network->visual->dinput[i] = NULL;
	}

	free(network->visual->dinput);
	network->visual->dinput = NULL;

	for (int i = 0; i < network->visual->sdw[0]; i++)
	{
		free(network->visual->dweights[i]);
		network->visual->dweights[i] = NULL;
	}

	free(network->visual->dweights);
	network->visual->dweights = NULL;

	free(network->visual->dbiases);
	network->visual->dbiases = NULL;

	free(network->visual->biases);
	network->visual->biases = NULL;

	if (n_layers > 1)
	{
		for (int layer = 0; layer < (n_layers - 1); layer++)
		{
			for (int i = 0; i < network->layer[layer]->si[0]; i++)
			{
				free(network->layer[layer]->input[i]);
				network->layer[layer]->input[i] = NULL;
			}

			free(network->layer[layer]->input);
			network->layer[layer]->input = NULL;

			for (int i = 0; i < network->layer[layer]->so[0]; i++)
			{
				free(network->layer[layer]->output[i]);
				network->layer[layer]->output[i] = NULL;

				free(network->layer[layer]->dactivation[i]);
				network->layer[layer]->dactivation[i] = NULL;
			}

			free(network->layer[layer]->dactivation);
			network->layer[layer]->dactivation = NULL;

			free(network->layer[layer]->output);
			network->layer[layer]->output = NULL;

			for (int i = 0; i < network->layer[layer]->sw[0]; i++)
			{
				free(network->layer[layer]->weights[i]);
				network->layer[layer]->weights[i] = NULL;
			}

			free(network->layer[layer]->weights);
			network->layer[layer]->weights = NULL;

			for (int i = 0; i < network->layer[layer]->sdi[0]; i++)
			{
				free(network->layer[layer]->dinput[i]);
				network->layer[layer]->dinput[i] = NULL;
			}

			free(network->layer[layer]->dinput);
			network->layer[layer]->dinput = NULL;

			for (int i = 0; i < network->layer[layer]->sdw[0]; i++)
			{
				free(network->layer[layer]->dweights[i]);
				network->layer[layer]->dweights[i] = NULL;
			}

			free(network->layer[layer]->dweights);
			network->layer[layer]->dweights = NULL;

			free(network->layer[layer]->biases);
			network->layer[layer]->biases = NULL;

			free(network->layer[layer]->dbiases);
			network->layer[layer]->dbiases = NULL;

			free(network->layer[layer]);
			network->layer[layer] = NULL;
		}
	}
	free(network->layer);
	network->layer = NULL;
	free(network->visual);
	network->visual = NULL;

	for (int i = 0; i < network->target->sh[0]; i++)
	{
		free(network->target->one_hot[i]);
		network->target->one_hot[i];
	}

	free(network->target->one_hot);
	network->target->one_hot = NULL;

	free(network->target);
	network->target = NULL;

	if (network != NULL)
	{
		free(network);
		network = NULL;
	}
}

void free_memory_output(double ***output, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		free((*output)[i]);
		(*output)[i] = NULL;
	}
	free(*output);
	*output = NULL;
}

void free_memory_weights(double ***weights, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		free((*weights)[i]);
		(*weights)[i] = NULL;
	}
	free(*weights);
	*weights = NULL;
}

void free_memory_layer_set(N_Layer ***layer, int n_layers)
{
	for (int i = 0; i < (n_layers - 1); i++)
	{
		free((*layer)[i]);
		(*layer)[i] = NULL;
	}
	free(*layer);
	*layer = NULL;
}

void allocate_memory(double ***input, double ***output, double ***weights, double **biases, int input_rows, int input_cols, int weights_rows, int weights_cols, int n_neurons)
{
	allocate_memory_biases(biases, n_neurons);
	allocate_memory_input(input, input_rows, input_cols);
	allocate_memory_output(output, n_neurons, n_neurons);
	allocate_memory_weights(weights, weights_rows, weights_cols);
}

void free_memory(double ***input, double ***output, double ***weights, double **biases, int input_rows, int weights_rows, int output_rows)
{
	free_memory_biases(biases);
	free_memory_input(input, input_rows);
	free_memory_output(output, output_rows);
	free_memory_weights(weights, weights_rows);
}

void free_memory_first_layer(V_Layer *layer)
{
	free_memory_biases(&layer->biases);
	free_memory_input(&layer->input, layer->si[0]);
	free_memory_output(&layer->output, layer->so[0]);
	free_memory_weights(&layer->weights, layer->sw[0]);
}

void free_memory_layer(N_Layer *layer)
{
	free_memory_biases(&layer->biases);
	free_memory_input(&layer->input, layer->si[0]);
	free_memory_output(&layer->output, layer->so[0]);
	free_memory_weights(&layer->weights, layer->sw[0]);
}

//FILE//

void file_read_first_layer(char *data_file_name, char *input_file_name, V_Layer *layer)
{
	FILE *fp;
	int layers;
	int depth = 0;
	int cols;
	int temp;

	fp = fopen(input_file_name, "r");

	fscanf(fp, "SETS: %d\n\n", &temp);
	fscanf(fp, "DATA SET: %d\n\n", &temp);
	fscanf(fp, "BATCH SIZE: %d\n\n", &layer->batch_size);
	fscanf(fp, "INPUTS: %d\n", &cols);

	allocate_memory_input(&layer->input, layer->batch_size, cols);

	for (int row = 0; row < layer->batch_size; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(fp, "%lf ", &layer->input[row][col]);
		}
		fscanf(fp, "\n");
	}

	fclose(fp);

	layer->si[0] = layer->batch_size;
	layer->si[1] = cols;

	fp = fopen(data_file_name, "r");

	rewind(fp);
	fscanf(fp, "LAYERS: %d\n\n", &layers);
	fscanf(fp, "DEPTH: %d\n\n", &depth);
	if (depth == 0)
	{
		fscanf(fp, "NEURONS: %d\n\n", &layer->neurons);
		fscanf(fp, "BIASES:\n");

		allocate_memory_biases(&layer->biases, layer->neurons);

		for (int col = 0; col < layer->neurons; col++)
		{
			fscanf(fp, "%lf ", &layer->biases[col]);
		}

		fscanf(fp, "\n\nWEIGHTS: %d %d\n\n", &layer->sw[0], &layer->sw[1]);

		allocate_memory_weights(&layer->weights, layer->sw[0], layer->sw[1]);

		for (int row = 0; row < layer->sw[0]; row++)
		{
			for (int col = 0; col < layer->sw[1]; col++)
			{
				fscanf(fp, "%lf ", &layer->weights[row][col]);
			}
			fscanf(fp, "\n");
		}

		fscanf(fp, "\nMETHOD: %s\n\n", layer->method);
	}
	else
	{
		printf("Error: Could not align function 'file_read_first_layer' with file %s", data_file_name);
	}

	fclose(fp);
}

void file_read_input(char *input_file_name, V_Layer *layer)
{
	FILE *fp;
	int cols;
	double temp;

	fp = fopen(input_file_name, "r");

	fscanf(fp, "SETS: %d\n\n", &temp);
	fscanf(fp, "DATA SET: %d\n\n", &temp);
	fscanf(fp, "BATCH SIZE: %d\n\n", &layer->batch_size);
	fscanf(fp, "INPUTS: %d\n", &cols);

	allocate_memory_input(&layer->input, layer->batch_size, cols);

	for (int row = 0; row < layer->batch_size; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(fp, "%lf ", &layer->input[row][col]);
		}
		fscanf(fp, "\n");
	}

	layer->si[0] = layer->batch_size;
	layer->si[1] = cols;

	fclose(fp);
}

void file_read_target(char *input_file_name, One_Hot *hot)
{
	FILE *fp;
	int cols;
	double temp;

	fp = fopen(input_file_name, "r");

	fscanf(fp, "SETS: %d\n\n", &temp);
	fscanf(fp, "DATA SET: %d\n\n", &temp);
	fscanf(fp, "BATCH SIZE: %d\n\n", &hot->batch_size);
	fscanf(fp, "INPUTS: %d\n", &cols);

	for (int row = 0; row < hot->batch_size; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(fp, "%lf ", &temp);
		}
		fscanf(fp, "\n");
	}

	fscanf(fp, "\nTARGET: %d %d %d\n", &hot->rows, &hot->classes, &hot->items);

	if (hot->items == 1)
	{
		allocate_memory_hot(&hot->one_hot, hot->items, hot->rows);

		for (int col = 0; col < hot->rows; col++)
		{
			fscanf(fp, "%d ", &hot->one_hot[0][col]);
		}

		fscanf(fp, "\n");

		hot->sh[0] = hot->items;
		hot->sh[1] = hot->rows;
	}
	else
	{
		allocate_memory_hot(&hot->one_hot, hot->rows, hot->classes);

		for (int row = 0; row < hot->rows; row++)
		{
			for (int col = 0; col < hot->classes; col++)
			{
				fscanf(fp, "%d ", &hot->one_hot[row][col]);
			}
			fscanf(fp, "\n");
		}

		hot->sh[0] = hot->rows;
		hot->sh[1] = hot->classes;
	}
}

void file_read_layer(char *data_file_name, N_Layer *layer)
{
	FILE *fp;
	int layers;
	int depth;
	int n_p_neurons = 0;
	double temp;
	int rows;
	int cols;
	char temp_char[Bound_4];

	fp = fopen(data_file_name, "r");

	if (fp != NULL)
	{
		rewind(fp);
		fscanf(fp, "LAYERS: %d\n\n", &layers);
		for (int i = 0; i < layers; i++)
		{
			fscanf(fp, "DEPTH: %d\n\n", &depth);
			if (depth == layer->depth)
			{
				fscanf(fp, "NEURONS: %d\n\n", &layer->neurons);
				fscanf(fp, "BIASES:\n");

				allocate_memory_biases(&layer->biases, layer->neurons);

				for (int col = 0; col < layer->neurons; col++)
				{
					fscanf(fp, "%lf ", &layer->biases[col]);
				}

				fscanf(fp, "\n\nWEIGHTS: %d %d\n", &layer->sw[0], &layer->sw[1]);

				allocate_memory_weights(&layer->weights, layer->sw[0], layer->sw[1]);

				for (int row = 0; row < layer->sw[0]; row++)
				{
					for (int col = 0; col < layer->sw[1]; col++)
					{
						fscanf(fp, "%lf ", &layer->weights[row][col]);
					}
					fscanf(fp, "\n");
				}

				fscanf(fp, "\nMETHOD: %s\n\n", layer->method);

				if (n_p_neurons != 0) layer->p_neurons = n_p_neurons;
			}
			else
			{
				fscanf(fp, "NEURONS: %d\n\n", &n_p_neurons);
				fscanf(fp, "BIASES:\n");
				for (int col = 0; col < n_p_neurons; col++)
				{
					fscanf(fp, "%lf ", &temp);
				}

				fscanf(fp, "\n\nWEIGHTS: %d %d\n", &rows, &cols);

				for (int row = 0; row < rows; row++)
				{
					for (int col = 0; col < cols; col++)
					{
						fscanf(fp, "%lf ", &temp);
					}
					fscanf(fp, "\n");
				}

				fscanf(fp, "\nMETHOD: %s\n\n", temp_char);
			}
		}

	}
	else
	{
		printf("Could not open file %s in function 'read_layer'", data_file_name);
	}

	fclose(fp);
}

void file_switch_target(char *input_file_name, int target_data_set, Network *N)
{
	FILE *fp;

	int temp;

	fp = fopen(input_file_name, "r");

	fscanf(fp, "SETS: %d\n\n", N->data_sets);

	for (int set = 0; set < N->data_sets; set++)
	{
		fscanf(fp, "DATA SET: %d\n\n", &temp);
		if (temp == target_data_set)
		{
			for (int i = 0; i < N->visual->si[0]; i++)
			{
				free(N->visual->input[i]);
				N->visual->input[i] = NULL;
			}

			free(N->visual->input);
			N->visual->input = NULL;

			fscanf(fp, "BATCH SIZE: %d\n\n", N->visual->batch_size);
			fscanf(fp, "INPUTS: %d\n", N->visual->si[1]);

			N->visual->si[0] = N->visual->batch_size;

			allocate_memory_input(&N->visual->input, N->visual->si[0], N->visual->si[1]);

			for (int i = 0; i < N->visual->si[0]; i++)
			{
				for (int j = 0; j < N->visual->si[1]; j++)
				{
					fscanf(fp, "%lf ", N->visual->input[i][j]);
				}
				fscanf(fp, "\n");
			}

			fscanf(fp, "\nTARGET: %d %d %d\n", N->target->rows, N->target->classes, N->target->items);

			for (int i = 0; i < N->target->sh[0]; i++)
			{
				free(N->target->one_hot[i]);
				N->target->one_hot[i];
			}

			free(N->target->one_hot);
			N->target->one_hot = NULL;

			if (N->target->items == 1)
			{
				allocate_memory_hot(&N->target->one_hot, N->target->items, N->target->rows);

				for (int col = 0; col < N->target->rows; col++)
				{
					fscanf(fp, "%d ", N->target->one_hot[0][col]);
				}

				fscanf(fp, "\n");

				N->target->sh[0] = N->target->items;
				N->target->sh[1] = N->target->rows;
			}
			else
			{
				allocate_memory_hot(&N->target->one_hot, N->target->rows, N->target->classes);

				for (int row = 0; row < N->target->rows; row++)
				{
					for (int col = 0; col < N->target->classes; col++)
					{
						fscanf(fp, "%d ", &N->target->one_hot[row][col]);
					}
					fscanf(fp, "\n");
				}

				N->target->sh[0] = N->target->rows;
				N->target->sh[1] = N->target->classes;
			}

		}
		else
		{

		}
	}

	fclose(fp);
	fp = NULL;
}

void file_write_data(char *filename, int n_layers)
{
	FILE *fp;

	fp = fopen(filename, "w");

	fprintf(fp, "LAYERS: %d\n\n", n_layers);

	if (fp != NULL)
	{
		fclose(fp);
		fp = NULL;
	}
}

void file_write_values(char *file_name, N_Layer *layer)
{
	FILE *fp;

	fp = fopen(file_name, "a");

	fprintf(fp, "DEPTH: %d\n\n", layer->depth);

	fprintf(fp, "NEURONS: %d\n\n", layer->neurons);

	fprintf(fp, "BIASES:\n");

	for (int col = 0; col < layer->neurons; col++)
	{
		fprintf(fp, "%f ", layer->biases[col]);
	}

	fprintf(fp, "\n\n");

	fprintf(fp, "WEIGHTS: %d %d\n", layer->sw[0], layer->sw[1]);

	for (int row = 0; row < layer->sw[0]; row++)
	{
		for (int col = 0; col < layer->sw[1]; col++)
		{
			fprintf(fp, "%f ", layer->weights[row][col]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\nMETHOD: %s\n\n", layer->method);

	fclose(fp);
}

void file_write_values_first(char *data_file_name, V_Layer *layer)
{
	FILE *fp;

	fp = fopen(data_file_name, "a");

	fprintf(fp, "DEPTH: %d\n\n", layer->depth);

	fprintf(fp, "NEURONS: %d\n\n", layer->neurons);

	fprintf(fp, "BIASES:\n");

	for (int col = 0; col < layer->neurons; col++)
	{
		fprintf(fp, "%f ", layer->biases[col]);
	}

	fprintf(fp, "\n\nWEIGHTS: %d %d\n", layer->sw[0], layer->sw[1]);

	for (int row = 0; row < layer->sw[0]; row++)
	{
		for (int col = 0; col < layer->sw[1]; col++)
		{
			fprintf(fp, "%f ", layer->weights[row][col]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\nMETHOD: %s\n\n", layer->method);

	fclose(fp);
}

void file_inject_input(char *input_file_name, int batch_size, int n_inputs, N_Layer *layer)
{
	FILE *fp;

	fp = fopen(input_file_name, "w");

	fprintf(fp, "BATCH SIZE: %d\n\n", batch_size);

	fprintf(fp, "INPUTS: %d\n", n_inputs);

	if (batch_size > 1)
	{
		for (int row = 0; row < batch_size; row++)
		{
			for (int col = 0; col < n_inputs; col++)
			{
				fprintf(fp, "%f ", layer->input[row][col]);
			}
			fprintf(fp, "\n");
		}
	}
	else
	{
		for (int col = 0; col < n_inputs; col++)
		{
			fprintf(fp, "%f ", layer->input[0][col]);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

//LOSS//

void calculate_loss(Network *N)
{
	double temp = 0;
	double count = 0;

	double temp_2[N->layer[N->layers - 2]->so[0]];

	if (N->target->sh[0] == 1)
	{
		clip(N, exp(-13), (1 - exp(-13)));

		for (int col = 0; col < N->layer[N->layers - 2]->so[0]; col++)
		{
			temp_2[col] = 0;
			temp_2[col] = -(log(N->layer[N->layers - 2]->output[col][N->target->one_hot[0][col]]));
		}

		for (int col = 0; col < N->layer[N->layers - 2]->so[0]; col++)
		{
			temp += temp_2[col];
			count += 1;
		}

		N->loss = temp / count;
	}
	else if (N->target->sh[0] > 1)
	{
		clip(N, exp(-13), (1 - exp(-13)));

		for (int row = 0; row < N->target->sh[0]; row++)
		{
			temp_2[row] = 0;
			temp = 0;

			for (int col = 0; col < N->target->sh[1]; col++)
			{
				temp += N->layer[N->layers - 2]->output[row][col] * N->target->one_hot[row][col];
			}
			temp_2[row] = -(log(temp));
		}

		temp = 0;

		for (int col = 0; col < N->layer[N->layers - 2]->so[0]; col++)
		{
			temp += temp_2[col];
			count += 1;
		}

		N->loss = temp / count;
	}
}

//ACCURACY//

void calculate_accuracy(Network *N)
{
	double temp[N->layer[N->layers - 2]->so[0]];

	double temp_2 = 0;

	double temp_3[N->layer[N->layers - 2]->so[0]];

	double temp_4[N->layer[N->layers - 2]->so[0]];

	int count;

	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		count = 0;
		temp_2 = 0;

		for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
		{
			if (N->layer[N->layers - 2]->output[i][j] > temp_2)
			{
				temp_2 = N->layer[N->layers - 2]->output[i][j];
				count = j;
			}
		}

		temp[i] = count;
	}

	if (N->target->sh[0] > 1)
	{
		for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
		{
			count = 0;
			temp_2 = 0;

			for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
			{
				if (N->target->one_hot[i][j] > temp_2)
				{
					temp_2 = N->target->one_hot[i][j];
					count = j;
				}
			}

			temp_3[i] = count;
		}
	}
	else
	{
		for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
		{
			temp_3[i] = N->target->one_hot[0][i];
		}
	}

	temp_2 = 0;

	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		temp_4[i] = 0;
		if (temp[i] == temp_3[i]) temp_4[i] = 1;
		temp_2 += temp_4[i];
	}

	N->accuracy = temp_2 / N->layer[N->layers - 2]->so[0];
}

//ACTIVATION FUNCTIONS//

void layer_activation_relu(N_Layer *layer)
{
	for (int row = 0; row < layer->so[0]; row++)
	{
		for (int col = 0; col < layer->so[1]; col++)
		{
			if (layer->output[row][col] < 0) layer->output[row][col] = 0;
		}
	}
}

void layer_activation_relu_first(V_Layer *layer)
{
        for (int row = 0; row < layer->so[0]; row++)
        {
                for (int col = 0; col < layer->so[1]; col++)
                {
                        if (layer->output[row][col] < 0) layer->output[row][col] = 0;
                }
        }
}

void layer_activation_sigmoid(N_Layer *layer)
{
	for (int row = 0; row < layer->so[0]; row++)
	{
		for (int col = 0; col < layer->so[1]; col++)
		{
			layer->output[row][col] = 1 / (1 + exp(-layer->output[row][col]));
		}
	}
}

void layer_activation_sigmoid_first(V_Layer *layer)
{
	for (int row = 0; row < layer->so[0]; row++)
	{
		for (int col = 0; col < layer->so[1]; col++)
		{
			layer->output[row][col] = 1 / (1 + exp(-layer->output[row][col]));
		}
	}
}

void layer_activation_softmax(N_Layer *layer)
{
	double exponents[layer->so[0]][layer->so[1]];
	double max[layer->so[0]];
	double sum[layer->so[0]];

	for (int i = 0; i < layer->so[0]; i++)
	{
		max[i] = 0;

		for (int j = 0; j < layer->so[1]; j++)
		{
			if (layer->output[i][j] > max[i])
			{
				max[i] = layer->output[i][j];
			}
		}
	}

	for (int i = 0; i < layer->so[0]; i++)
	{
		sum[i] = 0;

		for (int j = 0; j < layer->so[1]; j++)
		{
			exponents[i][j] = exp(layer->output[i][j] - max[i]);
			sum[i] += exponents[i][j];
		}
	}

	for (int i = 0; i < layer->so[0]; i++)
	{
		for (int j = 0; j < layer->so[1]; j++)
		{
			layer->output[i][j] = exponents[i][j] / sum[i];
		}
	}
}

void layer_activation_softmax_first(V_Layer *layer)
{
        double exponent[layer->so[0]][layer->so[1]];
        double sum[layer->so[0]];

	int temp;

	double temp_2;

	int temp_3[layer->so[0]];

	for (int row = 0; row < layer->so[0]; row++)
	{
		temp = 0;

		temp_2 = 0;

		temp_3[row] = 0;

		for (int col = 0; col < layer->so[1]; col++)
		{
			if (layer->output[row][col] > temp_2)
			{
				temp = col;
				temp_2 = layer->output[row][col];
			}
		}

		temp_3[row] = temp;
	}

        for (int row = 0; row < layer->so[0]; row++)
        {
                for (int col = 0; col < layer->so[1]; col++)
                {
                        exponent[row][col] = exp(layer->output[row][col]);
                        sum[row] += exponent[row][col];
                }
        }

        for (int row = 0; row < layer->so[0]; row++)
        {
                for (int col = 0; col < layer->so[1]; col++)
                {
                        layer->output[row][col] = exponent[row][col] / sum[row];
                }
        }
}

//NEURONS//

void get_first_output(N_Layer *layer, V_Layer *p_layer)
{
	for (int row = 0; row < p_layer->so[0]; row++)
	{
		for (int col = 0; col < p_layer->so[1]; col++)
		{
			layer->input[row][col] = p_layer->output[row][col];
		}
	}
}

void get_output(N_Layer *layer, N_Layer *p_layer)
{
	for (int row = 0; row < p_layer->so[0]; row++)
	{
		for (int col = 0; col < p_layer->so[1]; col++)
		{
			layer->input[row][col] = p_layer->output[row][col];
		}
	}
}

int get_max_layer(Network *N)
{
	int temp = N->visual->neurons;
	int temp_2 = N->visual->depth;

	for (int i = 0; i < N->layers - 1; i++)
	{
		if (N->layer[i]->neurons >= temp)
		{
			temp_2 = N->layer[i]->depth;
			temp = N->layer[i]->neurons;
		}
	}

	return temp_2;
}

int get_max_neurons(Network *N)
{
	int temp;

	temp = N->visual->neurons;

	for (int i = 0; i < N->layers - 1; i++)
	{
		if (N->layer[i]->neurons > temp) temp = N->layer[i]->neurons;
	}

	return temp;
}

void layer_activate(N_Layer *layer)
{
	if (strcmp(layer->method, "relu") == 0)
	{
		layer_activation_relu(layer);
	}
	else if (strcmp(layer->method, "softmax") == 0)
	{
		layer_activation_softmax(layer);
	}
	else if (strcmp(layer->method, "sigmoid") == 0)
	{
		layer_activation_sigmoid(layer);
	}
	else
	{
		printf("Error: Invalid input in function 'layer_activate'\n");
	}
}

void layer_activate_first(V_Layer *layer)
{
	if (strcmp(layer->method, "relu") == 0)
	{
		layer_activation_relu_first(layer);
	}
	else if (strcmp(layer->method, "softmax") == 0)
	{
		layer_activation_softmax_first(layer);
	}
	else if (strcmp(layer->method, "sigmoid") == 0)
	{
		layer_activation_sigmoid_first(layer);
	}
	else
	{
		printf("Error: Invalid input in function 'layer_activate_first'\n");
	}
}

void layer_forward(N_Layer *layer)
{
	if (layer->si[1] != layer->sw[0] && layer->si[0] > 1) transpose_weights(layer);
	dot(layer, &layer->input, &layer->weights, &layer->output, layer->si, layer->sw);
	add_bias(&layer->output, &layer->biases, layer->so, layer->neurons);
	if (layer->method != NULL && strcmp(layer->method, "none") != 0) layer_activate(layer);
}

void layer_forward_first(V_Layer *layer)
{
	if (layer->si[1] != layer->sw[0] && layer->si[0] > 1) transpose_weights_first(layer);
	dot_first(layer, &layer->input, &layer->weights, &layer->output, layer->si, layer->sw);
	add_bias(&layer->output, &layer->biases, layer->so, layer->neurons);
	if (layer->method != NULL && strcmp(layer->method, "none") != 0) layer_activate_first(layer);
}

void set_values(int n_neurons, int n_p_neurons, int depth, N_Layer *layer)
{
	layer->depth = depth;

	layer->si[0] = n_neurons;
	layer->si[1] = n_p_neurons;

	layer->sw[0] = n_p_neurons;
	layer->sw[1] = n_neurons;

	layer->so[0] = n_neurons;
	layer->so[1] = n_neurons;
}

void init_layer(char *file_name, N_Layer *layer, N_Layer *p_layer, int depth)
{
	layer->depth = depth;
	file_read_layer(file_name, layer);
	allocate_memory_input(&layer->input, p_layer->so[0], p_layer->so[1]);
	layer->si[0] = p_layer->so[0];
	layer->si[1] = p_layer->so[1];
	allocate_memory_output(&layer->output, layer->si[0], layer->neurons);
	layer->so[0] = layer->si[0];
	layer->so[1] = layer->neurons;
	allocate_memory_dbiases(&layer->dbiases, layer->neurons);
	allocate_memory_dinput(&layer->dinput, layer->so[0], layer->sw[0]);
	allocate_memory_dweights(&layer->dweights, layer->si[1], layer->neurons);
	layer->sdi[0] = layer->so[0];
	layer->sdi[1] = layer->sw[0];
	layer->sdw[0] = layer->si[1];
	layer->sdw[1] = layer->neurons;
	allocate_memory_dactivation(&layer->dactivation, layer->so[0], layer->so[1]);
}

void init_first_layer(char *data_file_name, char *input_file_name, V_Layer *layer)
{
	layer->depth = 0;
	file_read_first_layer(data_file_name, input_file_name, layer);
	layer->so[0] = layer->si[0];
	layer->so[1] = layer->neurons;
	allocate_memory_output(&layer->output, layer->si[0], layer->neurons);
	allocate_memory_dbiases(&layer->dbiases, layer->neurons);
	allocate_memory_dinput(&layer->dinput, layer->so[0], layer->sw[0]);
	allocate_memory_dweights(&layer->dweights, layer->si[1], layer->neurons);
        layer->sdi[0] = layer->so[0];
        layer->sdi[1] = layer->sw[0];
        layer->sdw[0] = layer->si[1];
        layer->sdw[1] = layer->neurons;
	allocate_memory_dactivation(&layer->dactivation, layer->so[0], layer->so[1]);
}

void init_second_layer(char *data_file_name, N_Layer *layer, V_Layer *p_layer)
{
	layer->depth = 1;
	file_read_layer(data_file_name, layer);
	allocate_memory_input(&layer->input, p_layer->so[0], p_layer->so[1]);
	layer->si[0] = p_layer->so[0];
	layer->si[1] = p_layer->so[1];
	allocate_memory_output(&layer->output, layer->si[0], layer->neurons);
	layer->so[0] = layer->si[0];
	layer->so[1] = layer->neurons;
	allocate_memory_dbiases(&layer->dbiases, layer->neurons);
	allocate_memory_dinput(&layer->dinput, layer->so[0], layer->sw[0]);
	allocate_memory_dweights(&layer->dweights, layer->si[1], layer->neurons);
        layer->sdi[0] = layer->so[0];
        layer->sdi[1] = layer->sw[0];
        layer->sdw[0] = layer->si[1];
        layer->sdw[1] = layer->neurons;
	allocate_memory_dactivation(&layer->dactivation, layer->so[0], layer->so[1]);
}

void inject(N_Layer *layer, double inputs[][Bound_1], double weights[][Bound_1], double biases[], int inputs_rows, int inputs_cols, int weights_rows, int weights_cols, int neurons, int p_neurons, int depth)
{
	layer->depth = depth;
	layer->neurons = neurons;
	layer->p_neurons = p_neurons;

	for (int col = 0; col < neurons; col++)
	{
		layer->biases[col] = biases[col];
	}

	for (int row = 0; row < weights_rows; row++)
	{
		for (int col = 0; col < weights_cols; col++)
		{
			layer->weights[row][col] = weights[row][col];
		}
	}

	for (int row = 0; row < inputs_rows; row++)
	{
		for (int col = 0; col < inputs_cols; col++)
		{
			layer->input[row][col] = inputs[row][col];
		}
	}

	layer->si[0] = inputs_rows;
	layer->si[1] = inputs_cols;

	layer->so[0] = neurons;
	layer->so[1] = neurons;

	layer->sw[0] = weights_rows;
	layer->sw[1] = weights_cols;
}

void inject_layer(N_Layer *layer, double ***inputs, double weights[][Bound_1], double biases[], int inputs_rows, int inputs_cols, int weights_rows, int weights_cols, int neurons, int p_neurons, int depth)
{
        layer->depth = depth;
        layer->neurons = neurons;
        layer->p_neurons = p_neurons;

        for (int col = 0; col < neurons; col++)
        {
                layer->biases[col] = biases[col];
        }

        for (int row = 0; row < weights_rows; row++)
        {
                for (int col = 0; col < weights_cols; col++)
                {
                        layer->weights[row][col] = weights[row][col];
                }
        }

        for (int row = 0; row < inputs_rows; row++)
        {
                for (int col = 0; col < inputs_cols; col++)
                {
                        layer->input[row][col] = *inputs[row][col];
                }
        }

        layer->si[0] = inputs_rows;
        layer->si[1] = inputs_cols;

        layer->so[0] = inputs_rows;
        layer->so[1] = neurons;

        layer->sw[0] = weights_rows;
        layer->sw[1] = weights_cols;
}

void inject_raw(N_Layer *layer, double inputs[][Bound_1], int inputs_rows, int inputs_cols, int depth)
{
	layer->depth = depth;

	for (int row = 0; row < inputs_rows; row++)
	{
		for (int col = 0; col < inputs_cols; col++)
		{
			layer->input[row][col] = inputs[row][col];
		}
	}
}

//DERIVATIVES//

void d_relu(N_Layer *layer, N_Layer *n_layer)
{
	for (int i = 0; i < layer->so[0]; i++)
	{
		for (int j = 0; j < layer->so[1]; j++)
		{
			layer->dactivation[i][j] = n_layer->dinput[i][j];
			if (layer->input[i][j] <= 0) layer->dactivation[i][j] = 0;
		}
	}
}

void d_relu_first(V_Layer *layer, N_Layer *n_layer)
{
	for (int i = 0; i < layer->so[0]; i++)
	{
		for (int j = 0; j < layer->so[1]; j++)
		{
			layer->dactivation[i][j] = n_layer->dinput[i][j];
			if (layer->input[i][j] <= 0) layer->dactivation[i][j] = 0;
		}
	}
}

void d_relu_last(Network *N)
{
	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
		{
			N->layer[N->layers - 2]->dactivation[i][j] = N->layer[N->layers - 2]->output[i][j];
			if (N->layer[N->layers - 2]->input[i][j] <= 0) N->layer[N->layers - 2]->dactivation[i][j];
		}
	}
}

void d_sigmoid(N_Layer *layer, N_Layer *n_layer)
{
	for (int i = 0; i < layer->so[0]; i++)
	{
		for (int j = 0; j < layer->so[1]; j++)
		{
			layer->dactivation[i][j] = n_layer->dinput[i][j] * (1 - n_layer->dinput[i][j]);
		}
	}
}

void d_sigmoid_first(V_Layer *layer, N_Layer *n_layer)
{
	for (int i = 0; i < layer->so[0]; i++)
	{
		for (int j = 0; j < layer->so[1]; j++)
		{
			layer->dactivation[i][j] = n_layer->dinput[i][j] * (1 - n_layer->dinput[i][j]);
		}
	}
}

void d_sigmoid_last(Network *N)
{
	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
		{
			N->layer[N->layers - 2]->dactivation[i][j] = N->layer[N->layers - 2]->output[i][j] * (1 - N->layer[N->layers - 2]->output[i][j]);
		}
	}
}

void d_softmax_loss_categorical_cross_entropy(Network *N)
{
        int target_index[N->layer[N->layers - 2]->so[0]];

	if (N->target->sh[0] > 1)
	{
		for (int i = 0; i < N->target->sh[0]; i++)
		{
			target_index[i] = get_one_hot_index(&N->target->one_hot[i], N->target->sh[1]);
		}
	}
	else
	{
		for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
		{
			target_index[i] = N->target->one_hot[0][i];
		}
	}

	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
		{
			N->layer[N->layers - 2]->dactivation[i][j] = N->layer[N->layers - 2]->output[i][j];
		}
		N->layer[N->layers - 2]->dactivation[i][target_index[i]] -= 1;
	}

	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
		{
			N->layer[N->layers - 2]->dactivation[i][j] = N->layer[N->layers - 2]->dactivation[i][j] / N->layer[N->layers - 2]->so[0];
		}
	}
}

//NETWORK//

void activate_network(Network *N)
{
	layer_forward_first(N->visual);

	for (int i = 0; i < (N->layers - 1); i++)
	{
		if (i == 0)
		{
			get_first_output(N->layer[i], N->visual);
			layer_forward(N->layer[i]);
		}
		else
		{
			get_output(N->layer[i], N->layer[i - 1]);
			layer_forward(N->layer[i]);
		}
	}
}

void backprop_network(Network *N)
{
	double temp;

	for (int layer = N->layers - 2; layer >= 0; layer--)
	{
		if (layer == N->layers - 2)
		{
			if (strcmp(N->layer[layer]->method, "softmax") == 0)
			{
				d_softmax_loss_categorical_cross_entropy(N);
			}
			else if (strcmp(N->layer[layer]->method, "sigmoid") == 0)
			{
				d_sigmoid_last(N);
			}
			else if (strcmp(N->layer[layer]->method, "relu") == 0)
			{
				d_relu_last(N);
			}

			transpose_input(N->layer[layer]);

			dot(N->layer[layer], &N->layer[layer]->input, &N->layer[layer]->dactivation, &N->layer[layer]->dweights, N->layer[layer]->si, N->layer[layer]->so);

			transpose_input(N->layer[layer]);

			transpose_weights(N->layer[layer]);

			dot(N->layer[layer], &N->layer[layer]->dactivation, &N->layer[layer]->weights, &N->layer[layer]->dinput, N->layer[layer]->so, N->layer[layer]->sw);

			transpose_weights(N->layer[layer]);

			for (int i = 0; i < N->layer[layer]->so[1]; i++)
			{
				temp = 0;

				for (int j = 0; j < N->layer[layer]->so[0]; j++)
				{
					temp += N->layer[layer]->dactivation[j][i];
				}

				N->layer[layer]->dbiases[i] = temp;
			}
		}
		else
		{
			if(strcmp(N->layer[layer]->method, "sigmoid") == 0)
			{
				d_sigmoid(N->layer[layer], N->layer[layer + 1]);
			}
			else if (strcmp(N->layer[layer]->method, "relu") == 0)
			{
				d_relu(N->layer[layer], N->layer[layer + 1]);
			}

			transpose_input(N->layer[layer]);

			dot(N->layer[layer], &N->layer[layer]->input, &N->layer[layer]->dactivation, &N->layer[layer]->dweights, N->layer[layer]->si, N->layer[layer]->so);

			transpose_input(N->layer[layer]);

			transpose_weights(N->layer[layer]);

			dot(N->layer[layer], &N->layer[layer]->dactivation, &N->layer[layer]->weights, &N->layer[layer]->dinput, N->layer[layer]->so, N->layer[layer]->sw);

			transpose_weights(N->layer[layer]);

			for (int i = 0; i < N->layer[layer]->so[1]; i++)
			{
				temp = 0;

				for (int j = 0; j < N->layer[layer]->so[0]; j++)
				{
					temp += N->layer[layer]->dactivation[j][i];
				}

				N->layer[layer]->dbiases[i] = temp;
			}
		}
	}

	if (strcmp(N->visual->method, "sigmoid") == 0)
	{
		d_sigmoid_first(N->visual, N->layer[0]);
	}
	else if (strcmp(N->visual->method, "relu") == 0)
	{
		d_relu_first(N->visual, N->layer[0]);
	}

	transpose_input_first(N->visual);

	dot_first(N->visual, &N->visual->input, &N->visual->dactivation, &N->visual->dweights, N->visual->si, N->visual->so);

	transpose_input_first(N->visual);

	transpose_weights_first(N->visual);

	dot_first(N->visual, &N->visual->dactivation, &N->visual->weights, &N->visual->dinput, N->visual->so, N->visual->sw);

	transpose_weights_first(N->visual);

	for (int i = 0; i < N->visual->so[1]; i++)
	{
		temp = 0;

		for (int j = 0; j < N->visual->so[0]; j++)
		{
			temp += N->visual->dactivation[j][i];
		}

		N->visual->dbiases[i] = temp;
	}
}

Network *create_network(char *input_file_name, int n_layers, int network_shape[])
{
	Network *N = malloc(sizeof(Network));

	srand(time(NULL));

	N->layers = n_layers;

	allocate_memory_network(N, n_layers, 0);
	file_read_input(input_file_name, N->visual);
	N->visual->depth = 0;
	N->visual->neurons = network_shape[0];

	N->visual->sw[0] = N->visual->si[1];
	N->visual->sw[1] = N->visual->neurons;

	allocate_memory_weights(&N->visual->weights, N->visual->sw[0], N->visual->sw[1]);

	for (int row = 0; row < N->visual->sw[0]; row++)
	{
		for (int col = 0; col < N->visual->sw[1]; col++)
		{
			N->visual->weights[row][col] = d_rand(-0.5, 0.5);
		}
	}

	allocate_memory_biases(&N->visual->biases, network_shape[0]);

	for (int col = 0; col < N->visual->neurons; col++)
	{
		N->visual->biases[col] = 0;
	}

	N->visual->so[0] = N->visual->si[0];
	N->visual->so[1] = N->visual->neurons;

	allocate_memory_output(&N->visual->output, N->visual->so[0], N->visual->so[1]);

	allocate_memory_dbiases(&N->visual->dbiases, N->visual->neurons);

	allocate_memory_dinput(&N->visual->dinput, N->visual->so[0],  N->visual->sw[0]);

	allocate_memory_dweights(&N->visual->dweights, N->visual->si[1], N->visual->neurons);

	N->visual->sdi[0] = N->visual->so[0];
	N->visual->sdi[1] = N->visual->sw[0];
	N->visual->sdw[0] = N->visual->si[1];
	N->visual->sdw[1] = N->visual->si[0];

	allocate_memory_dactivation(&N->visual->dactivation, N->visual->so[0], N->visual->so[1]);

	strcpy(N->visual->method, "relu");

	if (N->layers > 1)
	{
		allocate_memory_network(N, N->layers, 1);
		N->layer[0]->depth = 1;
		N->layer[0]->neurons = network_shape[1];
		N->layer[0]->p_neurons = N->visual->neurons;

		N->layer[0]->si[0] = N->visual->so[0];
		N->layer[0]->si[1] = N->visual->so[1];

		allocate_memory_input(&N->layer[0]->input, N->layer[0]->si[0], N->layer[0]->si[1]);

		N->layer[0]->sw[0] = N->layer[0]->si[1];
		N->layer[0]->sw[1] = N->layer[0]->neurons;

		allocate_memory_weights(&N->layer[0]->weights, N->layer[0]->sw[0], N->layer[0]->sw[1]);

		for (int row = 0; row < N->layer[0]->sw[0]; row++)
		{
			for (int col = 0; col < N->layer[0]->sw[1]; col++)
			{
				N->layer[0]->weights[row][col] = d_rand(-0.5, 0.5);
			}
		}

		allocate_memory_biases(&N->layer[0]->biases, N->layer[0]->neurons);

		for (int col = 0; col < N->layer[0]->neurons; col++)
		{
			N->layer[0]->biases[col] = 0;
		}

		N->layer[0]->so[0] = N->layer[0]->si[0];
		N->layer[0]->so[1] = N->layer[0]->neurons;

		allocate_memory_output(&N->layer[0]->output, N->layer[0]->so[0], N->layer[0]->so[1]);

		allocate_memory_dbiases(&N->layer[0]->dbiases, N->layer[0]->neurons);

		allocate_memory_dinput(&N->layer[0]->dinput, N->layer[0]->so[0], N->layer[0]->sw[0]);

		allocate_memory_dweights(&N->layer[0]->dweights, N->layer[0]->si[1], N->layer[0]->neurons);

		N->layer[0]->sdi[0] = N->layer[0]->so[0];
		N->layer[0]->sdi[1] = N->layer[0]->sw[0];
		N->layer[0]->sdw[0] = N->layer[0]->si[1];
		N->layer[0]->sdw[1] = N->layer[0]->si[0];

		allocate_memory_dactivation(&N->layer[0]->dactivation, N->layer[0]->so[0], N->layer[0]->so[1]);

		if (N->layer[0]->depth == n_layers - 1)
		{
			strcpy(N->layer[0]->method, "softmax");
		}
		else
		{
			strcpy(N->layer[0]->method, "relu");
		}

		if (N->layers > 2)
		{
			for (int layer = 1; layer < (n_layers - 1); layer++)
			{
				allocate_memory_network(N, N->layers, layer + 1);
				N->layer[layer]->depth = layer + 1;
				N->layer[layer]->neurons = network_shape[layer + 1];
				N->layer[layer]->p_neurons = N->layer[layer - 1]->neurons;

				N->layer[layer]->si[0] = N->layer[layer - 1]->so[0];
				N->layer[layer]->si[1] = N->layer[layer - 1]->so[1];

				allocate_memory_input(&N->layer[layer]->input, N->layer[layer]->si[0], N->layer[layer]->si[1]);

				N->layer[layer]->sw[0] = N->layer[layer]->si[1];
				N->layer[layer]->sw[1] = N->layer[layer]->neurons;

				allocate_memory_weights(&N->layer[layer]->weights, N->layer[layer]->sw[0], N->layer[layer]->sw[1]);

				for (int row = 0; row < N->layer[layer]->sw[0]; row++)
				{
					for (int col = 0; col < N->layer[layer]->sw[1]; col++)
					{
						N->layer[layer]->weights[row][col] = d_rand(-0.5, 0.5);
					}
				}

				allocate_memory_biases(&N->layer[layer]->biases, N->layer[layer]->neurons);

				for (int col = 0; col < N->layer[layer]->neurons; col++)
				{
					N->layer[layer]->biases[col] = 0;
				}

				N->layer[layer]->so[0] = N->layer[layer]->si[0];
				N->layer[layer]->so[1] = N->layer[layer]->neurons;

				allocate_memory_output(&N->layer[layer]->output, N->layer[layer]->so[0], N->layer[layer]->so[1]);

				allocate_memory_dbiases(&N->layer[layer]->dbiases, N->layer[layer]->neurons);

                		allocate_memory_dinput(&N->layer[layer]->dinput, N->layer[layer]->so[0], N->layer[layer]->sw[0]);

                		allocate_memory_dweights(&N->layer[layer]->dweights, N->layer[layer]->si[1], N->layer[layer]->neurons);

				N->layer[layer]->sdi[0] = N->layer[layer]->so[0];
				N->layer[layer]->sdi[1] = N->layer[layer]->sw[0];
				N->layer[layer]->sdw[0] = N->layer[layer]->si[1];
				N->layer[layer]->sdw[1] = N->layer[layer]->si[0];

				allocate_memory_dactivation(&N->layer[layer]->dactivation, N->layer[layer]->so[0], N->layer[layer]->so[1]);

                		if (N->layer[layer]->depth == n_layers - 1)
                		{
                        		strcpy(N->layer[layer]->method, "softmax");
                		}
                		else
                		{
                        		strcpy(N->layer[layer]->method, "relu");
                		}
			}
		}
	}

	file_read_target(input_file_name, N->target);

	return N;
}

Network *load_network(char *data_file_name, char *input_file_name, int n_layers)
{
	Network *network = malloc(sizeof(Network));

	network->layers = n_layers;
	allocate_memory_network(network, n_layers, 0);
	init_first_layer(data_file_name, input_file_name, network->visual);
	allocate_memory_network(network, n_layers, 1);
	init_second_layer(data_file_name, network->layer[0], network->visual);

	if (n_layers > 2)
	{
		for (int i = 1; i < (n_layers - 1); i++)
		{
			allocate_memory_network(network, n_layers, i + 1);
			init_layer(data_file_name, network->layer[i], network->layer[i-1], i + 1);
		}
	}

	file_read_target(input_file_name, network->target);

	return network;
}

void save_network(char *output_file_name, Network *network)
{
	file_write_data(output_file_name, network->layers);
	file_write_values_first(output_file_name, network->visual);

	for (int i = 0; i < (network->layers - 1); i++)
	{
		file_write_values(output_file_name, network->layer[i]);
	}
}

//OPTIMIZATION//

void stochastic_gradient_descent(Network *N, double learning_rate)
{
	for (int layer = 0; layer < N->layers - 1; layer++)
	{
		for (int i = 0; i < N->layer[layer]->sw[0]; i++)
		{
			for (int k = 0; k < N->layer[layer]->sw[1]; k++)
			{
				N->layer[layer]->weights[i][k] += -(learning_rate) * N->layer[layer]->dweights[i][k];
			}
		}

		for (int i = 0; i < N->layer[layer]->neurons; i++)
		{
			N->layer[layer]->biases[i] += -(learning_rate) * N->layer[layer]->dbiases[i];
		}
	}

	for (int i = 0; i < N->visual->sw[0]; i++)
	{
		for (int j = 0; j < N->visual->sw[1]; j++)
		{
			N->visual->weights[i][j] += -(learning_rate) * N->visual->dweights[i][j];
		}
	}

	for (int i = 0; i < N->visual->neurons; i++)
	{
		N->visual->biases[i] += -(learning_rate) * N->visual->dbiases[i];
	}
}

//TRAINING//

void train_network(Network *N, char *input_file_name, int epochs)
{
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		if (epoch == 0)
		{
			activate_network(N);
			calculate_loss(N);
			calculate_accuracy(N);
			printf("EPOCH: %d\tLOSS: %f\tACCURACY: %f% \n NETWORK OUTPUT:\n", epoch, N->loss, N->accuracy * 100);
			for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
			{
				for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
				{
					printf("%f ", N->layer[N->layers - 2]->output[i][j]);
				}
				printf("\n");
			}
		}
		else
		{
			backprop_network(N);
			stochastic_gradient_descent(N, Alpha);
			activate_network(N);
			if (epoch % 1000 == 0)
			{
				calculate_loss(N);
				calculate_accuracy(N);
				printf("EPOCH: %d\tLOSS: %f\tACCURACY: %f% \n NETWORK OUTPUT:\n", epoch, N->loss, N->accuracy * 100);
				for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
				{
					for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
					{
						printf("%f ", N->layer[N->layers - 2]->output[i][j]);
					}
					printf("\n");
				}
			}
		}
	}

	calculate_loss(N);
	calculate_accuracy(N);
	printf("EPOCH: %d\tLOSS: %f\tACCURACY: %f% \n NETWORK OUTPUT:\n", epochs, N->loss, N->accuracy * 100);
	for (int i = 0; i < N->layer[N->layers - 2]->so[0]; i++)
	{
		for (int j = 0; j < N->layer[N->layers - 2]->so[1]; j++)
		{
			printf("%f ", N->layer[N->layers - 2]->output[i][j]);
		}
		printf("\n");
	}
}
