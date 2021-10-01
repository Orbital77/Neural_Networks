//INCLUDES//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//DEFINES//

#define Bound_1 128
#define Bound_2 2
#define Bound_3 255

//STRUCTS//

typedef struct
{
	int depth;
	int neurons;
	int p_neurons;
	double **input;
	double **output;
	double **weights;
	double *biases;
	int si[Bound_2];
	int so[Bound_2];
	int sw[Bound_2];
} N_Layer;

typedef struct
{
	int depth;
	int neurons;
	int batch_size;
	double **input;
	double **output;
	double **weights;
	double *biases;
	int si[Bound_2];
	int so[Bound_2];
	int sw[Bound_2];
} V_Layer;

typedef struct
{
	int layers;
	V_Layer *visual;
	N_Layer **layer;
} Network;

//BIAS//

void bias_1(double ***input, double **bias, int n_neurons)
{
	for (int i = 0; i < n_neurons; i++)
	{
		(*input)[0][i] += (*bias)[i];
	}
}

void bias_2(double ***input, double **bias, int n_neurons)
{
	for (int row = 0; row < n_neurons; row++)
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
		bias_2(input, bias, n_neurons);
	}
}

//TRANSPOSE//

void transpose_weights(N_Layer *layer, double ***input, int rows, int cols)
{
	double **temp = malloc(sizeof(double*) * cols);

        for (int i = 0; i < cols; i++)
        {
                temp[i] = malloc(sizeof(double) * rows);
        }

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			temp[j][i] = (*input)[i][j];
		}
	}

	*input = realloc(*input, sizeof(double*) * cols);

	for (int i = 0; i < cols; i++)
	{
		(*input)[i] = realloc((*input)[i], sizeof(double) * rows);
	}

	for (int i = 0; i < cols; i++)
	{
		(*input)[i] = malloc(sizeof(double) * rows);
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			(*input)[j][i] = temp[j][i];
		}
	}

	for (int i = 0; i < cols; i++)
	{
		free(temp[i]);
	}

	free(temp);

	layer->sw[0] = cols;
	layer->sw[1] = rows;
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
	for (int row = 0; row < shape_a[0]; row++)
	{
		for (int col = 0; row < shape_b[1]; col++)
		{
			for (int iteration = 0; iteration < shape_a[1]; iteration++)
			{
				(*r)[row][col] += (*a)[row][iteration] * (*b)[iteration][col];
			}
		}
	}
}

void dot(N_Layer *layer, double ***a, double ***b, double ***r, int shape_a[], int shape_b[], int shape_r[])
{
	int choice;

	if (layer->si[0] == 1 && layer->sw[0] == 1) choice = 1;
	if (layer->si[0] == 1 && layer->sw[0] > 1) choice = 2;
	if (layer->si[0] > 1 && layer->sw[0] > 1) choice = 3;

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

void dot_first(V_Layer *layer, double ***a, double ***b, double ***r, int shape_a[], int shape_b[], int shape_r[])
{
        int choice;

        if (layer->si[0] == 1 && layer->sw[0] == 1) choice = 1;
        if (layer->si[0] == 1 && layer->sw[0] > 1) choice = 2;
        if (layer->si[0] > 1 && layer->sw[0] > 1) choice = 3;

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

		printf("\nSI: %d %d\n\n", network->layer[layer]->si[0], network->layer[layer]->si[1]);
		printf("SO: %d %d\n\n", network->layer[layer]->so[0], network->layer[layer]->so[1]);
		printf("SW: %d %d\n", network->layer[layer]->sw[0], network->layer[layer]->sw[1]);
	}
}

//MEMORY//

void allocate_memory_biases(double **biases, int cols)
{
	*biases = malloc(sizeof(double) * cols);
}

void allocate_memory_input(double ***input, int rows, int cols)
{
	*input = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*input)[i] = malloc(sizeof(double) * cols);
	}
}

void allocate_memory_network(Network *network, int n_layers, int iteration)
{
	if (iteration == 0)
	{
		network->visual = malloc(sizeof(V_Layer));
		network->layer = malloc(sizeof(N_Layer*) * (n_layers - 1));
	}
	else if (iteration > 0)
	{
		network->layer[iteration - 1] = malloc(sizeof(N_Layer));
	}
}

void allocate_memory_output(double ***output, int rows, int cols)
{
	*output = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*output)[i] = malloc(sizeof(double) * cols);
	}
}

void allocate_memory_weights(double ***weights, int rows, int cols)
{
	*weights = malloc(sizeof(double*) * rows);
	for (int i = 0; i < rows; i++)
	{
		(*weights)[i] = malloc(sizeof(double) * cols);
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
	}

	free(network->visual->output);
	network->visual->output = NULL;

	for (int i = 0; i < network->visual->sw[0]; i++)
	{
		free(network->visual->weights[i]);
		network->visual->weights[i] = NULL;
	}

	free(network->visual->weights);
	network->visual->weights = NULL;

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
			}

			free(network->layer[layer]->output);
			network->layer[layer]->output = NULL;

			for (int i = 0; i < network->layer[layer]->sw[0]; i++)
			{
				free(network->layer[layer]->weights[i]);
				network->layer[layer]->weights[i] = NULL;
			}

			free(network->layer[layer]->weights);
			network->layer[layer]->weights = NULL;

			free(network->layer[layer]->biases);
			network->layer[layer]->biases = NULL;

			free(network->layer[layer]);
			network->layer[layer] = NULL;
		}
	}
	free(network->layer);
	network->layer = NULL;
	free(network->visual);
	network->visual = NULL;
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
	double weight;
	double bias;
	double input;
	int layers;
	int depth = 0;
	int cols;
	int test;

	fp = fopen(input_file_name, "r");

	fscanf(fp, "BATCH SIZE: %d\n\n", &layer->batch_size);
	fscanf(fp, "INPUTS: %d\n", &cols);

	allocate_memory_input(&layer->input, layer->batch_size, cols);

	for (int row = 0; row < layer->batch_size; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(fp, "%lf ", &layer->input[row][col]);
		}
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
		fscanf(fp, "\n");
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
	double weight;
	double bias;
	double input;
	int layers;
	int depth = 0;
	int cols;
	int test;

	fp = fopen(input_file_name, "r");

	fscanf(fp, "BATCH SIZE: %d\n\n", &layer->batch_size);
	fscanf(fp, "INPUTS: %d\n", &cols);

	allocate_memory_input(&layer->input, layer->batch_size, cols);

	for (int row = 0; row < layer->batch_size; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			fscanf(fp, "%lf ", &layer->input[row][col]);
		}
	}

	fclose(fp);

	layer->si[0] = layer->batch_size;
	layer->si[1] = cols;
}

void file_read_layer(char *data_file_name, N_Layer *layer)
{
	FILE *fp;
	int layers;
	int depth;
	int bias;
	int n_p_neurons = 0;
	char buf[Bound_3];
	double temp;
	int rows;
	int cols;

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

				if (n_p_neurons != 0) layer->p_neurons = n_p_neurons;

				fscanf(fp, "\n");
				}
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

				for (int col = 0; col < cols; col++)
				{
					for (int row = 0; row < rows; row++)
					{
						fscanf(fp, "%lf ", &temp);
					}
					fscanf(fp, "\n");
				}

				fscanf(fp, "\n");
			}
		}

	}
	else
	{
		printf("Could not open file %s in function 'read_layer'", data_file_name);
	}

	fclose(fp);
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

	for (int row = 0; row < layer->neurons; row++)
	{
		for (int col = 0; col < layer->p_neurons; col++)
		{
			fprintf(fp, "%f ", layer->weights[row][col]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n");

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

	for (int row = 0; row < layer->neurons; row++)
	{
		for (int col = 0; col < layer->batch_size; col++)
		{
			fprintf(fp, "%f ", layer->weights[row][col]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "\n");

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

void layer_forward(N_Layer *layer)
{
	dot(layer, &layer->input, &layer->weights, &layer->output, layer->si, layer->sw, layer->so);
	add_bias(&layer->output, &layer->biases, layer->so, layer->neurons);
}

void layer_forward_first(V_Layer *layer)
{
	dot_first(layer, &layer->input, &layer->weights, &layer->output, layer->si, layer->sw, layer->so);
	add_bias(&layer->output, &layer->biases, layer->so, layer->neurons);
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
	allocate_memory_input(&layer->input, 1, layer->p_neurons);
	layer->si[0] = layer->neurons;
	layer->si[1] = layer->p_neurons;
	allocate_memory_output(&layer->output, layer->neurons, layer->neurons);
	layer->so[0] = layer->neurons;
	layer->so[1] = layer->neurons;
	get_output(layer, p_layer);
}

void init_first_layer(char *data_file_name, char *input_file_name, V_Layer *layer)
{
	layer->depth = 0;
	file_read_first_layer(data_file_name, input_file_name, layer);
	layer->so[0] = layer->batch_size;
	layer->so[1] = layer->neurons;
	allocate_memory_output(&layer->output, layer->batch_size, layer->neurons);
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
	get_first_output(layer, p_layer);
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

//NETWORK//

Network *create_network(char *input_file_name, int n_layers, int network_shape[])
{
	Network network;
	Network *N = &network;

	N->layers = n_layers;

	if (n_layers > 1)
	{
		srand(n_layers % 2);
	}
	else
	{
		srand(0);
	}

	allocate_memory_network(N, n_layers, 0);
	file_read_input(input_file_name, N->visual);
	N->visual->depth = 0;
	N->visual->neurons = network_shape[0];

	N->visual->sw[0] = N->visual->neurons;
	N->visual->sw[1] = N->visual->si[1];

	allocate_memory_weights(&N->visual->weights, N->visual->sw[0], N->visual->sw[1]);

	for (int row = 0; row < N->visual->sw[0]; row++)
	{
		for (int col = 0; col < N->visual->sw[1]; col++)
		{
			N->visual->weights[row][col] = rand();
		}
	}

	allocate_memory_biases(&N->visual->biases, network_shape[0]);

	for (int col = 0; col < N->visual->neurons; col++)
	{
		N->visual->biases[col] = 0;
	}

	N->visual->so[0] = N->visual->batch_size;
	N->visual->so[1] = N->visual->neurons;

	allocate_memory_output(&N->visual->output, N->visual->so[0], N->visual->so[1]);

	layer_forward_first(N->visual);

	if (N->layers > 1)
	{
		allocate_memory_network(N, N->layers, 1);
		N->layer[0]->depth = 1;
		N->layer[0]->neurons = network_shape[1];
		N->layer[0]->p_neurons = N->visual->neurons;

		N->layer[0]->si[0] = N->visual->so[0];
		N->layer[0]->si[1] = N->visual->so[1];

		allocate_memory_input(&N->layer[0]->input, N->layer[0]->si[0], N->layer[0]->si[1]);

		get_first_output(N->layer[0], N->visual);

		N->layer[0]->sw[0] = N->layer[0]->neurons;
		N->layer[0]->sw[1] = N->layer[0]->si[1];

		allocate_memory_weights(&N->layer[0]->weights, N->layer[0]->sw[0], N->layer[0]->sw[1]);

		for (int row = 0; row < N->layer[0]->sw[0]; row++)
		{
			for (int col = 0; col < N->layer[0]->sw[1]; col++)
			{
				N->layer[0]->weights[row][col] = rand();
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

		layer_forward(N->layer[0]);

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

				get_output(N->layer[layer], N->layer[layer - 1]);

				N->layer[layer]->sw[0] = N->layer[layer]->neurons;
				N->layer[layer]->sw[1] = N->layer[layer]->si[1];

				allocate_memory_weights(&N->layer[layer]->weights, N->layer[layer]->sw[0], N->layer[layer]->sw[1]);

				for (int row = 0; row < N->layer[layer]->sw[0]; row++)
				{
					for (int col = 0; col < N->layer[layer]->sw[1]; col++)
					{
						N->layer[layer]->weights[row][col] = rand();
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

				layer_forward(N->layer[layer]);
			}
		}
	}

	return N;
}

void load_network(char *data_file_name, char *input_file_name, Network *network, int n_layers)
{
	network->layers = n_layers;
	allocate_memory_network(network, n_layers, 0);
	init_first_layer(data_file_name, input_file_name, network->visual);
	layer_forward_first(network->visual);
	allocate_memory_network(network, n_layers, 1);
	init_second_layer(data_file_name, network->layer[0], network->visual);
	layer_forward(network->layer[0]);
	if (n_layers > 2)
	{
		for (int i = 1; i < n_layers; i++)
		{
			allocate_memory_network(network, n_layers, i + 1);
			init_layer(data_file_name, network->layer[i], network->layer[i-1], i + 1);
			layer_forward(network->layer[i]);
		}
	}
}

void save_network(char *data_file_name, Network *network)
{
	file_write_data(data_file_name, network->layers);
	file_write_values_first(data_file_name, network->visual);

	for (int i = 0; i < (network->layers - 1); i++)
	{
		file_write_values(data_file_name, network->layer[i]);
	}
	free_memory_network(network, network->layers);
}

//INIT//

int main(int argc, char **argv)
{
	if (argc == 3)
	{
		Network network_1;

		load_network(argv[1], argv[2], &network_1, 2);

		display_network(&network_1);

		free_memory_network(&network_1, 2);

		return 0;
	}
	else
	{
		printf("Error: Invalid amount of arguments %d\n", argc);
		return 1;
	}
}
