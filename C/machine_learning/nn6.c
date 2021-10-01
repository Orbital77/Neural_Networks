//INCLUDES//

#include <stdio.h>
#include <stdlib.h>

//DEFINES//

#define Bound_1 128

#define Bound_2 2

//VARIABLES//

int a_s[2] = {10, 10};

int b_s[2] = {10, 10};

int r_s[2] = {10, 10};

int bound = 128;

double inputs[1][4] = {1.0, 2.0, 3.0, 2.5};

double inputs_2[3][4] = {{1.0, 2.0, 3.0, 2.5},
		       {2.0, 5.0, -1.0, 2.0},
		       {-1.5, 2.7, 3.3, -0.8}};

double weights[3][4] = {{0.2, 0.8, -0.5, 1.0},
			{0.5, -0.91, 0.26, -0.5},
			{-0.26, -0.27, 0.17, 0.87}};

double biases[3] = {2, 3, 0.5};

int inputs_1_c = sizeof(inputs) / sizeof(inputs[0]);

int inputs_c = sizeof(inputs_2[0]) / sizeof(inputs_2[0][0]);
int inputs_r = sizeof(inputs_2) / sizeof(inputs_2[0]);
int inputs_total;

int weights_c = sizeof(weights[0]) / sizeof(weights[0][0]);
int weights_r = sizeof(weights) / sizeof(weights[0]);
int weights_total;

double answer[4][3];

int a_r = sizeof(answer) / sizeof(answer[0]);
int a_c = sizeof(answer[0]) / sizeof(answer[0][0]);

void set_bound(int new_bound)
{
	bound = new_bound;
}

void set_shape_a(int r, int c)
{
	a_s[0] = r;
	a_s[1] = c;
}

void set_shape_b(int r, int c)
{
	b_s[0] = r;
	b_s[1] = c;
}

void set_shape_r(int r, int c)
{
	r_s[0] = r;
	r_s[1] = c;
}

void set_shapes(int a[], int b[], int r[])
{
	set_shape_a(a[0], a[1]);
	set_shape_b(b[0], b[1]);
	set_shape_r(r[0], r[1]);
}

//STRUCTS//

typedef struct
{
	int neurons;
	int p_neurons;
	double input[Bound_1][Bound_1];
	double output[Bound_1][Bound_1];
	double biases[Bound_1];
	double weights[Bound_1][Bound_1];
	int si[Bound_2];
	int so[Bound_2];
	int sw[Bound_2];
} N_Layer;

//ARRAY//

void bias_1(double input[][bound], double b[], int n_neurons)
{
	for (int i = 0; i < n_neurons; i++)
	{
		input[0][i] += b[i];
	}
}

void bias_2(double input[][bound], double b[], int n_neurons)
{
	for (int i = 0; i < n_neurons; i++)
	{
		for (int j = 0; j < n_neurons; j++)
		{
			input[i][j] += b[j];
		}
	}
}

void add_bias(double input[][bound], double b[], int n_neurons)
{
	if (n_neurons == 1)
	{
		bias_1(input, b, n_neurons);
	}
	else
	{
		bias_2(input, b, n_neurons);
	}
}

double get_max(double a[][a_s[1]], double b[][b_s[1]], int size_a_max, int size_b_max)
{
	int max_1 = 0;
	int max_2 = 0;
	int result;

	for (int i = 0; i < size_a_max; i++)
	{
		for(int j = 0; j < size_b_max; j++)
		{
			if (a[i][j] > max_1)
			{
				max_1 = a[i][j];
			}

			if (b[i][j] > max_2)
			{
				max_2 = b[i][j];
			}
		}
	}

	if (max_1 > max_2)
	{
		result = max_1;
	}
	else
	{
		result = max_2;
	}

	return result;
}

int get_max_shape(int a[], int b[], int size_max)
{
	int max_1 = 0;
	int max_2 = 0;

	for (int i = 0; i < size_max; i++)
	{
		if (a[i] > max_1) max_1 = a[i];
		if (b[i] > max_2) max_2 = b[i];
	}

	if (max_1 > max_2) return max_1;
	if (max_2 > max_1) return max_2;
}

int is_1d(int shape[], double input[][bound])
{
	for (int col = 0; col < shape[1]; col++)
	{
		if (input[1][col] == 0)
		{
			printf("input 1 %d: %f\n", col, input[0][col]);
			return 1;
		}
		else
		{
			return 0;
		}
	}
}

int is_2d(int shape[], double input[][bound])
{
	for (int col = 0; col < shape[1]; col++)
	{
		if (input[1][col] == 0)
		{
			continue;
		}
		else
		{
			printf("is 2d.\n");
			return 1;
		}
	}
	return 0;
}

//TRANSPOSE//

void transpose(double input[][weights_c], double output[][weights_r], int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output[j][i] = input[i][j];
		}
	}
}

//DOT PRODUCT//

void dot_product_1_1(double a[][a_s[1]], double b[][b_s[1]], double r[][r_s[1]], int size)
{
	for (int i = 0; i < size; i++)
	{
		r[0][0] += a[0][i] * b[0][i];
	}
}

void dot_product_1_2(double a[][a_s[1]], double b[][b_s[1]], double r[][r_s[1]], int shape_b[])
{
	for (int row = 0; row < shape_b[0]; row++)
	{
		for (int column = 0; column < shape_b[1]; column ++)
		{
			r[0][row] += a[0][column] * b[row][column];
			printf("a: %f\n", a[0][column]*b[row][column]);
			printf("b: %f\n", b[row][column]);
			printf("%d %d %f\n", row, column, r[0][row]);
		}
	}
}

void dot_product_2_2(double a[][a_s[1]], double b[][b_s[1]], int size_a[], int size_b[], double result[][r_s[1]])
{
	int row, column, iteration;
	for (row = 0; row < size_a[0]; row++)
	{
		for (column = 0; column < size_b[1]; column++)
		{
			for (iteration = 0; iteration < size_a[1]; iteration++)
			{
				result[row][column] += a[row][iteration] * b[iteration][column];
			}
		}
	}
}

void dot(double a[][a_s[1]], double b[][b_s[1]], double r[][r_s[1]], int shape_a[], int shape_b[], int shape_r[])
{
	int choice;

	if(is_1d(shape_a, a) && is_1d(shape_b, b)) choice = 1;

	if(is_1d(shape_a, a) && is_2d(shape_b, b)) choice = 2;

	if(is_2d(shape_a, a) && is_2d(shape_b, b)) choice = 3;

	switch(choice)
	{
		case 1:
			set_shapes(shape_a, shape_b, shape_r);
			dot_product_1_1(a, b, r, shape_b[1]);
			printf("um\n");
			break;
		case 2:
			set_shapes(shape_a, shape_b, shape_r);
			dot_product_1_2(a, b, r, shape_b);
			printf("Now this one!\n");
			break;
		case 3:
			set_shapes(shape_a, shape_b, shape_r);
			dot_product_2_2(a, b, shape_a, shape_b, r);
			break;
		default:
			printf("Invalid input for function 'dot'\n");
			break;
	}
}

//DISPLAY//

void display_1_1(double result[][r_s[1]])
{
	printf("%f\n", result[0][0]);
}

void display_1_2(double result[][r_s[1]], int shape[])
{
	for (int i = 0; i < shape[1]; i++)
	{
		printf("%f ",  result[0][i]);
	}
	printf("\n");
}

void display_2_2(double result[][r_s[1]], int shape[])
{
	for (int i = 0; i < shape[0]; i++)
	{
		for (int j = 0; j < shape[1]; j++)
		{
			printf("%f ", result[i][j]);
		}
		printf("\n");
	}
}

void display_transpose(double input[][bound], int rows, int cols)
{
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			printf("%f ", input[i][j]);
			if (j == rows - 1) printf("\n");
		}
	}
}

//MEMORY//

void allocate_memory_input(double** input, int rows, int cols)
{
	input = malloc(sizeof(double*) * rows);
	for (int i = 0; i < cols; i++)
	{
		input[i] = malloc(sizeof(double) * cols);
	}
}

void free_memory_input(double** input, int rows, int cols)
{
	for (int i = 0; i < cols; i++)
	{
		free(input[i]);
		input[i] = NULL;
	}
	free(input);
	input = NULL;
}

//NEURONS//

void forward_layer_old(double input[][bound], double op[][bound], double output[][bound], double b[], int shape_input[], int shape_op[], int shape_output[], int neurons)
{
	set_bound(get_max_shape(shape_input, shape_output, 2));
	set_shapes(shape_input, shape_op, shape_output);
	dot(input, op, output, shape_input, shape_op, shape_input);
	add_bias(output, b, neurons);
}

void forward_layer(N_Layer *layer)
{
	set_bound(get_max_shape(layer->si, layer->sw, 2));
	set_shapes(layer->si, layer->sw, layer->so);
	dot(layer->input, layer->weights, layer->output, layer->si, layer->sw, layer->so);
	add_bias(layer->output, layer->biases, layer->neurons);
}

void set_values(double input[][bound], double b[], double w[][bound], int n_neurons, int n_p_neurons, N_Layer *layer)
{
	int input_r = sizeof(layer->input) / sizeof(layer->input[0]);
	int input_c = sizeof(layer->input[0]) / sizeof(layer->input[0][0]);

	layer->neurons = n_neurons;

	layer->p_neurons = n_p_neurons;

	for (int row = 0; row < n_neurons; row++)
	{
		for (int col = 0; col < n_p_neurons; col++)
		{
			layer->input[row][col] = input[row][col];
		}
	}

	for (int i = 0; i < n_neurons; i++)
	{
		layer->biases[i] = b[i];
	}

	for (int row = 0; row < n_neurons; row++)
	{
		for (int col = 0; col < n_p_neurons; col++)
		{
			layer->weights[row][col] = w[row][col];
		}
	}

	layer->si[0] = input_r;
	layer->si[1] = input_c;

	layer->sw[0] = n_neurons;
	layer->sw[1] = n_p_neurons;

	if (input_r > 1)
	{
		layer->so[0] = n_neurons;
		layer->so[1] = n_neurons;
	}
	else
	{
		layer->so[0] = 1;
		layer->so[1] = n_neurons;
	}
}

//INIT//

int main()
{
	N_Layer layer_1;
	N_Layer *pointer_layer_1 = &layer_1;

	double output[3][3];
	int w_f[2] = {weights_r, weights_c};
	int i_f[2] = {1, 4};

	set_values(inputs, biases, weights, 3, 4, pointer_layer_1);
	forward_layer(pointer_layer_1);
	//forward_layer_old(layer_1.input, layer_1.weights, output, layer_1.biases, layer_1.si, layer_1.sw, layer_1.so, layer_1.neurons);
	display_1_2(layer_1.output, layer_1.so);
	printf("is 1d: %d\n", is_1d(w_f, weights));
	printf("Weights 1 0: %f\n", layer_1.weights[1][0]);

	//------------------------

	//double x[3][4] = {{1.0, 2.0, 3.0, 2.5},
	//		  {2.0, 5.0, -1.0, 2.0},
	//		  {-1.5, 2.7, 3.3, -0.8}};

	//double y[3][4];

	//int x_r = 1;
	//int x_c = sizeof(x[0]) / sizeof(x[0][0]);

	//int y_r = 1;
	//int y_c = sizeof(y[0]) / sizeof(y[0][0]);

	//double result[x_c][x_c];
	//double weights_new[weights_c][weights_r];

	//int a_f[2] = {x_r, x_c};
	//int b_f[2] = {weights_r, weights_c};
	//int r_f[2] = {weights_r, weights_r};

	//forward_layer(x, weights, result, biases, a_f, b_f, r_f, 3);
	//display_2_2(result, r_f);

	//-------------------------

	//display_2_2(inputs_2, a_f);

	//printf("\n");

	//transpose(weights, weights_new, weights_r, weights_c);
	//display_transpose(weights_new, weights_r, weights_c);

	//printf("\n");

	//forward_layer(inputs_2, weights_new, result, biases, a_f, b_f, r_f, 3);
	//display_2_2(result, r_f);

	//set_bound(get_max_shape(a_f, b_f, 2));
	//set_shapes(a_f, b_f, r_f);
	//dot(x, weights, result, a_f, b_f, r_f);
	//display_1_2(result, r_f);

	//set_shapes(a_f, b_f, r_f);
	//transpose(weights, answer, weights_r, weights_c);
	//dot(inputs_2, answer, result, a_f, b_f, r_f);
	//display_2_2(result, r_f);

	//printf("%f \n", get_max_num(weights, inputs_2, b_f[0], a_f[1]));
}

//Each neuron has its own bias
//Each neuron has a connection to every neuron before it
//Each connection has a weight

