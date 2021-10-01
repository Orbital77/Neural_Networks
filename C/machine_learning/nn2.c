#include <stdio.h>
#include <stdlib.h>

//VARIABLES//

int temp[2] = {3, 4};

//double x[3] = {1, 2, 3};
//double y[3] = {2, 3, 4};

double inputs[4] = {1, 2, 3, 2.5};

double inputs_2[3][4] = {{1.0, 2.0, 3.0, 2.5},
		       {2.0, 5.0, -1.0, 2.0},
		       {-1.5, 2.7, 3.3, -0.8}};

double weights[3][4] = {{0.2, 0.8, -0.5, 1.0},
			{0.5, -0.91, 0.26, -0.5},
			{-0.26, -0.27, 0.17, 0.87}};

//double biases[3] = {2, 3, 0.5};

//int size_inputs = (sizeof(inputs) / sizeof(inputs[0]));

int inputs_c = sizeof(inputs_2[0]) / sizeof(inputs_2[0][0]);
int inputs_r = sizeof(inputs_2) / sizeof(inputs_2[0]);
int inputs_total;

int weights_c = sizeof(weights[0]) / sizeof(weights[0][0]);
int weights_r = sizeof(weights) / sizeof(weights[0]);
int weights_total;

//int size_weights = (sizeof(weights) / sizeof(weights[0]));

//TRANSPOSE//

void transpose(double input[][weights_c], double output[][weights_c], int rows, int cols)
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

double dot_product_1_1(double a[], double b[], int size)
{
	double result;

	double p_result = 0;

	for (int i = 0; i <= size; i++)
	{
		result = a[i]*b[i] + p_result;

		p_result = result;

		if (i >= size)
		{
			return result;
		}
	}
}

double * dot_product_1_2(double a[], double b[temp[0]][temp[1]], int size_a, int size_b[], double *result_f)
{
	double result;
	double p_result = 0;
	double p_result_2 = 0;

	for (int row = 0; row < size_b[0]; row++)
	{
		p_result = 0;

		for (int column = 0; column < size_b[1]; column++)
		{
			result = a[column]*b[row][column] + p_result;
			p_result = result;

			result_f[row] = result;
		}
	}

	return result_f;
}

void dot_product_2_2(double a[][3], double b[][4], int size_a[], int size_b[], double result[][3]) //weights_r, inputs_c, inputs_r
{
/*
	for (int row = 0; row < size_a[0]; row++)
	{
		for (int column = 0; column < size_b[1]; column++)
		{
			result_f_2[row][column] = 0;
		}
	}
*/
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

void dp_2_2(double a[][3], double b[][2], int size_a[], int size_b[], double result[][2])
{
/*
        for (int row = 0; row < size_a[0]; row++)
        {
                for (int column = 0; column < size_b[1]; column++)
                {
                        result_f_2[row][column] = 0;
                }
        }
*/
        int row, column, iteration;
        for (row = 0; row < size_a[0] + 1; row++)
        {
                for (column = 0; column < size_b[1] + 1; column++)
                {
                        for (iteration = 0; iteration < size_a[1]; iteration++)
                        {
                                result[row][column] = (a[iteration][row] * b[iteration][column]) + result[row][column];
                        }
                }
        }
}

//DISPLAY//

void print_transpose(double input[][temp[1]], int rows, int cols)
{
	int i, j;
	for (i = 0; i < cols; i++)
	{
		for (j = 0; j < rows; j++)
		{
			printf("%f ", input[i][j]);
			if (j == rows - 1) printf("\n");
		}
	}
}

void display_2_2(double result[][inputs_r], int size_a[], int size_b[])
{
	int i, j;
	for (i = 0; i < size_a[0]; i++)
	{
		for (j = 0; j < size_b[1]; j++)
		{
			printf("%f ", result[i][j]);
			if (j == size_b[1] - 1) printf("\n");
		}
	}
}
/*
void get_inputs(double a[][3], double b[][2], int shape_a[], int shape_b[])
{
	int i, j;
	printf("\nEnter elements of matrix 1: \n");
	for (i = 0; i < shape_a[0]; i++)
	{
		for (j = 0; j < shape_a[1]; j++)
		{
			printf("Enter elements a%d%d: ", i + 1, j + 1);
			scanf("%f", &a[i][j]);
		}
	}

	printf("\nEnter elements of matrix 2: \n");
	for (i = 0; i < shape_b[0]; i++)
	{
		for (j = 0; j < shape_b[1]; j++)
		{
			printf("Enter elements b%d%d: ", i + 1, j + 1);
			scanf("%f", &b[i][j]);
		}
	}
}
*/
//INIT//

int main()
{
	double matrix_1[2][3];
	double matrix_2[3][2];

	int temp_2[2] = {2, 3};
	int temp_3[2] = {3, 2};

	double result_2_2[inputs_r][inputs_r]; //inputs_r for both

	inputs_total = inputs_r * inputs_c;
	weights_total = weights_r * weights_c;

	int inputs_f[2] = {inputs_r, inputs_c};
	int weights_f[2] = {weights_r, weights_c};

	double answer[weights_c][weights_r]; //was weights_r weights_c

	double answer_2[2][2];

	transpose(weights, answer, weights_r, weights_c);

	print_transpose(answer, weights_r, weights_c);

        int size_a_c = sizeof(answer[0]) / sizeof(answer[0][0]);
        int size_a_r = sizeof(answer) / sizeof(answer[0]);
        int size_a_total = size_a_r * size_a_c;
        int size_a_f[2] = {size_a_r, size_a_c};

	printf("\n");

	dot_product_2_2(answer, inputs_2, size_a_f, inputs_f, result_2_2);

	display_2_2(result_2_2, inputs_f, size_a_f);
}
