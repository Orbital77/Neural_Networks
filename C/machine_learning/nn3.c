
#include <stdio.h>
#include <stdlib.h>

//VARIABLES//

int temp[2] = {3, 4};

double x[2][3] = {{3.0, -2.0, 5.0},
		  {3.0, 0.0, 4.0}};

double y[3][2] = {{2.0, 3.0},
		  {-9.0, 0.0},
		  {0.0, 4.0}};

int x_r = sizeof(x) / sizeof(x[0]);
int x_c = sizeof (x[0]) / sizeof(x[0][0]);

int y_r = sizeof(y) / sizeof(y[0]);
int y_c = sizeof(y[0]) / sizeof(y[0][0]);

double inputs[4] = {1, 2, 3, 2.5};

double inputs_2[3][4] = {{1.0, 2.0, 3.0, 2.5},
		       {2.0, 5.0, -1.0, 2.0},
		       {-1.5, 2.7, 3.3, -0.8}};

double weights[3][4] = {{0.2, 0.8, -0.5, 1.0},
			{0.5, -0.91, 0.26, -0.5},
			{-0.26, -0.27, 0.17, 0.87}};

int inputs_c = sizeof(inputs_2[0]) / sizeof(inputs_2[0][0]);
int inputs_r = sizeof(inputs_2) / sizeof(inputs_2[0]);
int inputs_total;

int weights_c = sizeof(weights[0]) / sizeof(weights[0][0]);
int weights_r = sizeof(weights) / sizeof(weights[0]);
int weights_total;

double answer[4][3];

int a_r = sizeof(answer) / sizeof(answer[0]);
int a_c = sizeof(answer[0]) / sizeof(answer[0][0]);

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

void dot_product_2_2(double a[][inputs_c], double b[][a_c], int size_a[], int size_b[], double result[][inputs_r]) //4, 3 ,3
{
	int row, column, iteration;
	for (row = 0; row < size_a[0]; row++) //2
	{
		for (column = 0; column < size_b[1]; column++) //2
		{
			//result[row][column] = 0;
			for (iteration = 0; iteration < size_a[1]; iteration++) //3
			{
				result[row][column] += a[row][iteration] * b[iteration][column];
			}
		}
	}
}

//DISPLAY//

void print_transpose(double input[][a_c], int rows, int cols)
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

void display_2_2(double result[][3], int size[])
{
	int i, j;
	for (i = 0; i < size[0]; i++)
	{
		for (j = 0; j < size[1]; j++)
		{
			printf("%f ", result[i][j]);
			//if (j == size[1] - 1) printf("\n");
		}
		printf("\n");
	}
}

//INIT//

int main()
{
	double result_2_2[inputs_r][inputs_r];
	double answer_2[2][2];

	int inputs_f[2] = {inputs_r, inputs_c};
	int weights_f[2] = {weights_r, weights_c};
	int a_f[2] = {a_r, a_c};
	int x_f[2] = {x_r, x_c};
	int y_f[2] = {y_r, y_c};
	int s[2] = {a_f[1], inputs_f[0]};

	transpose(weights, answer, weights_r, weights_c);
	print_transpose(answer, a_c, a_r);

	printf("\n");

	dot_product_2_2(inputs_2, answer, inputs_f, a_f, result_2_2); display_2_2(result_2_2, s);
}

