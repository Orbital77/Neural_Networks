
#include <stdio.h>
#include <stdlib.h>

//VARIABLES//

int a_s[2] = {10, 10};

int b_s[2] = {10, 10};

int r_s[2] = {10, 10};

int bound = 10;

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

//ARRAY//

double get_max_num(double a[][a_s[1]], double b[][b_s[1]], int shape_a_max, int shape_b_max)
{
	int max_1 = 0;
	int max_2 = 0;
	int result;

	for (int i = 0; i < shape_a_max; i++)
	{
		for(int j = 0; j < shape_b_max; j++)
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

int is_1d(int shape[])
{
        if (shape[0] == 1)
        {
                return 1;
        }
        else
        {
                return 0;
        }
}

int is_2d(int shape[])
{
        if (shape[0] != 1)
        {
                return 1;
        }
        else
        {
                return 0;
        }
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
		r[0][0] += a[0][i]*b[0][i];
	}
}

void dot_product_1_2(double a[][a_s[1]], double b[][b_s[1]], double result[][r_s[1]], int shape_b[])
{
	for (int row = 0; row < shape_b[0]; row++)
	{
		for (int column = 0; column < shape_b[1]; column ++)
		{
			result[0][row] += a[0][column] * b[row][column];
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

	if(is_1d(shape_a) && is_1d(shape_b)) choice = 1;

	if(is_1d(shape_a) && is_2d(shape_b)) choice = 2;

	if(is_2d(shape_a) && is_2d(shape_b)) choice = 3;

	switch(choice)
	{
		case 1:
			printf("1\n");
			set_shapes(shape_a, shape_b, shape_r);
			dot_product_1_1(a, b, r, shape_b[1]);
			break;
		case 2:
			printf("2\n");
			set_shapes(shape_a, shape_b, shape_r);
			dot_product_1_2(a, b, r, shape_b);
			break;
		case 3:
			printf("3\n");
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

//NEURONS//
/*
void calculate()
{

}

void forward_input()
{

}

void forward_layer(double input[][], double output[][])
{

}

void forward_output()
{

}
*/
//INIT//

int main()
{
	double x[1][3] = {1, 2, 3};
	double y[1][3] = {0.2, 0.8, -0.5};
	double z = 3.0;

	int x_r = 1;
	int x_c = sizeof(x[0]) / sizeof(x[0][0]);

	int y_r = 1;
	int y_c = sizeof(y[0]) / sizeof(y[0][0]);

	double result[x_r][x_r];
	double weights_new[weights_c][weights_r];

	int a_f[2] = {x_r, x_c};
	int b_f[2] = {y_r, y_c};
	int r_f[2] = {x_r, x_r};

	set_shapes(a_f, b_f, r_f);
	dot(x, y, result, a_f, b_f, r_f);
	display_1_1(result);

	//set_shapes(a_f, b_f, r_f);
	//transpose(weights, answer, weights_r, weights_c);
	//dot(inputs_2, answer, result, a_f, b_f, r_f);
	//display_2_2(result, r_f);

	//printf("%f \n", get_max_num(weights, inputs_2, b_f[0], a_f[1]));
}

