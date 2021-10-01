#include <stdio.h>

//This script is identical in function to its python equivalent

//This is code for a single neuron with 3 inputs, 1 cell, 3 connections with weights, and 1 output

double inputs[3] = {1.2, 5.1, 2.1};
double weights[3] = {3.1, 2.1, 8.7};
double biases = 3.0;

int size_inputs = (sizeof(inputs) / sizeof(inputs[0]));
int size_weights = (sizeof(weights) / sizeof(weights[0]));

double think(double input[], double weight[], double bias)
{
	double result;
	double p_result = 0;
	double result_2 = 0;

	int size_input = (sizeof(input) / sizeof(input[0]));

	for (int i = 0; i <= size_inputs; i++)
	{
		result = input[i]*weight[i] + p_result;

		p_result = result;

		if (i >= size_inputs)
		{
			result += bias;
			return result;
		}
	}
}

int main()
{
	printf("Result: %f\n", think(inputs, weights, biases));
}
