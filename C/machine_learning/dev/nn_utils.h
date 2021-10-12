#ifndef NN_UTILS_H
#define NN_UTILS_H

#define Bound_1 128
#define Bound_2 2
#define Bound_3 255
#define Bound_4 25

typedef struct{
	int depth;
	int neurons;
	int p_neurons;
	double **input;
	double **output;
	double **weights;
	double *biases;
	double **dinput;
	double **dweights;
	double *dbiases;
	int si[Bound_2];
	int sdi[Bound_2];
	int so[Bound_2];
	int sw[Bound_2];
	int sdw[Bound_2];
	char method[Bound_4];
	double **dactivation;
} N_Layer;

typedef struct{
	int depth;
	int neurons;
	int batch_size;
	double **input;
	double **output;
	double **weights;
	double *biases;
	double **dinput;
	double **dweights;
	double *dbiases;
	int si[Bound_2];
	int sdi[Bound_2];
	int so[Bound_2];
	int sw[Bound_2];
	int sdw[Bound_2];
	char method[Bound_4];
	double **dactivation;
} V_Layer;

typedef struct{
	int classes;
	int batch_size;
	int rows;
	int label;
	int **one_hot;
	int items;
	int sh[Bound_2];
	double *expected;
} One_Hot;

typedef struct{
	int data_sets;
	int layers;
	V_Layer *visual;
	N_Layer **layer;
	One_Hot *target;
	double *output;
	double loss;
	double accuracy;
} Network;

//ARRAY//

void clip(Network *N, double min, double max);

void diagonal(double ***input_array, int shape[], double ***result);

void get_max_values(Network *N);

//MATH//

double f(double x, char *function);

double approximate_derivative(double x, char *function);

double d_rand(double min, double max);

double d_softmax(double x, double y);

//BIAS//

void bias_1(double ***input, double **bias, int n_neurons);

void bias_2(double ***input, double **bias, int shpae_input[], int n_neurons);

void add_bias(double ***input, double **bias, int shape_input[], int n_neurons);

//TRANSPOSE//

void transpose_input(N_Layer *layer);

void transpose_input_first(V_Layer *layer);

void transpose_weights(N_Layer *layer);

void transpose_weights_first(V_Layer *layer);

//DOT PRODUCT//

void dot_product_1_1(double ***a, double ***b, double ***r, int shape);

void dot_product_1_2(double ***a, double ***b, double ***r, int shape_b[]);

void dot_product_2_2(double ***a, double ***b, double ***r, int shape_a[], int shape_b[]);

void dot(N_Layer *layer, double ***a, double ***b, double ***r, int shape_a[], int shape_b[]);

void dot_first(V_Layer *layer, double ***a, double ***b, double ***r, int shape_a[], int shape_b[]);

//DISPLAY//

void display_1_1(double ***result);

void display_1_2(double ***result, int cols);

void display_2_2(double ***result, int shape[]);

void display_visual_layer(V_Layer *layer);

void display_layer(N_Layer *layer);

void display_network(Network *network);

//MEMORY//

void allocate_memory_biases(double **biases, int cols);

void allocate_memory_dactivation(double ***dactivation, int rows, int cols);

void allocate_memory_delta(double ***delta, int rows, int cols);

void allocate_memory_dbiases(double **dbiases, int cols);

void allocate_memory_dinput(double ***dinput, int rows, int cols);

void allocate_memory_dweights(double ***dweights, int rows, int cols);

void allocate_memory_errors(double ***errors, int rows, int cols);

void allocate_memory_expected(double **expected, int cols);

void allocate_memory_hot(int ***hot, int rows, int cols);

void allocate_memory_input(double ***input, int rows, int cols);

void allocate_memory_network(Network *network, int n_layers, int iteration);

void allocate_memory_op(double ***op, int rows, int cols);

void allocate_memory_output(double ***output, int rows, int cols);

void allocate_memory_weights(double ***weights, int rows, int cols);

void allocate_memory_layer_set(N_Layer ***layer, int n_layers);

void free_memory_biases(double **biases);

void free_memory_hot(int ***hot, int rows);

void free_memory_input(double ***input, int rows);

void free_memory_network(Network *network, int n_layers);

void free_memory_output(double ***output, int rows);

void free_memory_weights(double ***weights, int rows);

void free_memory_layer_set(N_Layer ***layer, int n_layers);

void allocate_memory(double ***input, double ***output, double ***weights, double **biases, int input_rows, int input_cols, int weights_rows, int weights_cols, int n_neurons);

void free_memory(double ***input, double ***output, double ***weights, double **biases, int input_rows, int weights_rows, int output_rows);

void free_memory_first_layer(V_Layer *layer);

void free_memory_layer(N_Layer *layer);

//FILE//

void file_read_first_layer(char *data_file_name, char *input_file_name, V_Layer *layer);

void file_read_input(char *input_file_name, V_Layer *layer);

void file_read_target(char *input_file_name, One_Hot *hot);

void file_read_layer(char *data_file_name, N_Layer *layer);

void file_switch_target(char *input_file_name, int target_data_set, Network *N);

void file_write_data(char *filename, int n_layers);

void file_write_values(char *file_name, N_Layer *layer);

void file_write_values_first(char *data_file_name, V_Layer *layer);

void file_inject_input(char *input_file_name,  int batch_size, int n_inputs, N_Layer *layer);

//LOSS//

void calculate_loss(Network *N);

//ACCURACY//

void calculate_accuracy(Network *N);

//ACTIVATION FUNCTIONS//

void layer_activation_relu(N_Layer *layer);

void layer_activation_relu_first(V_Layer *layer);

void layer_activation_sigmoid(N_Layer *layer);

void layer_activation_sigmoid_first(V_Layer *layer);

void layer_activation_softmax(N_Layer *layer);

void layer_activation_softmax_first(V_Layer *layer);

//NEURONS//

void get_first_output(N_Layer *layer, V_Layer *p_layer);

void get_output(N_Layer *layer, N_Layer *p_layer);

int get_max_layer(Network *N);

int get_max_neurons(Network *N);

void layer_activate(N_Layer *layer);

void layer_activate_first(V_Layer *layer);

void layer_forward(N_Layer *layer);

void layer_forward_first(V_Layer *layer);

void set_values(int n_neurons, int n_p_neurons, int depth, N_Layer *layer);

void init_layer(char *file_name, N_Layer *layer, N_Layer *p_layer, int depth);

void init_first_layer(char *data_file_name,  char *input_file_name, V_Layer *layer);

void init_second_layer(char *data_file_name, N_Layer *layer, V_Layer *p_layer);

void inject(N_Layer *layer, double inputs[][Bound_1], double weights[][Bound_1], double biases[], int inputs_rows, int inputs_cols, int weights_rows, int weights_cols, int neurons, int p_neurons, int depth);

void inject_layer(N_Layer *layer, double ***inputs, double weights[][Bound_1], double biases[], int inputs_rows, int inputs_cols, int weights_rows, int weights_cols, int neurons, int p_neurons, int depth);

void inject_raw(N_Layer *layer, double inputs[][Bound_1], int inputs_rows, int inputs_cols, int depth);

//DERIVATIVES//

void d_relu(N_Layer *layer, N_Layer *n_layer);

void d_relu_first(V_Layer *layer, N_Layer *n_layer);

void d_relu_last(Network *N);

void d_sigmoid(N_Layer *layer, N_Layer *n_layer);

void d_sigmoid_first(V_Layer *layer, N_Layer *n_layer);

void d_sigmoid_last(Network *N);

void d_softmax_loss_categorical_cross_entropy(Network *N);

//NETWORK//

void activate_network(Network *N);

void backprop_network(Network *N);

Network *create_network(char *input_file_name, int n_layers, int network_shape[]);

Network *load_network(char *data_file_name, char *input_file_name, int n_layers);

void save_network(char *output_file_name, Network *network);

void update_network(Network *N, double learning_rate);

//OPTIMIZATION//

void stochastic_gradient_descent(Network *N, double learning_rate);

//TRAINING//

void train_network(Network *N, char *input_file_name, int epochs);

#endif
