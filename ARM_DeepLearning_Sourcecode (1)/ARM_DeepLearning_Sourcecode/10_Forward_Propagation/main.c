#include "simple_neural_networks.h"


#define NUM_OF_FEATURES   2  // n values
#define NUM_OF_EXAMPLES		3  // m values
#define NUM_OF_HID_NODES	3
#define NUM_OF_OUT_NODES	1



/*
Train x:

  2 5 1
	8 5 8
	dim =  nx X m
*/

double raw_x[NUM_OF_FEATURES][NUM_OF_EXAMPLES] = {{2,5,1},
																									{8,5,8}};

/*Train y
	200  90   100 
	dim  = 1 x m*/
																									
double raw_y[1][NUM_OF_EXAMPLES] = {200,90,190};

/*Input layer to hidden layer weight matrix*/
double syn0[NUM_OF_HID_NODES][NUM_OF_FEATURES];

/*Hidden layer to output layer weight matrix*/
double syn1 [NUM_OF_HID_NODES];


double train_x[NUM_OF_FEATURES][NUM_OF_EXAMPLES];
double train_y[1][NUM_OF_EXAMPLES];

double train_x_eg1[NUM_OF_FEATURES];
double train_y_eg1;
double z1_eg1[NUM_OF_HID_NODES];
double a1_eg1[NUM_OF_HID_NODES];
double z2_eg1;
double yhat_eg1;

int main(void){

	 USART2_Init();
	 
  normalize_data_2d(NUM_OF_FEATURES,NUM_OF_EXAMPLES,raw_x,train_x);
	normalize_data_2d(1,NUM_OF_EXAMPLES,raw_y,train_y);
	
	train_x_eg1[0] =  train_x[0][0];
	train_x_eg1[1] =  train_x[1][0];
	
	train_y_eg1 =  train_y[0][0];
	
 /*Simple test to prove we have the right data*/	
//	printf("train_x_eg1 is [%f   %f] ",train_x_eg1[0],train_x_eg1[1]);
//	printf("\n\r");
//	printf("\n\r");
//	printf("train_y_eg1 is %f", train_y_eg1);
//	
	
	/*Initialize syn0 and syn1 weights*/
	weights_random_initialization(NUM_OF_HID_NODES,NUM_OF_FEATURES,syn0);
	weights_random_initialization_1d(syn1,NUM_OF_OUT_NODES);
	
	/*compute z1*/
	multiple_inputs_multiple_outputs_nn(train_x_eg1,NUM_OF_FEATURES,z1_eg1,NUM_OF_HID_NODES,syn0);
 /*compute a1*/
 vector_sigmoid(z1_eg1,a1_eg1,NUM_OF_FEATURES);
 /*compute z2*/
 z2_eg1 =  multiple_inputs_single_output_nn(a1_eg1,syn1,NUM_OF_FEATURES);
  printf("z2_eg1 :  %f \n\r",z2_eg1);
 /*compute yhat*/
 yhat_eg1 = sigmoid(z2_eg1);
 printf("yhat_eg1 :  %f\n\r", yhat_eg1);
 
	 while(1){}
}