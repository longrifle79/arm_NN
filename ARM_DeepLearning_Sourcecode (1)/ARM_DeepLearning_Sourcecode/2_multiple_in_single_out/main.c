#include "uart.h"
#include "simple_neural_networks.h"

#define NUM_OF_INPUTS 	3

double temperature[5] = {12,23,50,-10,16};
double humidity[5] =    {60,67,50,65,63};
double air_quality[5] = {60,47,167,187,94};

double weight[3] = {-2,2,1};


int main(void){

	 double training_eg1[3] ={temperature[0],humidity[0], air_quality[0]};
	 USART2_Init();
	 printf("Prediction from first training example is  : %f\r\n ",multiple_inputs_single_output_nn(training_eg1,weight,NUM_OF_INPUTS));
	 


	
	 while(1){}
}