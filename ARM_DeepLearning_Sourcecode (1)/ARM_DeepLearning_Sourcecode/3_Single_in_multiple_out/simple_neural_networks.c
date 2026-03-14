#include "simple_neural_networks.h"

double single_in_single_out_nn(double  input, double weight)
{
   	return (input * weight);
}


double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN){

	double output;
	
	for(int i=0;i<INPUT_LEN;i++){
	  output += input[i]*weight[i];
	}
 return output;
}

double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN){
  
	double predicted_value;
	
	
	predicted_value =  weighted_sum(input,weight,INPUT_LEN);
	
	return predicted_value;
}


void elementwise_multiple( double input_scalar,
													 double *weight_vector,
													 double *output_vector,
													 double VECTOR_LEN)
{
	for(int i =0;i<VECTOR_LEN;i++){
	   output_vector[i] =  input_scalar *weight_vector[i];
	}
	
}

void single_input_multiple_output_nn(	double input_scalar,
																			double *weight_vector,
																			double *output_vector,
																			double VECTOR_LEN){
  elementwise_multiple(input_scalar, weight_vector,output_vector,VECTOR_LEN);																			
}