#include "simple_neural_networks.h"
#include "math.h"
#include <stdlib.h>

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
																			

void matrix_vector_multiplication(double * input_vector,
																	uint32_t INPUT_LEN,
																	double * output_vector,
																	uint32_t OUTPUT_LEN,
																	double weights_matrix[OUTPUT_LEN][INPUT_LEN])
{
	
	for(int k=0;k<OUTPUT_LEN;k++){
	   for(int i =0;i<INPUT_LEN;i++){
		   output_vector[k] += input_vector[i] *weights_matrix[k][i];
		 }
	}
	
}

void multiple_inputs_multiple_outputs_nn(double * input_vector,
																	uint32_t INPUT_LEN,
																	double * output_vector,
																	uint32_t OUTPUT_LEN,
																	double weights_matrix[OUTPUT_LEN][INPUT_LEN])
{
	matrix_vector_multiplication(input_vector,INPUT_LEN,output_vector,OUTPUT_LEN,weights_matrix);														
}
	


void hidden_nn( double *input_vector,
								uint32_t INPUT_LEN,
								uint32_t HIDDEN_LEN,
                double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
								uint32_t OUTPUT_LEN,
								double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN],
								double *output_vector
							)
{
	double hidden_pred_vector[HIDDEN_LEN];
	matrix_vector_multiplication(input_vector,INPUT_LEN,hidden_pred_vector,OUTPUT_LEN,in_to_hid_weights);
  matrix_vector_multiplication(hidden_pred_vector,HIDDEN_LEN,output_vector,OUTPUT_LEN,hid_to_out_weights);
}


double  find_error_smpl( double yhat, double y){
   
	return powf((yhat -y),2);
}


double find_error_(double input, double weight, double expected_value){
	
   return powf(((input*weight) - expected_value),2);
}

void brute_force_learning( double input,
													 double weight,
													 double expected_value,
													 double step_amount,
													 uint32_t itr)
{
   double prediction,error;
   double up_prediction,down_prediction,up_error, down_error;
	
	 for(int i=0;i<itr;i++){
	   
		 prediction  = input * weight;
		 error = powf((prediction- expected_value),2);
		 
		 printf("Error : %f    Prediction : %f \r\n",error,prediction);
		 
		 up_prediction =  input * (weight +step_amount);
		 up_error      =   powf((up_prediction- expected_value),2);
		 
		 down_prediction =  input * (weight - step_amount);
		 down_error      =  powf((down_prediction - expected_value),2);
		 
		 if(down_error <  up_error)
			   weight  = weight - step_amount;
		 if(down_error >  up_error)
			   weight = weight  + step_amount;
		 

	   
	 }

}
	

void normalize_data(double *input_vector, double * output_vector,uint32_t LEN){
 /*Find max*/
	double max = input_vector[0];
	for(int i =0;i<LEN;i++){
	  if(input_vector[i] >  max){
		  max = input_vector[i];
		}
	}
	
	/*Divide each elem by max*/
	for(int i=0;i<LEN;i++){
	  output_vector[i] = input_vector[i]/max;
	}

}



void  weights_random_initialization( uint32_t HIDDEN_LEN,
																		 uint32_t INPUT_LEN,
																		 double weight_matrix[HIDDEN_LEN][INPUT_LEN]){
 
	double d_rand;
																		 
	/*Seed random number generator*/
	srand(1);
																			 
	for(int i =0;i<HIDDEN_LEN ;i++){
	
	  for(int j=0; j<INPUT_LEN;j++){
		
		    /*Generate random numbers between 0 and 1*/
			d_rand = (rand() %10);
			d_rand /=10;
      
     weight_matrix[i][j] = d_rand;			
		}
	}		
	 }

																		 
 double  sigmoid(double x){
	 
	 double result =  1/(1+exp(-x));
 
 }
 
 
 void vector_sigmoid(double * input_vector, double * output_vector, uint32_t LEN)
 {
	 
	  for(int i =0;i<LEN;i++){
		  output_vector[i] =  sigmoid(input_vector[i]);
		}
 }


void normalize_data_2d(uint32_t ROW,
											 uint32_t COL,
												double input_matrix[ROW][COL],
												double output_matrix[ROW][COL]){
	
double max =  -99999999;
	for(int i =0;i<ROW;i++){
	  for(int j =0;j<COL;j++){
		  if(input_matrix[i][j] >max){
			  max = input_matrix[i][j];
			}
		}
	}

 for(int i=0;i<ROW;i++){
   for(int j=0;j<COL;j++){
	    output_matrix[i][j] =  input_matrix[i][j]/max;
	 }
 }
}													


void weights_random_initialization_1d( double * output_vector,uint32_t LEN){
 srand(1);
	double d_rand;
	for(int j=0;j<LEN;j++){
	  d_rand = (rand() %10);
		d_rand /=10;
		output_vector[j] =  d_rand;
	}
}
