#include "uart.h"
#include "simple_neural_networks.h"


#define  SAD_PREDICTION_IDX		 0
#define  SICK_PREDCITION_IDX   1
#define  ACTIVE_PREDICTION_IDX	2

#define OUT_LEN		3
#define IN_LEN		3
#define HID_LEN   3


double predicted_output[OUT_LEN];

                                    //temp hum  air_q

double input_to_hidden_weights[HID_LEN][IN_LEN] ={  {-2.0,9.5,2.01},       //hid[0]
																										{-0.8,7.2,6.3 },        //hid[1]
																										{-0.5,0.45,0.9}  };     //hid[2]

																													//hid[0] hid[1] hid[2] 
double			hidden_to_output_weights[OUT_LEN][HID_LEN] ={ {-1.0,  1.15,   0.11},   //sad?
																													{-0.18,  0.15, -0.01},   //sick?
																													{0.25, -0.25, -0.1}      //active?
																												};																						

																		
														//  temp  hum air_q
																		
double input_vector[IN_LEN] = { 30.0,87.0,110.0};

double expected_values[OUT_LEN] = {600,10,-90};  //i.e y values

int main(void){

	 USART2_Init();
	
    hidden_nn(input_vector,IN_LEN,HID_LEN,input_to_hidden_weights,OUT_LEN,hidden_to_output_weights,predicted_output)	;
    
	  printf("Sad prediction  : %f\r\n", predicted_output[SAD_PREDICTION_IDX]);
	  printf("Sad error  :  %f\r\n",find_error_smpl(predicted_output[SAD_PREDICTION_IDX],expected_values[SAD_PREDICTION_IDX]));

    printf("Sick prediction  : %f\r\n", predicted_output[SICK_PREDCITION_IDX]);
		printf("Sick error  :  %f\r\n",find_error_smpl(predicted_output[SICK_PREDCITION_IDX],expected_values[SICK_PREDCITION_IDX]));

    printf("Active prediction  : %f\r\n", predicted_output[ACTIVE_PREDICTION_IDX]);
		printf("Active error  :  %f\r\n",find_error_smpl(predicted_output[ACTIVE_PREDICTION_IDX],expected_values[ACTIVE_PREDICTION_IDX]));


	 while(1){}
}