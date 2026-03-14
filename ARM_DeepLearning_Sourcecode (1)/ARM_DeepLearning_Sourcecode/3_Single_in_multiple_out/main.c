#include "uart.h"
#include "simple_neural_networks.h"

#define Sad   0.9

#define TEMPERATURE_PREDICTION_IDX 	0
#define HUMIDITY_PREDICTION_IDX			1
#define AIR_QUALITY_PREDICTION_IDX  2

#define OUT_LEN		3

double predicted_results[3];

                     //temp   hum   air_q
double weights[3] = {  -20,   95, 201.0};



int main(void){

	 USART2_Init();
	
	  single_input_multiple_output_nn(Sad,weights,predicted_results,OUT_LEN);
	 
   printf("Predicted temperature is : %f\r\n", predicted_results[TEMPERATURE_PREDICTION_IDX]);
   printf("Predicted humidity is : %f\r\n", predicted_results[HUMIDITY_PREDICTION_IDX]);

	 printf("Predicted air quality is : %f\r\n", predicted_results[AIR_QUALITY_PREDICTION_IDX]);

	 while(1){}
}