#include "uart.h"
#include "simple_neural_networks.h"

int32_t temperature[]={12,23,50,-10,16};
int32_t weight =-2;


int main(void){

	 USART2_Init();
	 
	 printf("The first predicted value is %d :\r\n",single_in_single_out_nn(temperature[0],weight));
	 printf("The second predicted value is %d :\r\n",single_in_single_out_nn(temperature[1],weight));
	 printf("The third predicted value is %d :\r\n",single_in_single_out_nn(temperature[2],weight));

	
	 while(1){}
}