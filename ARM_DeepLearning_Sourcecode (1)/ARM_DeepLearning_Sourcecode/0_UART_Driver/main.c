#include "uart.h"


int main(void){

	USART2_Init();
	
	//test_setup();
	printf("Thus is a message from main.");
	while(1){}
		
}