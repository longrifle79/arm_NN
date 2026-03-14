#include "simple_neural_networks.h"


#define NUM_OF_FEATURES   2  // n values
#define NUM_OF_EXAMPLES		3  // m values
#define NUM_OF_HID_NODES	3
#define NUM_OF_OUT_NODES	1


/*Hours of workout data*/
double x1[NUM_OF_EXAMPLES]  = {2,5,1};
double _x1[NUM_OF_EXAMPLES];

/*Hours of rest data*/
double x2[NUM_OF_EXAMPLES]  = {8,5,8};
double _x2[NUM_OF_EXAMPLES];


/*Muscle gain data*/
double y[NUM_OF_EXAMPLES]  = {200,90,190};
double _y[NUM_OF_EXAMPLES];

/*Input layer to hidden layer weight matrix*/
double syn0[NUM_OF_HID_NODES][NUM_OF_FEATURES];

/*Hidden layer to output layer weight matrix*/
double syn1[NUM_OF_OUT_NODES][NUM_OF_HID_NODES];

int main(void){

	 USART2_Init();
	 
	weights_random_initialization(NUM_OF_HID_NODES,NUM_OF_FEATURES,syn0);
	weights_random_initialization(NUM_OF_OUT_NODES,NUM_OF_HID_NODES,syn1);

	 /*Synase 0 weights*/
	for(int i =0;i< NUM_OF_HID_NODES;i++){
	  for(int j=0;j<NUM_OF_FEATURES;j++){
		
		  printf("  %f  ",syn0[i][j]);
		}
		printf("\n\r");
		printf("\n\r");
	}

	 printf("\n\r");
		printf("\n\r");
	
		 /*Synase 1 weights*/
	for(int i =0;i< NUM_OF_OUT_NODES;i++){
	  for(int j=0;j<NUM_OF_HID_NODES;j++){
		
		  printf("  %f  ",syn1[i][j]);
		}
		printf("\n\r");
		printf("\n\r");
	}
	
	 while(1){}
}