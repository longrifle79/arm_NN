#include "simple_neural_networks.h"


#define NUM_OF_FEATURES   2  // n values
#define NUM_OF_EXAMPLES		3  // m values

/*Hours of workout data*/
double x1[NUM_OF_EXAMPLES]  = {2,5,1};
double _x1[NUM_OF_EXAMPLES];

/*Hours of rest data*/
double x2[NUM_OF_EXAMPLES]  = {8,5,8};
double _x2[NUM_OF_EXAMPLES];


/*Muscle gain data*/
double y[NUM_OF_EXAMPLES]  = {200,90,190};
double _y[NUM_OF_EXAMPLES];

int main(void){

	 USART2_Init();
	
	 normalize_data(x1,_x1,NUM_OF_EXAMPLES);
   normalize_data(x2,_x2,NUM_OF_EXAMPLES);
	 normalize_data(y,_y,NUM_OF_EXAMPLES);
  
	 printf("Raw x1 data : \n\r");
	 for(int i =0;i<NUM_OF_EXAMPLES;i++){
	  printf(" %f ",x1[i]);
	 }
	 		printf("\n\r");

	 	 printf("Normalized x1 data : \n\r");
	 for(int i =0;i<NUM_OF_EXAMPLES;i++){
	  printf(" %f ",_x1[i]);
	 }

	
	 	 printf("Raw x2 data : \n\r");
	 for(int i =0;i<NUM_OF_EXAMPLES;i++){
	  printf(" %f ",x2[i]);
	 }
			printf("\n\r");

	 	 printf("Normalized x2 data : \n\r");
	 for(int i =0;i<NUM_OF_EXAMPLES;i++){
	  printf(" %f ",_x2[i]);
	 }
		printf("\n\r");

	 
	 	 printf("Raw y data : \n\r");
	 for(int i =0;i<NUM_OF_EXAMPLES;i++){
	  printf(" %f ",y[i]);
	 }
	 		printf("\n\r");

	 	 	 printf("Normalized y data : \n\r");
	 for(int i =0;i<NUM_OF_EXAMPLES;i++){
	  printf(" %f ",_y[i]);
	 }
	 		printf("\n\r");

	 while(1){}
}