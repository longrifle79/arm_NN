#ifndef __SIMPLE_NEURAL_NETWORK
#define __SIMPLE_NEURAL_NETWORK

#include <stdint.h>

double single_in_single_out_nn(double input, double weight);
double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN);

#endif