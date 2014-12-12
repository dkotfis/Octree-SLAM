
// CUDA Dependencies
#include <cuda.h>

// Project Dependencies
#include "timingUtils.h"

//Global data
cudaEvent_t beginEvent, endEvent;

void startTiming() {
	//Add timing options
	cudaEventCreate( &beginEvent );
	cudaEventCreate( &endEvent );

	//Execute the naive prefix sum and compute the time (in milliseconds)
	cudaEventRecord(beginEvent, 0);
}

float stopTiming() {
	float time;

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&time, beginEvent, endEvent);

	//Cleanup timers
	cudaEventDestroy(beginEvent);
	cudaEventDestroy(endEvent);

	return time;
}