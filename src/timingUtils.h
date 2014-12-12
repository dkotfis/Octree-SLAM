#ifndef TIMING_UTILS_H_
#define TIMING_UTILS_H_

/* Function to mark the timer to start */
void startTiming();

/* Function to mark the timer to end and return elapsed time */
/* @return Time (in ms) since startTiming was called */
float stopTiming();

#endif // TIMING_UTILS_H_
