// flowIO.h

// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#ifndef UNKNOWN_FLOW_THRESH
#define UNKNOWN_FLOW_THRESH 1e9
#endif

// value to use to represent unknown flow
#ifndef UNKNOWN_FLOW
#define UNKNOWN_FLOW 1e10
#endif

// return whether flow vector is unknown
bool unknown_flow(float u, float v);
bool unknown_flow(float *f);

// read the size of a flow file 
void ReadFlowFileSize(int& height, int& width, const char* filename);

// read a flow file into 2-band image
void ReadFlowFile(CFloatImage& img, const char* filename);

// write a 2-band image into flow file 
void WriteFlowFile(CFloatImage img, const char* filename);


