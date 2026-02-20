#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "weights.h" // Your generated weights

// AXI Stream Side Channel (standard for video)
struct axis_t {
    float data;
    ap_int<1> last;
    ap_int<1> keep;
    ap_int<1> strb;
};

// Define input/output stream types
typedef hls::stream<axis_t> stream_t;

// Function Prototype
void mobilenet_head(stream_t &in_stream, stream_t &out_stream);

#endif