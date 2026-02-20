#include "accelerator.h"
#include "hls_math.h" // Fixes the sqrt error

// --- Helper: Batch Norm + ReLU ---
float activate(float val, float gamma, float beta, float mean, float var) {
    #pragma HLS INLINE
    // Use hls::sqrt from hls_math.h
    float std = hls::sqrt(var + 0.001f);
    float norm = (val - mean) / std;
    float res = (norm * gamma) + beta;
    return (res > 0) ? res : 0.0f; // ReLU
}

// --- Layer 0: Standard Conv2D (3x3) ---
// --- Layer 0: Standard Conv2D (3x3) ---
void layer0_conv(stream_t &in, hls::stream<float> &out) {
    axis_t pkt;
    if (!in.empty()) {
        pkt = in.read();
        // REMOVED layer_0_bias. MobileNet handles this in BatchNorm.
        out.write(pkt.data);
    }
}

// --- Layer 3: Depthwise Conv ---
void layer3_depthwise(hls::stream<float> &in, hls::stream<float> &out) {
    // Placeholder logic
    if (!in.empty()) {
        float val = in.read();
        out.write(val * 1.5f); // Dummy operation
    }
}

// --- Layer 6: Pointwise Conv (1x1) ---
void layer6_pointwise(hls::stream<float> &in, stream_t &out) {
    // Placeholder logic
    if (!in.empty()) {
        float val = in.read();
        axis_t pkt_out;
        pkt_out.data = val * 0.5f; // Dummy operation
        pkt_out.last = 0; // Requires logic to set TLAST on the end of frame
        pkt_out.keep = -1;
        pkt_out.strb = -1;
        out.write(pkt_out);
    }
}

// --- TOP FUNCTION ---
void mobilenet_head(stream_t &in_stream, stream_t &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    static hls::stream<float> s1("s1");
    static hls::stream<float> s2("s2");

    #pragma HLS DATAFLOW
    
    // 1. Conv2D
    layer0_conv(in_stream, s1);
    
    // 2. Depthwise
    layer3_depthwise(s1, s2);
    
    // 3. Pointwise
    layer6_pointwise(s2, out_stream);
}
