#include <ap_int.h>

#define H 32
#define W 32
#define K 3

extern "C" {

void conv2d_hls(float *in,
                float *kernel,
                float *out)
{
#pragma HLS INTERFACE m_axi port=in     offset=slave bundle=gmem depth=1024
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=gmem depth=9
#pragma HLS INTERFACE m_axi port=out    offset=slave bundle=gmem depth=900

#pragma HLS INTERFACE s_axilite port=in     bundle=control
#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=out    bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local BRAM buffers
    float in_local[H][W];
    float k_local[K][K];

#pragma HLS ARRAY_PARTITION variable=k_local complete dim=0

    // Load input image
    for(int i=0;i<H;i++){
        for(int j=0;j<W;j++){
#pragma HLS PIPELINE II=1
            in_local[i][j] = in[i*W + j];
        }
    }

    // Load kernel
    for(int i=0;i<K;i++){
        for(int j=0;j<K;j++){
#pragma HLS PIPELINE II=1
            k_local[i][j] = kernel[i*K + j];
        }
    }

    // Convolution
    for(int i=0;i<H-2;i++){
        for(int j=0;j<W-2;j++){
#pragma HLS PIPELINE II=1

            float sum = 0;

            for(int ki=0;ki<K;ki++){
#pragma HLS UNROLL
                for(int kj=0;kj<K;kj++){
#pragma HLS UNROLL
                    sum += in_local[i+ki][j+kj] * k_local[ki][kj];
                }
            }

            out[i*(W-2) + j] = sum;
        }
    }
}

}
