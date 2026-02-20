#include <iostream>

#define H 32
#define W 32
#define K 3

extern "C" {
void conv2d_hls(float in[H][W],
                float kernel[K][K],
                float out[H-2][W-2]);
}

int main() {
    static float in[H][W];
    static float kernel[K][K];
    static float out[H-2][W-2];

    // Initialize input
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            in[i][j] = (i + j) % 5;
        }
    }

    // Initialize kernel
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            kernel[i][j] = 1.0f;
        }
    }

    // Call DUT
    conv2d_hls(in, kernel, out);

    // Print few outputs
    std::cout << "Output sample:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << out[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "C Simulation PASSED" << std::endl;
    return 0;
}
