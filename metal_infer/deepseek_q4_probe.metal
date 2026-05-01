#include <metal_stdlib>
using namespace metal;

constant float kvalues_mxfp4[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

static inline float e8m0_to_fp32_half(uchar x) {
    uint bits = x < 2 ? (0x00200000u << x) : ((uint(x) - 1u) << 23);
    return as_type<float>(bits);
}

kernel void mxfp4_matvec(
    device const uchar * blocks [[buffer(0)]],
    device const float * x      [[buffer(1)]],
    device float * out          [[buffer(2)]],
    constant uint & rows        [[buffer(3)]],
    constant uint & cols        [[buffer(4)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= rows) {
        return;
    }

    const uint groups = cols / 32u;
    const uint row_base = tid * groups * 17u;
    float acc = 0.0f;

    for (uint g = 0; g < groups; ++g) {
        const uint block_base = row_base + g * 17u;
        const float d = e8m0_to_fp32_half(blocks[block_base]);
        const uint x_base = g * 32u;

        for (uint j = 0; j < 16u; ++j) {
            const uchar q = blocks[block_base + 1u + j];
            acc += kvalues_mxfp4[q & 0x0Fu] * d * x[x_base + j];
            acc += kvalues_mxfp4[q >> 4] * d * x[x_base + 16u + j];
        }
    }

    out[tid] = acc;
}

kernel void silu_mul(
    device const float * gate [[buffer(0)]],
    device const float * up   [[buffer(1)]],
    device float * out        [[buffer(2)]],
    constant uint & n         [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    const float g = gate[tid];
    const float clipped = clamp(g, -80.0f, 80.0f);
    const float sigmoid = 1.0f / (1.0f + exp(-clipped));
    out[tid] = g * sigmoid * up[tid];
}
