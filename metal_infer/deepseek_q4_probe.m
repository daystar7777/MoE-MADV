#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIDDEN_DIM 4096
#define INTERMEDIATE_DIM 2048
#define NUM_EXPERTS 256
#define COMPONENT_SIZE 4456448
#define EXPERT_SIZE (COMPONENT_SIZE * 3)
#define GROUP_SIZE 32
#define MXFP4_BLOCK_SIZE 17

static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static float scale_lut[256];

static void init_scale_lut(void) {
    for (int i = 0; i < 256; ++i) {
        scale_lut[i] = ldexpf(1.0f, i - 128);
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--packed-dir PATH] [--layer N] [--expert N] [--shader PATH]\n",
            argv0);
}

static int parse_int_arg(const char *value, const char *name) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!value[0] || *end) {
        fprintf(stderr, "Invalid %s: %s\n", name, value);
        exit(2);
    }
    return (int)parsed;
}

static uint8_t *read_expert(const char *packed_dir, int layer, int expert) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, layer);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        return NULL;
    }

    uint8_t *expert_data = (uint8_t *)malloc(EXPERT_SIZE);
    if (!expert_data) {
        fprintf(stderr, "malloc expert buffer failed\n");
        close(fd);
        return NULL;
    }

    off_t offset = (off_t)expert * (off_t)EXPERT_SIZE;
    ssize_t n = pread(fd, expert_data, EXPERT_SIZE, offset);
    close(fd);

    if (n != EXPERT_SIZE) {
        fprintf(stderr, "short read from %s: %zd/%d\n", path, n, EXPERT_SIZE);
        free(expert_data);
        return NULL;
    }

    printf("Packed layer: %s\n", path);
    printf("Layer/expert: %d/%d\n", layer, expert);
    printf("Expert offset: %lld\n", (long long)offset);
    return expert_data;
}

static void make_input(float *x) {
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        x[i] = sinf((float)i * 0.013f) + 0.5f * cosf((float)i * 0.021f);
    }
}

static void cpu_mxfp4_matvec(const uint8_t *component, int rows, int cols, const float *x, float *out) {
    int groups = cols / GROUP_SIZE;
    int row_stride = groups * MXFP4_BLOCK_SIZE;
    for (int row = 0; row < rows; ++row) {
        const uint8_t *row_ptr = component + (size_t)row * row_stride;
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const uint8_t *block = row_ptr + g * MXFP4_BLOCK_SIZE;
            float d = scale_lut[block[0]];
            const uint8_t *q = block + 1;
            int x_base = g * GROUP_SIZE;
            for (int j = 0; j < 16; ++j) {
                uint8_t packed = q[j];
                acc += (float)kvalues_mxfp4[packed & 0x0F] * d * x[x_base + j];
                acc += (float)kvalues_mxfp4[packed >> 4] * d * x[x_base + 16 + j];
            }
        }
        out[row] = acc;
    }
}

static void cpu_silu_mul(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        float clipped = fmaxf(-80.0f, fminf(80.0f, g));
        float sigmoid = 1.0f / (1.0f + expf(-clipped));
        out[i] = g * sigmoid * up[i];
    }
}

static void describe(const char *name, const float *values, int n) {
    float min_v = values[0];
    float max_v = values[0];
    double sum = 0.0;
    double sq = 0.0;
    for (int i = 0; i < n; ++i) {
        float v = values[i];
        min_v = fminf(min_v, v);
        max_v = fmaxf(max_v, v);
        sum += v;
        sq += (double)v * (double)v;
    }
    printf("%s: shape=(%d,) min=%.6g max=%.6g mean=%.6g rms=%.6g\n",
           name, n, min_v, max_v, sum / n, sqrt(sq / n));
}

static void sha256_hex(const float *values, int n, char out[65]) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(values, (CC_LONG)((size_t)n * sizeof(float)), hash);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        snprintf(out + i * 2, 3, "%02x", hash[i]);
    }
    out[64] = '\0';
}

static BOOL encode_mxfp4_matvec(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    id<MTLBuffer> weights,
    id<MTLBuffer> x,
    id<MTLBuffer> out,
    uint32_t rows,
    uint32_t cols
) {
    [enc setComputePipelineState:pipe];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&rows length:sizeof(rows) atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    NSUInteger tpg = MIN((NSUInteger)128, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    return YES;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        const char *packed_dir = "../models/deepseek-v4-flash-4bit/packed_experts_q4";
        const char *shader_path = "deepseek_q4_probe.metal";
        int layer = 0;
        int expert = 0;

        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--packed-dir") == 0 && i + 1 < argc) {
                packed_dir = argv[++i];
            } else if (strcmp(argv[i], "--shader") == 0 && i + 1 < argc) {
                shader_path = argv[++i];
            } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
                layer = parse_int_arg(argv[++i], "--layer");
            } else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) {
                expert = parse_int_arg(argv[++i], "--expert");
            } else if (strcmp(argv[i], "--help") == 0) {
                usage(argv[0]);
                return 0;
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (layer < 0 || layer >= 43 || expert < 0 || expert >= NUM_EXPERTS) {
            fprintf(stderr, "layer must be 0..42 and expert must be 0..255\n");
            return 2;
        }

        init_scale_lut();
        uint8_t *expert_data = read_expert(packed_dir, layer, expert);
        if (!expert_data) return 1;

        const uint8_t *gate_data = expert_data;
        const uint8_t *up_data = expert_data + COMPONENT_SIZE;
        const uint8_t *down_data = expert_data + COMPONENT_SIZE * 2;

        float *x = (float *)calloc(HIDDEN_DIM, sizeof(float));
        float *gate_cpu = (float *)calloc(INTERMEDIATE_DIM, sizeof(float));
        float *up_cpu = (float *)calloc(INTERMEDIATE_DIM, sizeof(float));
        float *act_cpu = (float *)calloc(INTERMEDIATE_DIM, sizeof(float));
        float *out_cpu = (float *)calloc(HIDDEN_DIM, sizeof(float));
        if (!x || !gate_cpu || !up_cpu || !act_cpu || !out_cpu) {
            fprintf(stderr, "calloc failed\n");
            return 1;
        }

        make_input(x);

        CFAbsoluteTime cpu_t0 = CFAbsoluteTimeGetCurrent();
        cpu_mxfp4_matvec(gate_data, INTERMEDIATE_DIM, HIDDEN_DIM, x, gate_cpu);
        cpu_mxfp4_matvec(up_data, INTERMEDIATE_DIM, HIDDEN_DIM, x, up_cpu);
        cpu_silu_mul(gate_cpu, up_cpu, act_cpu, INTERMEDIATE_DIM);
        cpu_mxfp4_matvec(down_data, HIDDEN_DIM, INTERMEDIATE_DIM, act_cpu, out_cpu);
        CFAbsoluteTime cpu_t1 = CFAbsoluteTimeGetCurrent();

        describe("cpu gate", gate_cpu, INTERMEDIATE_DIM);
        describe("cpu up", up_cpu, INTERMEDIATE_DIM);
        describe("cpu act", act_cpu, INTERMEDIATE_DIM);
        describe("cpu out", out_cpu, HIDDEN_DIM);
        char cpu_hash[65];
        sha256_hex(out_cpu, HIDDEN_DIM, cpu_hash);
        printf("cpu out sha256: %s\n", cpu_hash);
        printf("cpu elapsed: %.3fs\n", cpu_t1 - cpu_t0);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device found\n");
            return 1;
        }

        NSError *error = nil;
        NSString *shaderFile = [NSString stringWithUTF8String:shader_path];
        NSString *source = [NSString stringWithContentsOfFile:shaderFile
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            fprintf(stderr, "Failed to read shader %s: %s\n", shader_path,
                    [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
        if (!lib) {
            fprintf(stderr, "Metal compile failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLFunction> matvecFn = [lib newFunctionWithName:@"mxfp4_matvec"];
        id<MTLFunction> siluFn = [lib newFunctionWithName:@"silu_mul"];
        id<MTLComputePipelineState> matvecPipe = [device newComputePipelineStateWithFunction:matvecFn error:&error];
        if (!matvecPipe) {
            fprintf(stderr, "matvec pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLComputePipelineState> siluPipe = [device newComputePipelineStateWithFunction:siluFn error:&error];
        if (!siluPipe) {
            fprintf(stderr, "silu pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLBuffer> gateBuf = [device newBufferWithBytes:gate_data length:COMPONENT_SIZE options:MTLResourceStorageModeShared];
        id<MTLBuffer> upBuf = [device newBufferWithBytes:up_data length:COMPONENT_SIZE options:MTLResourceStorageModeShared];
        id<MTLBuffer> downBuf = [device newBufferWithBytes:down_data length:COMPONENT_SIZE options:MTLResourceStorageModeShared];
        id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:HIDDEN_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gateOut = [device newBufferWithLength:INTERMEDIATE_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> upOut = [device newBufferWithLength:INTERMEDIATE_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> actOut = [device newBufferWithLength:INTERMEDIATE_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> finalOut = [device newBufferWithLength:HIDDEN_DIM * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        CFAbsoluteTime gpu_t0 = CFAbsoluteTimeGetCurrent();
        encode_mxfp4_matvec(enc, matvecPipe, gateBuf, xBuf, gateOut, INTERMEDIATE_DIM, HIDDEN_DIM);
        encode_mxfp4_matvec(enc, matvecPipe, upBuf, xBuf, upOut, INTERMEDIATE_DIM, HIDDEN_DIM);

        uint32_t n = INTERMEDIATE_DIM;
        [enc setComputePipelineState:siluPipe];
        [enc setBuffer:gateOut offset:0 atIndex:0];
        [enc setBuffer:upOut offset:0 atIndex:1];
        [enc setBuffer:actOut offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        NSUInteger tpg = MIN((NSUInteger)128, siluPipe.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(INTERMEDIATE_DIM, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

        encode_mxfp4_matvec(enc, matvecPipe, downBuf, actOut, finalOut, HIDDEN_DIM, INTERMEDIATE_DIM);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        CFAbsoluteTime gpu_t1 = CFAbsoluteTimeGetCurrent();

        if (cb.error) {
            fprintf(stderr, "Metal command failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            return 1;
        }

        float *out_gpu = (float *)[finalOut contents];
        describe("gpu out", out_gpu, HIDDEN_DIM);
        char gpu_hash[65];
        sha256_hex(out_gpu, HIDDEN_DIM, gpu_hash);
        printf("gpu out sha256: %s\n", gpu_hash);
        printf("gpu elapsed: %.3fs\n", gpu_t1 - gpu_t0);

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            float diff = fabsf(out_gpu[i] - out_cpu[i]);
            float rel = diff / fmaxf(1e-6f, fabsf(out_cpu[i]));
            if (diff > max_abs) {
                max_abs = diff;
                max_idx = i;
            }
            max_rel = fmaxf(max_rel, rel);
        }
        printf("compare: max_abs=%.9g at %d cpu=%.9g gpu=%.9g max_rel=%.9g\n",
               max_abs, max_idx, out_cpu[max_idx], out_gpu[max_idx], max_rel);

        free(expert_data);
        free(x);
        free(gate_cpu);
        free(up_cpu);
        free(act_cpu);
        free(out_cpu);

        return max_abs < 5e-5f ? 0 : 1;
    }
}
