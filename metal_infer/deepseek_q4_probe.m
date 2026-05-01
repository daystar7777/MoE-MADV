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
#define MAX_ACTIVE_EXPERTS 16

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
            "Usage: %s [--packed-dir PATH] [--layer N] [--expert N | --experts A,B,C] [--weights W,...] [--shader PATH]\n",
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

static int parse_list_ints(const char *value, int *out, int max_count, const char *name) {
    char *copy = strdup(value);
    if (!copy) {
        fprintf(stderr, "strdup failed\n");
        exit(1);
    }
    int count = 0;
    char *token = strtok(copy, ",");
    while (token) {
        if (count >= max_count) {
            fprintf(stderr, "%s supports at most %d entries\n", name, max_count);
            exit(2);
        }
        out[count++] = parse_int_arg(token, name);
        token = strtok(NULL, ",");
    }
    free(copy);
    return count;
}

static int parse_list_floats(const char *value, float *out, int max_count, const char *name) {
    char *copy = strdup(value);
    if (!copy) {
        fprintf(stderr, "strdup failed\n");
        exit(1);
    }
    int count = 0;
    char *token = strtok(copy, ",");
    while (token) {
        if (count >= max_count) {
            fprintf(stderr, "%s supports at most %d entries\n", name, max_count);
            exit(2);
        }
        char *end = NULL;
        float parsed = strtof(token, &end);
        if (!token[0] || *end) {
            fprintf(stderr, "Invalid %s: %s\n", name, token);
            exit(2);
        }
        out[count++] = parsed;
        token = strtok(NULL, ",");
    }
    free(copy);
    return count;
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
        int experts[MAX_ACTIVE_EXPERTS] = {0};
        float weights[MAX_ACTIVE_EXPERTS] = {1.0f};
        int num_active = 1;
        int weights_count = 0;

        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--packed-dir") == 0 && i + 1 < argc) {
                packed_dir = argv[++i];
            } else if (strcmp(argv[i], "--shader") == 0 && i + 1 < argc) {
                shader_path = argv[++i];
            } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
                layer = parse_int_arg(argv[++i], "--layer");
            } else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) {
                experts[0] = parse_int_arg(argv[++i], "--expert");
                num_active = 1;
            } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
                num_active = parse_list_ints(argv[++i], experts, MAX_ACTIVE_EXPERTS, "--experts");
            } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
                weights_count = parse_list_floats(argv[++i], weights, MAX_ACTIVE_EXPERTS, "--weights");
            } else if (strcmp(argv[i], "--help") == 0) {
                usage(argv[0]);
                return 0;
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (weights_count == 0) {
            for (int i = 0; i < num_active; ++i) {
                weights[i] = 1.0f / (float)num_active;
            }
        } else if (weights_count != num_active) {
            fprintf(stderr, "--weights count must match active experts\n");
            return 2;
        }

        if (layer < 0 || layer >= 43) {
            fprintf(stderr, "layer must be 0..42\n");
            return 2;
        }
        for (int i = 0; i < num_active; ++i) {
            if (experts[i] < 0 || experts[i] >= NUM_EXPERTS) {
                fprintf(stderr, "expert must be 0..255: %d\n", experts[i]);
                return 2;
            }
        }

        init_scale_lut();
        uint8_t *expert_data[MAX_ACTIVE_EXPERTS] = {0};
        for (int i = 0; i < num_active; ++i) {
            expert_data[i] = read_expert(packed_dir, layer, experts[i]);
            if (!expert_data[i]) return 1;
        }
        printf("Active experts:");
        for (int i = 0; i < num_active; ++i) {
            printf(" %d@%.6g", experts[i], weights[i]);
        }
        printf("\n");

        float *x = (float *)calloc(HIDDEN_DIM, sizeof(float));
        float *gate_cpu = (float *)calloc(INTERMEDIATE_DIM, sizeof(float));
        float *up_cpu = (float *)calloc(INTERMEDIATE_DIM, sizeof(float));
        float *act_cpu = (float *)calloc(INTERMEDIATE_DIM, sizeof(float));
        float *tmp_cpu = (float *)calloc(HIDDEN_DIM, sizeof(float));
        float *out_cpu = (float *)calloc(HIDDEN_DIM, sizeof(float));
        if (!x || !gate_cpu || !up_cpu || !act_cpu || !tmp_cpu || !out_cpu) {
            fprintf(stderr, "calloc failed\n");
            return 1;
        }

        make_input(x);

        CFAbsoluteTime cpu_t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < num_active; ++i) {
            const uint8_t *gate_data = expert_data[i];
            const uint8_t *up_data = expert_data[i] + COMPONENT_SIZE;
            const uint8_t *down_data = expert_data[i] + COMPONENT_SIZE * 2;
            cpu_mxfp4_matvec(gate_data, INTERMEDIATE_DIM, HIDDEN_DIM, x, gate_cpu);
            cpu_mxfp4_matvec(up_data, INTERMEDIATE_DIM, HIDDEN_DIM, x, up_cpu);
            cpu_silu_mul(gate_cpu, up_cpu, act_cpu, INTERMEDIATE_DIM);
            cpu_mxfp4_matvec(down_data, HIDDEN_DIM, INTERMEDIATE_DIM, act_cpu, tmp_cpu);
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                out_cpu[j] += weights[i] * tmp_cpu[j];
            }
        }
        CFAbsoluteTime cpu_t1 = CFAbsoluteTimeGetCurrent();

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
        id<MTLFunction> accumFn = [lib newFunctionWithName:@"accumulate_weighted"];
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
        id<MTLComputePipelineState> accumPipe = [device newComputePipelineStateWithFunction:accumFn error:&error];
        if (!accumPipe) {
            fprintf(stderr, "accumulate pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLBuffer> gateBufs[MAX_ACTIVE_EXPERTS];
        id<MTLBuffer> upBufs[MAX_ACTIVE_EXPERTS];
        id<MTLBuffer> downBufs[MAX_ACTIVE_EXPERTS];
        for (int i = 0; i < num_active; ++i) {
            gateBufs[i] = [device newBufferWithBytes:expert_data[i] length:COMPONENT_SIZE options:MTLResourceStorageModeShared];
            upBufs[i] = [device newBufferWithBytes:expert_data[i] + COMPONENT_SIZE length:COMPONENT_SIZE options:MTLResourceStorageModeShared];
            downBufs[i] = [device newBufferWithBytes:expert_data[i] + COMPONENT_SIZE * 2 length:COMPONENT_SIZE options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:HIDDEN_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> gateOut = [device newBufferWithLength:INTERMEDIATE_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> upOut = [device newBufferWithLength:INTERMEDIATE_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> actOut = [device newBufferWithLength:INTERMEDIATE_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> tmpOut = [device newBufferWithLength:HIDDEN_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> finalOut = [device newBufferWithLength:HIDDEN_DIM * sizeof(float) options:MTLResourceStorageModeShared];
        memset([finalOut contents], 0, HIDDEN_DIM * sizeof(float));

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        CFAbsoluteTime gpu_t0 = CFAbsoluteTimeGetCurrent();
        for (int i = 0; i < num_active; ++i) {
            encode_mxfp4_matvec(enc, matvecPipe, gateBufs[i], xBuf, gateOut, INTERMEDIATE_DIM, HIDDEN_DIM);
            encode_mxfp4_matvec(enc, matvecPipe, upBufs[i], xBuf, upOut, INTERMEDIATE_DIM, HIDDEN_DIM);

            uint32_t n = INTERMEDIATE_DIM;
            [enc setComputePipelineState:siluPipe];
            [enc setBuffer:gateOut offset:0 atIndex:0];
            [enc setBuffer:upOut offset:0 atIndex:1];
            [enc setBuffer:actOut offset:0 atIndex:2];
            [enc setBytes:&n length:sizeof(n) atIndex:3];
            NSUInteger tpg = MIN((NSUInteger)128, siluPipe.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(INTERMEDIATE_DIM, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            encode_mxfp4_matvec(enc, matvecPipe, downBufs[i], actOut, tmpOut, HIDDEN_DIM, INTERMEDIATE_DIM);

            uint32_t out_n = HIDDEN_DIM;
            float weight = weights[i];
            [enc setComputePipelineState:accumPipe];
            [enc setBuffer:tmpOut offset:0 atIndex:0];
            [enc setBuffer:finalOut offset:0 atIndex:1];
            [enc setBytes:&weight length:sizeof(weight) atIndex:2];
            [enc setBytes:&out_n length:sizeof(out_n) atIndex:3];
            NSUInteger atpg = MIN((NSUInteger)128, accumPipe.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(HIDDEN_DIM, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
        }
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

        for (int i = 0; i < num_active; ++i) {
            free(expert_data[i]);
        }
        free(x);
        free(gate_cpu);
        free(up_cpu);
        free(act_cpu);
        free(tmp_cpu);
        free(out_cpu);

        return max_abs < 5e-5f ? 0 : 1;
    }
}
