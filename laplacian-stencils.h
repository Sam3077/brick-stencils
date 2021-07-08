#include <brick-cuda.h>
#include "vecscatter.h"
#include "brick.h"

#define N 256
#define OFF (GZ + PADDING)
#define STRIDE (N + 2 * (OFF))

#define TILE 8
// there should be exactly one brick of ghost-zone
#define GZ (TILE)
#define PADDING 8
#define GB (GZ / (TILE))

#define BLOCK (N / TILE)

#define NAIVE_BSTRIDE ((N + 2 * GZ) / (TILE))

#define VSVEC "CUDA"
#define FOLD 4,8

#define BRICK_SIZE TILE, TILE, TILE

#define BType Brick<Dim<BRICK_SIZE>, Dim<FOLD>>

__global__ void no_prof_single_thread_7pt(bElem *in, bElem *out) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    for (int k = OFF; k < N + OFF; k++) {
        for (int j = OFF; j < N + OFF; j++) {
            for (int i = OFF; i < N + OFF; i++) {
                out_sized[k][j][i] = in_sized[k][j][i] + 
                                        in_sized[k][j][i + 1] + in_sized[k][j][i - 1] +
                                        in_sized[k][j + 1][i] + in_sized[k][j - 1][i] +
                                        in_sized[k + 1][j][i] + in_sized[k - 1][j][i];
            }
        }
    }
}

__global__ void no_prof_single_thread_13pt(bElem *in, bElem *out) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    for (int k = OFF; k < N + OFF; k++) {
        for (int j = OFF; j < N + OFF; j++) {
            for (int i = OFF; i < N + OFF; i++) {
                out_sized[k][j][i] = in_sized[k][j][i] + 
                            in_sized[k][j][i + 1] + in_sized[k][j][i + 2] + in_sized[k][j][i - 2] + in_sized[k][j][i - 1] +
                            in_sized[k][j + 1][i] + in_sized[k][j + 2][i] + in_sized[k][j - 2][i] + in_sized[k][j - 1][i] +
                            in_sized[k + 1][j][i] + in_sized[k + 2][j][i] + in_sized[k - 2][j][i] + in_sized[k - 1][j][i];
            }
        }
    }
}

__global__ void no_prof_single_thread_49pt(bElem *in, bElem *out) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    for (int k = OFF; k < N + OFF; k++) {
        for (int j = OFF; j < N + OFF; j++) {
            for (int i = OFF; i < N + OFF; i++) {
                out_sized[k][j][i] = in_sized[k][j][i] +
                       in_sized[k][j][i + 1] + in_sized[k][j][i + 2] + in_sized[k][j][i + 3] + in_sized[k][j][i + 4] +
                       in_sized[k][j][i + 5] + in_sized[k][j][i + 6] + in_sized[k][j][i + 7] + in_sized[k][j][i + 8] +
                       in_sized[k][j][i - 1] + in_sized[k][j][i - 2] + in_sized[k][j][i - 3] + in_sized[k][j][i - 4] +
                       in_sized[k][j][i - 5] + in_sized[k][j][i - 6] + in_sized[k][j][i - 7] + in_sized[k][j][i - 8] +
                       in_sized[k][j + 1][i] + in_sized[k][j + 2][i] + in_sized[k][j + 3][i] + in_sized[k][j + 4][i] +
                       in_sized[k][j + 5][i] + in_sized[k][j + 6][i] + in_sized[k][j + 7][i] + in_sized[k][j + 8][i] +
                       in_sized[k][j - 1][i] + in_sized[k][j - 2][i] + in_sized[k][j - 3][i] + in_sized[k][j - 4][i] +
                       in_sized[k][j - 5][i] + in_sized[k][j - 6][i] + in_sized[k][j - 7][i] + in_sized[k][j - 8][i] +
                       in_sized[k + 1][j][i] + in_sized[k + 2][j][i] + in_sized[k + 3][j][i] + in_sized[k + 4][j][i] +
                       in_sized[k + 5][j][i] + in_sized[k + 6][j][i] + in_sized[k + 7][j][i] + in_sized[k + 8][j][i] +
                       in_sized[k - 1][j][i] + in_sized[k - 2][j][i] + in_sized[k - 3][j][i] + in_sized[k - 4][j][i] +
                       in_sized[k - 5][j][i] + in_sized[k - 6][j][i] + in_sized[k - 7][j][i] + in_sized[k - 8][j][i];
            }
        }
    }
}

__global__ void naive_7pt_sum(bElem *in, bElem *out) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    unsigned i = OFF + (blockIdx.x) * TILE + threadIdx.x;
    unsigned j = OFF + (blockIdx.y) * TILE + threadIdx.y;
    unsigned k = OFF + (blockIdx.z) * TILE + threadIdx.z;

    out_sized[k][j][i] = in_sized[k][j][i] + 
                in_sized[k][j][i + 1] + in_sized[k][j][i - 1] +
                in_sized[k][j + 1][i] + in_sized[k][j - 1][i] +
                in_sized[k + 1][j][i] + in_sized[k - 1][j][i];
}

__global__ void naive_13pt_sum(bElem *in, bElem *out) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    unsigned i = OFF + (blockIdx.x) * TILE + threadIdx.x;
    unsigned j = OFF + (blockIdx.y) * TILE + threadIdx.y;
    unsigned k = OFF + (blockIdx.z) * TILE + threadIdx.z;

    out_sized[k][j][i] = in_sized[k][j][i] + 
                in_sized[k][j][i + 1] + in_sized[k][j][i + 2] + in_sized[k][j][i - 2] + in_sized[k][j][i - 1] +
                in_sized[k][j + 1][i] + in_sized[k][j + 2][i] + in_sized[k][j - 2][i] + in_sized[k][j - 1][i] +
                in_sized[k + 1][j][i] + in_sized[k + 2][j][i] + in_sized[k - 2][j][i] + in_sized[k - 1][j][i];
}

__global__ void naive_49pt_sum(bElem *in, bElem *out) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    unsigned i = OFF + (blockIdx.x) * TILE + threadIdx.x;
    unsigned j = OFF + (blockIdx.y) * TILE + threadIdx.y;
    unsigned k = OFF + (blockIdx.z) * TILE + threadIdx.z;

    out_sized[k][j][i] = in_sized[k][j][i] +
                       in_sized[k][j][i + 1] + in_sized[k][j][i + 2] + in_sized[k][j][i + 3] + in_sized[k][j][i + 4] +
                       in_sized[k][j][i + 5] + in_sized[k][j][i + 6] + in_sized[k][j][i + 7] + in_sized[k][j][i + 8] +
                       in_sized[k][j][i - 1] + in_sized[k][j][i - 2] + in_sized[k][j][i - 3] + in_sized[k][j][i - 4] +
                       in_sized[k][j][i - 5] + in_sized[k][j][i - 6] + in_sized[k][j][i - 7] + in_sized[k][j][i - 8] +
                       in_sized[k][j + 1][i] + in_sized[k][j + 2][i] + in_sized[k][j + 3][i] + in_sized[k][j + 4][i] +
                       in_sized[k][j + 5][i] + in_sized[k][j + 6][i] + in_sized[k][j + 7][i] + in_sized[k][j + 8][i] +
                       in_sized[k][j - 1][i] + in_sized[k][j - 2][i] + in_sized[k][j - 3][i] + in_sized[k][j - 4][i] +
                       in_sized[k][j - 5][i] + in_sized[k][j - 6][i] + in_sized[k][j - 7][i] + in_sized[k][j - 8][i] +
                       in_sized[k + 1][j][i] + in_sized[k + 2][j][i] + in_sized[k + 3][j][i] + in_sized[k + 4][j][i] +
                       in_sized[k + 5][j][i] + in_sized[k + 6][j][i] + in_sized[k + 7][j][i] + in_sized[k + 8][j][i] +
                       in_sized[k - 1][j][i] + in_sized[k - 2][j][i] + in_sized[k - 3][j][i] + in_sized[k - 4][j][i] +
                       in_sized[k - 5][j][i] + in_sized[k - 6][j][i] + in_sized[k - 7][j][i] + in_sized[k - 8][j][i];
}

__global__ void naive_brick_7pt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB][blockIdx.y + GB][blockIdx.x + GB];
    unsigned i = threadIdx.x + (blockIdx.x) * TILE;
    unsigned j = threadIdx.y + (blockIdx.y) * TILE;
    unsigned k = threadIdx.z + (blockIdx.z) * TILE;
    auto brick = bIn[b];
    bOut[b][k][j][i] = brick[k][j][i] +
                       brick[k][j][i + 1] + brick[k][j][i - 1] + 
                       brick[k][j + 1][i] + brick[k][j - 1][i] + 
                       brick[k + 1][j][i] + brick[k - 1][j][i];
}

__global__ void naive_brick_13pt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB][blockIdx.y + GB][blockIdx.x + GB];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bOut[b][k][j][i] = bIn[b][k][j][i] +
                       bIn[b][k][j][i + 1] + bIn[b][k][j][i + 2] + bIn[b][k][j][i - 2] + bIn[b][k][j][i - 1] + 
                       bIn[b][k][j + 1][i] + bIn[b][k][j + 2][i] + bIn[b][k][j - 2][i] + bIn[b][k][j - 1][i] + 
                       bIn[b][k + 1][j][i] + bIn[b][k + 2][j][i] + bIn[b][k - 2][j][i] + bIn[b][k - 1][j][i];
}

__global__ void naive_brick_49pt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB][blockIdx.y + GB][blockIdx.x + GB];
    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;
    bOut[b][k][j][i] = bIn[b][k][j][i] +
                       bIn[b][k][j][i + 1] + bIn[b][k][j][i + 2] + bIn[b][k][j][i + 3] + bIn[b][k][j][i + 4] +
                       bIn[b][k][j][i + 5] + bIn[b][k][j][i + 6] + bIn[b][k][j][i + 7] + bIn[b][k][j][i + 8] +
                       bIn[b][k][j][i - 1] + bIn[b][k][j][i - 2] + bIn[b][k][j][i - 3] + bIn[b][k][j][i - 4] +
                       bIn[b][k][j][i - 5] + bIn[b][k][j][i - 6] + bIn[b][k][j][i - 7] + bIn[b][k][j][i - 8] +
                       bIn[b][k][j + 1][i] + bIn[b][k][j + 2][i] + bIn[b][k][j + 3][i] + bIn[b][k][j + 4][i] +
                       bIn[b][k][j + 5][i] + bIn[b][k][j + 6][i] + bIn[b][k][j + 7][i] + bIn[b][k][j + 8][i] +
                       bIn[b][k][j - 1][i] + bIn[b][k][j - 2][i] + bIn[b][k][j - 3][i] + bIn[b][k][j - 4][i] +
                       bIn[b][k][j - 5][i] + bIn[b][k][j - 6][i] + bIn[b][k][j - 7][i] + bIn[b][k][j - 8][i] +
                       bIn[b][k + 1][j][i] + bIn[b][k + 2][j][i] + bIn[b][k + 3][j][i] + bIn[b][k + 4][j][i] +
                       bIn[b][k + 5][j][i] + bIn[b][k + 6][j][i] + bIn[b][k + 7][j][i] + bIn[b][k + 8][j][i] +
                       bIn[b][k - 1][j][i] + bIn[b][k - 2][j][i] + bIn[b][k - 3][j][i] + bIn[b][k - 4][j][i] +
                       bIn[b][k - 5][j][i] + bIn[b][k - 6][j][i] + bIn[b][k - 7][j][i] + bIn[b][k - 8][j][i];
}

template<unsigned factor, typename B>
__global__ void larger_brick_13pt(unsigned *grid, B bIn, B bOut) {
    auto g = (unsigned (*)[NAIVE_BSTRIDE / factor][NAIVE_BSTRIDE / factor]) grid;
    unsigned b = g[blockIdx.z / factor + GB][blockIdx.y / factor + GB][blockIdx.x / factor + GB];
    for (int i = 0; i < factor; i++) {
        for (int j = 0; j < factor; j++) {
            for (int k = 0; k < factor; k++) {
                unsigned x = i * factor + threadIdx.x;
                bOut[b][i][j][k] = bIn[b][i][j][k] +
                                   bIn[b][i][j][k + 1] + bIn[b][i][j][k + 2] + bIn[b][i][j][k - 1] + bIn[b][i][j][k - 2] +
                                   bIn[b][i][j + 1][k] + bIn[b][i][j + 2][k] + bIn[b][i][j - 1][k] + bIn[b][i][j - 2][k] +
                                   bIn[b][i + 1][j][k] + bIn[b][i + 2][j][k] + bIn[b][i - 1][j][k] + bIn[b][i - 2][j][k];
            }
        }
    }
}

__global__ void brick_gen(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[blockIdx.z + GB][blockIdx.y + GB][blockIdx.x + GB];
    brick("~/bricklib/stencils/7pt.py", "CUDA", (TILE, TILE, TILE), (FOLD), b);
}