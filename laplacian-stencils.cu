#include <iostream>

#include "laplacian-stencils.h"
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include <omp.h>

__host__ int runNaiveTest(bool verify, bElem* arr_a);
__host__ int runLargeBrickTest(bElem *arr_a);

#define TO_TIME 3
template<typename T>
double time_func(T func) {
  int it = 1;
  func(); // Warm up
  double st = omp_get_wtime();
  double ed = st;
  while (ed < st + TO_TIME) {
    for (int i = 0; i < it; ++i)
      func();
    it <<= 1;
    ed = omp_get_wtime();
  }
  return (ed - st) / (it - 1);
}

#define CU_ITER 100
template<typename T>
double cutime_func(T func) {
  func(); // Warm up
  cudaEvent_t start, stop;
  float elapsed;
  cudaDeviceSynchronize();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < CU_ITER; ++i)
    func();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  return elapsed / CU_ITER / 1000;
}

int main(void) {
    bElem *arr_a = randomArray({STRIDE, STRIDE, STRIDE});
    int i = runNaiveTest(true, arr_a);
    i |= runLargeBrickTest(arr_a);
    return i;
}

__host__ int runLargeBrickTest(bElem *arr_a) {
    const int brick_stride = ((N + 2 * GZ) / (2 * TILE));
    unsigned *bgrid;
    auto binfo = init_grid<3>(bgrid, {brick_stride, brick_stride, brick_stride});
    unsigned *device_bgrid;
    {
        unsigned grid_size = (brick_stride * brick_stride * brick_stride) * sizeof(unsigned);
        cudaMalloc(&device_bgrid, grid_size);
        cudaMemcpy(device_bgrid, bgrid, grid_size, cudaMemcpyHostToDevice);
    }
    BrickInfo<3> *device_binfo = movBrickInfoDeep(binfo, cudaMemcpyHostToDevice);

    auto brick_size = cal_size<TILE * 2, TILE * 2, TILE * 2>::value;
    auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);

    Brick<Dim<TILE * 2, TILE * 2, TILE * 2>, Dim<FOLD>> bIn(&binfo, brick_storage, 0);
    Brick<Dim<TILE * 2, TILE * 2, TILE * 2>, Dim<FOLD>> bOut(&binfo, brick_storage, brick_size);

    copyToBrick<3>({N + 2 * GZ, N + 2 * GZ, N + 2 * GZ}, {PADDING, PADDING, PADDING}, {0, 0, 0}, arr_a, bgrid, bIn);
    BrickStorage device_bstorage = movBrickStorage(brick_storage, cudaMemcpyHostToDevice);

    bIn = Brick<Dim<TILE * 2, TILE * 2, TILE * 2>, Dim<FOLD>>(device_binfo, device_bstorage, 0);
    bOut = Brick<Dim<TILE * 2, TILE * 2, TILE * 2>, Dim<FOLD>>(device_binfo, device_bstorage, brick_size);
    dim3 blocks(BLOCK, BLOCK, BLOCK);
    dim3 threads(TILE, TILE, TILE);

    larger_brick_13pt<2><<<blocks, threads>>>(device_bgrid, bIn, bOut);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::cout << "Failed to run larger kernel. " << cudaGetErrorString(e) << std::endl;
    }
    return 0;
}

__host__ int runNaiveTest(bool verify, bElem *arr_a) {
    const int brick_stride = ((N + 2 * GZ) / (TILE));
    // intialize and move the brick grid
    unsigned *bgrid;
    auto binfo = init_grid<3>(bgrid, {brick_stride, brick_stride, brick_stride});
    unsigned *device_bgrid;
    {
        unsigned grid_size = (brick_stride * brick_stride * brick_stride) * sizeof(unsigned);
        cudaMalloc(&device_bgrid, grid_size);
        cudaMemcpy(device_bgrid, bgrid, grid_size, cudaMemcpyHostToDevice);
    }

    BrickInfo<3> *device_binfo = movBrickInfoDeep(binfo, cudaMemcpyHostToDevice);

    bElem *A;
    bElem *B;
    cudaMalloc(&A, STRIDE * STRIDE * STRIDE * sizeof(bElem));
    cudaMalloc(&B, (N + 2 * (OFF)) * STRIDE * STRIDE * sizeof(bElem));
    cudaError_t e = cudaMemcpy(A, arr_a, STRIDE * STRIDE * STRIDE * sizeof(bElem), cudaMemcpyHostToDevice);
    if (e != cudaSuccess)
    {
        std::cout << "Failed to copy A. " << cudaGetErrorString(e) << std::endl;
    }

    auto brick_size = cal_size<BRICK_SIZE>::value;
    // double number of bricks for a and b
    auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);

    BType bIn(&binfo, brick_storage, 0);
    BType bOut(&binfo, brick_storage, brick_size);

    copyToBrick<3>({N + 2 * GZ, N + 2 * GZ, N + 2 * GZ}, {PADDING, PADDING, PADDING}, {0, 0, 0}, arr_a, bgrid, bIn);

    BrickStorage device_bstorage = movBrickStorage(brick_storage, cudaMemcpyHostToDevice);
    bIn = BType(device_binfo, device_bstorage, 0);
    bOut = BType(device_binfo, device_bstorage, brick_size);

    dim3 blocks(BLOCK, BLOCK, BLOCK);
    dim3 threads(TILE, TILE, TILE);

    printf("Naive Array 7pt\n");
    naive_49pt_sum<<<blocks, threads>>>(A, B);

    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
    {
        std::cout << "Kernel execution failed. " << cudaGetErrorString(e) << std::endl;
    }

    printf("Naive Brick 7pt\n");
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
    brick_gen<<<blocks, threads>>>((unsigned(*)[brick_stride][brick_stride]) device_bgrid, bIn, bOut);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
    {
        std::cout << "Brick kernel execution failed. " << cudaGetErrorString(e) << std::endl;
    }

    if (verify) {
        bElem gpu_single_b[STRIDE][STRIDE][STRIDE];
        bElem gpu_b[STRIDE][STRIDE][STRIDE];
        bElem *dev_gpu_b;
        cudaMalloc(&dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem));
        no_prof_single_thread_49pt<<<1, 1>>>(A, dev_gpu_b);
        cudaDeviceSynchronize();
        cudaMemcpy(gpu_single_b, dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_b, B, STRIDE * STRIDE * STRIDE * sizeof(bElem), cudaMemcpyDeviceToHost);

        brick_storage = movBrickStorage(device_bstorage, cudaMemcpyDeviceToHost);
        bOut = BType(&binfo, brick_storage, brick_size);
        for (int i = OFF; i < N + OFF; i++) {
            for (int j = OFF; j < N + OFF; j++) {
                for (int k = OFF; k < N + OFF; k++) {
                    if (gpu_single_b[i][j][k] != gpu_b[i][j][k]) {
                        printf("Results mismatch at %d, %d, %d\n", i, j, k);
                        printf("Single threaded value: %f, Optimized value: %f\n", gpu_single_b[i][j][k], gpu_b[i][j][k]);
                        return 1;
                    }
                }
            }
        }
        if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, (bElem *) gpu_single_b, bgrid, bOut)) {
            throw std::runtime_error("Brick solution doesn't match!\n");
        }
        
        cudaFree(dev_gpu_b);
    }

    free(bgrid);
    free(binfo.adj);
    cudaFree(device_binfo);
    cudaFree(device_bgrid);
    cudaFree(A);
    cudaFree(B);
    return 0;
}
