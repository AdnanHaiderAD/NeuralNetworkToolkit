#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "HCUDA.h"

#define CEIL(x,y) (((x)+(y)-1) / (y))
cublasHandle_t handle;				/*  */


void StartCUDA(void) {
    cublasStatus_t status;
  	/* initiate CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf(" Failed to initialise CUBLAS");
        exit(0);
    }    
}
void StopCUDA(void) {
	/* destroy the context on the GPU */
    cublasDestroy(handle);
}

//-------------------------------------------------------------------------------------------------------------------
/* routines transfer of data from host(device) to device(host)*/
//-------------------------------------------------------------------------------------------------------------------
void SyncDev2Host(void *devPtr, void *hostPtr, size_t size) {
    cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
}

void initialiseDeviceArrayWithZero(void *devPtr,size_t size){
	cudaMemset(devPtr, 0, size);
}
/*  */
void SyncHost2Dev(void *hostPtr, NFloat *devPtr, size_t size) {
    cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);	
}

/*  */
void DevDispose(void *devPtr) {
    cudaFree(devPtr);
}

/*  */
void DevNew(void **devAddr, size_t size) {
    cudaMalloc(devAddr, size);
}

//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents code contains auxillary  functions that are frequently used*/
//-------------------------------------------------------------------------------------------------------------------
void AddNSegmentCUDA(float * srcPtr, int segLen, float * dstPtr, float  lambda) {
	cublasStatus_t status;
	status = cublasSaxpy(handle, segLen, &lambda, srcPtr, 1, dstPtr, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("AddNSegmentCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }
}

/*  */
void ScaleNSegmentCUDA(int segLen, float scale, float * valPtr) {
    cublasStatus_t status;
	status = cublasSscal(handle, segLen, &scale, valPtr, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ScaleNSegmentCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }
}

void SubNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr,float lambda) {
    cublasStatus_t status;
    status = cublasSaxpy(handle, segLen, &lambda, srcPtr, 1, dstPtr, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("SubNSegmentCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }
}

void CopyMatrixOrVecCUDA(float * src , float *dest, int dim){
    const float * sr = src;
	cublasStatus_t status;
    status = cublasScopy(handle,dim, sr, 1,dest, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CopyMatrixOrVecCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }

}
//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for forward propagation*/
//-------------------------------------------------------------------------------------------------------------------

void CopyNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr) {
    cublasStatus_t status;
    status = cublasScopy(handle, segLen, srcPtr, 1, dstPtr, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CopyNSegmentCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }
}

void HNBlasTNgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, C, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("HNBlasTNgemmCUDA: CUBLAS library gemm function failed\n");
        exit(0);
    }
}

__global__ void HKern_ApplySigmoidAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = -1.0 * srcPtr[pos];
        dstPtr[pos] = 1.0 / (1.0 + exp(floatVal));
    }
}
void ApplySigmoidActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplySigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

__global__ void HKern_ApplyTanHAct(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = srcPtr[pos];
        floatVal = exp(floatVal);
        dstPtr[pos] = (floatVal - 1 / floatVal) / (floatVal + 1 / floatVal);
    }
}
void ApplyTanHActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr) {
    int nBlocks;
    
    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf( "ApplyTanHActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }	
    HKern_ApplyTanHAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

__global__ void HKern_ApplySoftmaxAct(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int frame, i, base, off;
    NFloat den, floatVal;

    frame = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (frame < row) {
        den = 0.0;
        base = frame * col;
        for (i = 0, off = base; i < col; ++i, ++off) {
            floatVal = srcPtr[off];
            floatVal = exp(floatVal);
            dstPtr[off] = floatVal;
            den += floatVal;
        }
        for (i = 0, off = base; i < col; ++i, ++off) {
            dstPtr[off] /= den;
        }
    }
}
void ApplySoftmaxActCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){}
        printf("ApplySoftmaxActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplySoftmaxAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, dstPtr);
}

//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for back propagation*/
//-------------------------------------------------------------------------------------------------------------------
__global__ void  HKern_fillArrayWithValue(float* array,float value,int len){
	int pos;
	pos = (blockIdx.x * blockDim.x) + threadIdx.x
	if (pos<len){
		array[pos] = value;
	}
}
void sumColsOfMatrix(float *dyFeatMat,float *dbFeatMat,int dim,int batchsamples){
 	int nBlocks;
 	float * array;
 	const float alpha = 1;
 	const float beta = 0;
 	cudaMalloc((**void)&array,sizeof(float)*batchsamples);
	nBlocks = CEIL(batchsamples, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){}
        printf("ApplySoftmaxActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_fillArrayWithValue<<<nBlocks, THREADPERBLOCK>>>(array,value,batchsamples);
    cublasSgemv(handle,CUBLAS_OP_N, dim,batchsamples,&alpha,dyFeatMat,dim,ones,1,&beta,dbFeatMat,1);
	DevDispose(array);
}


__global__ void HKern_ApplyDerivativeSigmoidAct(float * srcPtr, int len, float *dstPtr){
	int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = dstPtr[pos] * (srcPtr[pos] * (1 - srcPtr[pos])) ;
    }
}

void  computeSigmoidDrvCUDA(NFloat *srcPtr, int len, NFloat *dstPtr){
	 int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplyDerivativeSigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

__global__ void HKern_ApplyDerivativeTanHAct(float * srcPtr, int len, float *dstPtr){
	int pos;
    NFloat floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = dstPtr[pos] * (1 - (srcPtr[pos] * srcPtr[pos]));
    }
}

void  computeTanHDrvCUDA(NFloat *srcPtr, int len, NFloat *dstPtr){
	 int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplyDerivativeTanHAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

void HNBlasNNgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("HNBlasNNgemmCUDA: CUBLAS library gemm function failed\n");
        exit(0);
    }
}
void HNBlasNTgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C) {
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("HNBlasNNgemmCUDA: CUBLAS library gemm function failed\n");
        exit(0);
    }

}



