#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ANNCUDA.cuh"
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

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
    printf("Successfully initialised handle for cublas\n");
         
}
void StopCUDA(void) {
	/* destroy the context on the GPU */
    printf("Attempting to  destroy handle for cublas\n");
    cublasStatus_t status;
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf(" Failed to close CUBLAS\n");
        exit(0);
    }else{
        printf ("Successfully destroyed handle\n");
    }  
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
void SyncHost2Dev(void *hostPtr, void *devPtr, size_t size) {
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

void SubNSegmentCUDA(float *srcPtr, int segLen, float *dstPtr,float lambda) {
    cublasStatus_t status;
    lambda = -1*lambda;
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

float computeDotProductCUDA(float * vectorL, float * vectorR,int dim,float  result){
	cublasStatus_t status;
    status= cublasSdot (handle, dim, vectorL, 1, vectorR, 1, &result);
     if (status != CUBLAS_STATUS_SUCCESS) {
        printf("computeDotProductCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }
    return result;
}

__global__ void  Hkern_SetValue(float * devarray, int dim , float value){
    int pos = 0 ;
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < dim) {
        devarray[pos] = value;
    }
}
void setValueCUDA(float * devarray, int dim , float value){
    int nBlocks;
    nBlocks = CEIL(dim, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
        exit(0);
    }
    Hkern_SetValue<<<nBlocks,THREADPERBLOCK>>>(devarray,dim,value);    


}
//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for forward propagation*/
//-------------------------------------------------------------------------------------------------------------------

void CopyNSegmentCUDA(float *srcPtr, int segLen, float *dstPtr) {
    cublasStatus_t status;
    status = cublasScopy(handle, segLen, srcPtr, 1, dstPtr, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CopyNSegmentCUDA: CUBLAS library copy function failed\n");
        exit(0);
    }
}

void HNBlasTNgemmCUDA(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, k, B, k, &beta, C, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("HNBlasTNgemmCUDA: CUBLAS library gemm function failed\n");
        exit(0);
    }
}

__global__ void HKern_ApplySigmoidAct(float *srcPtr, int len, float *dstPtr) {
    int pos;
    float floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = -1.0 * srcPtr[pos];
        dstPtr[pos] = 1.0 / (1.0 + exp(floatVal));
    }
}
void ApplySigmoidActCUDA(float *srcPtr, int len, float *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplySigmoidAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

__global__ void HKern_ApplyTanHAct(float *srcPtr, int len, float *dstPtr) {
    int pos;
    float floatVal;

    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        floatVal = srcPtr[pos];
        floatVal = exp(floatVal);
        dstPtr[pos] = (floatVal - 1 / floatVal) / (floatVal + 1 / floatVal);
    }
}
void ApplyTanHActCUDA(float *srcPtr, int len, float *dstPtr) {
    int nBlocks;
    
    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf( "ApplyTanHActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }	
    HKern_ApplyTanHAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

__global__ void HKern_ApplySoftmaxAct(float *srcPtr, int row, int col, float *dstPtr) {
    int frame, i, base, off;
    float den, floatVal;

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
void ApplySoftmaxActCUDA(float *srcPtr, int row, int col, float *dstPtr) {
    int nBlocks;

    nBlocks = CEIL(row, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySoftmaxActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplySoftmaxAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, row, col, dstPtr);
}

//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for back propagation*/
//-------------------------------------------------------------------------------------------------------------------
__global__ void  HKern_fillArrayWithValue(float * array,float value,int len){
	int pos;
	pos = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pos<len){
		array[pos] = value;
	}
}
void sumColsOfMatrixCUDA(float *dyFeatMat,float *dbFeatMat,int dim,int batchsamples){
 	int nBlocks;
 	float * ones;
    float value = 1.0;
 	const float alpha = 1;
 	const float beta = 0;

    cudaMalloc(&ones,sizeof(float)*batchsamples);
	
    nBlocks = CEIL(batchsamples, THREADPERBLOCK);
    
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySoftmaxActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_fillArrayWithValue<<<nBlocks, THREADPERBLOCK>>>(ones,value,batchsamples);
    
    cublasSgemv(handle,CUBLAS_OP_N, dim,batchsamples,&alpha,dyFeatMat,dim,ones,1,&beta,dbFeatMat,1);
	DevDispose(ones);
}


__global__ void HKern_ApplyDerivativeSigmoidAct(float * srcPtr, int len, float *dstPtr){
	int pos;
   
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = dstPtr[pos] * (srcPtr[pos] * (1 - srcPtr[pos])) ;
    }
}

void  computeSigmoidDrvCUDA(float *srcPtr, int len, float *dstPtr){
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
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos < len) {
        dstPtr[pos] = dstPtr[pos] * (1 - (srcPtr[pos] * srcPtr[pos]));
    }
}

void  computeTanHDrvCUDA(float *srcPtr, int len, float *dstPtr){
	int nBlocks;

    nBlocks = CEIL(len, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_ApplyDerivativeTanHAct<<<nBlocks, THREADPERBLOCK>>>(srcPtr, len, dstPtr);
}

void HNBlasNNgemmCUDA(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("HNBlasNNgemmCUDA: CUBLAS library gemm function failed\n");
        exit(0);
    }
}
void HNBlasNTgemmCUDA(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    cublasStatus_t status;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, n, &beta, C, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("HNBlasNNgemmCUDA: CUBLAS library gemm function failed\n");
        exit(0);
    }

}
//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for computing the hessian of the loss function with respect to network outputs*/
//-------------------------------------------------------------------------------------------------------------------
__global__ void HKern_AddElementstoDiagonalOfMatrix(float * lhs , float * rhs , int dim ,float * dst){
	int pos;
	pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos<dim){
    	dst[pos*(dim+1)] = lhs[pos*(dim+1)]+ rhs[pos]; 
    }
}

void AddElementstoDiagonalOfMatrix(float * lhs , float * rhs , int dim, float * dst){
	int nBlocks;
	nBlocks = CEIL(dim, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
    	exit(0);
    }
    HKern_AddElementstoDiagonalOfMatrix<<<nBlocks,THREADPERBLOCK>>>(lhs,rhs,dim,dst);

}
void computeMatVecProductCUDA(int m ,int n, float alpha,float * A, int lda, float * B,int incx, float beta, float * C,int incy ){
    cublasStatus_t status;
    status = cublasSgemv(handle,CUBLAS_OP_N,m,n,&alpha,A,lda,B,incx,&beta,C,incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("computeMatVecProduct: CUBLAS library gemm function failed\n");
        exit(0);
    }        
}

//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for individual parameter update through Rprop*/
//-------------------------------------------------------------------------------------------------------------------
__global__ void HKern_singleIterationOfRprop_plus(float * grad, float * cachedgrad, float* stepsize, float *delupdates, float * param, float cost, float oldcost, float eta_plus,float eta_minus,float delta_min,float delta_max,int dim){
    int pos;
    float v;
    pos = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (pos<dim){
        v = grad[pos] * cachedgrad[pos];
        if (v > 0){
            stepsize[pos] = min(stepsize[pos]*eta_plus, delta_max);
            delupdates[pos] = grad[pos] > 0 ? -1 * stepsize[pos] : stepsize[pos];
            param[pos]+= delupdates[pos];
        }else if (v < 0){
            stepsize[pos] = max (stepsize[pos]*eta_minus,delta_min);
            if (cost > oldcost) param[pos] =  param[pos] - delupdates[pos];
            grad[pos] = 0;
        }else {
            delupdates[pos] = grad[pos] > 0 ? -1 * stepsize[pos] : stepsize[pos];
            delupdates[pos] = grad[pos] ==0 ? 0: delupdates[pos];
            param[pos]+= delupdates[pos];
        }

    }

}

void singleIterationOfRprop_plus(float * grad, float * cachedgrad, float* stepsize, float *delupdates, float * param, float cost, float oldcost, float eta_plus,float eta_minus,float delta_min,float delta_max,int dim){
    int nBlocks;
    nBlocks = CEIL(dim, THREADPERBLOCK);
    if (nBlocks > MAXBLOCKNUM){
        printf("ApplySigmoidActCUDA: Block number exceeds the maximum\n");
        exit(0);
    }
    HKern_singleIterationOfRprop_plus<<<nBlocks,THREADPERBLOCK>>>(grad,cachedgrad,stepsize,delupdates,param,cost,oldcost,eta_plus,eta_minus,delta_min,delta_max,dim);
}

void unitTestsCUDA(){

    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
    //cublasHandle_t handle ; // CUBLAS context
    int j; // index of elements
    float * x; // n- vector on the host
    int n =6;
    //allocate memory to host
    x=( float *) malloc (n* sizeof (*x)); // host memory alloc

    for(j=0;j<n;j++){
        x[j]=( float )j; 
    }// x={0 ,1 ,2 ,3 ,4 ,5}

    printf ("x: ");
    for(j=0;j<n;j++){
        printf (" %2.0f,",x[j]); // print x
        printf ("\n");
    }
// on the device
    float * d_x; // d_x - x on the device
    cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device

    // memory alloc
    StartCUDA();
    //stat = cublasCreate (& handle ); // initialize CUBLAS context
//use cublas routine here

    stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); // cp x- >d_x

    float result =0; 

    stat=cublasSasum(handle,n,d_x,1,&result);

    // print the result
    printf ("sum of the absolute values of elements of x : %f \n",result );
    cudaFree (d_x ); // free device memory
    free (x); // free host memory

    float * A;
    float * B ;

    float * devA;
    float * devB;

    int num_elements = 5;
    size_t num_bytes = num_elements * sizeof(float);

    A  = (float *) malloc (num_bytes);
    B  = (float *) malloc(num_bytes);

    cudaMalloc(&devA,num_bytes);
    cudaMalloc(&devB,num_bytes);

    int i;
    for (i =0 ; i< 5;i++){
        A[i] = 1;
        B[i] = 2;
    }
    //stat = cublasSetVector (num_elements, sizeof (*A) ,A ,1 ,devA ,1); // cp x- >d_x
    //stat = cublasSetVector (num_elements, sizeof (*B) ,B ,1 ,devB ,1); // cp x- >d_x

   SyncHost2Dev(A,devA,num_bytes);
   SyncHost2Dev(B,devB,num_bytes);

   /*checking dot product*/
    float r=0 ;
    printf("doing dot product directly\n");
    r = computeDotProductCUDA(devA, devB, num_elements ,r);
    printf("result of dot product is  %f \n",r);
    
       

    cudaFree(devA);
    cudaFree(devB);
    free(A);
    free(B);

    /*testing sum of cols of matrix*/
    float matrix [] ={ 1,2,3,
                        4,5,6,
                        7,8,9
                    } ;   
    float *devM;
    size_t memory = 9 *  sizeof(float);
    cudaMalloc(&devM,memory);                  
    SyncHost2Dev(matrix,devM,memory);

    float *devV;
    size_t memoryV = 3 *  sizeof(float);
    cudaMalloc(&devV,memoryV);        
    sumColsOfMatrixCUDA(devM,devV,3,3);

    float *hostV;
    hostV = (float *) malloc(memoryV);
    SyncDev2Host(devV,hostV,memoryV);
    printf("sum of col result %f %f %f \n",hostV[0],hostV[1],hostV[2]);

    /*checking initialisation and copy  routine*/
    float *devVcopy;
    cudaMalloc(&devVcopy,memoryV);  
    SyncDev2Host(devVcopy,hostV,memoryV);
    printf("devVcopy before initialisation %f %f %f \n",hostV[0],hostV[1],hostV[2]);
    initialiseDeviceArrayWithZero(devVcopy,memoryV);
    SyncDev2Host(devVcopy,hostV,memoryV);
    printf("devVcopy after initialisation %f %f %f \n",hostV[0],hostV[1],hostV[2]);
    
    CopyMatrixOrVecCUDA(devV,devVcopy,3);
    SyncDev2Host(devVcopy,hostV,memoryV);
    printf("devVcopy after copy %f %f %f \n",hostV[0],hostV[1],hostV[2]);
    
    /*checking add segment*/
    AddNSegmentCUDA(devV,3, devVcopy,1.0);
    SyncDev2Host(devVcopy,hostV,memoryV);
    printf("devVcopy after addition %f %f %f \n",hostV[0],hostV[1],hostV[2]);
    
    SubNSegmentCUDA(devV,3, devVcopy,1.0);
    SyncDev2Host(devVcopy,hostV,memoryV);
    printf("devVcopy after subtraction %f %f %f \n",hostV[0],hostV[1],hostV[2]);
    
    ScaleNSegmentCUDA(3,2.5, devVcopy);
    SyncDev2Host(devVcopy,hostV,memoryV);
    printf("devVcopy after scaling %f %f %f \n",hostV[0],hostV[1],hostV[2]);
    
    float *devones;
    cudaMalloc(&devones,memoryV);
    float ones[] ={ 1 , 1,1 };
    SyncHost2Dev(ones,devones,memoryV);
    //cudaMemset(devones, 1, memoryV);


    /*checking matrix vector multiplication **/

    SyncDev2Host(devM,hostV,memoryV);
    printf("devM first 3 elements %f %f %f \n",hostV[0],hostV[1],hostV[2]);
     
    SyncDev2Host(devones,hostV,memoryV);
    printf("devones  %f %f %f \n",hostV[0],hostV[1],hostV[2]);
     
    computeMatVecProductCUDA(3,3,1,devM,3,devones,1,0,devones,1);
    SyncDev2Host(devones,hostV,memoryV);
    printf("devones after matrix vector multiplication %f %f %f \n",hostV[0],hostV[1],hostV[2]);
     


    cudaFree(devVcopy);
    cudaFree(devV);
    cudaFree(devM);
    free(hostV);
    
    StopCUDA();


}   


#ifdef __cplusplus
}
#endif




