
#define THREADPERBLOCK 256                      /*  */
#define MAXBLOCKNUM 2147483647  

void StartCUDA(void);
void StopCUDA(void);

//-------------------------------------------------------------------------------------------------------------------
/* routines transfer of data from host(device) to device(host)*/
//-------------------------------------------------------------------------------------------------------------------
void SyncDev2Host(void *devPtr, void *hostPtr, size_t size);
void initialiseDeviceArrayWithZero(void *devPtr,size_t size);
void SyncHost2Dev(void *hostPtr, float *devPtr, size_t size);
void DevDispose(void *devPtr); 
void DevNew(void **devAddr, size_t size);


//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents code contains auxillary  functions that are frequently used*/
//-------------------------------------------------------------------------------------------------------------------
void AddNSegmentCUDA(float * srcPtr, int segLen, float * dstPtr, float  lambda);
void ScaleNSegmentCUDA(int segLen, float scale, float * valPtr);
void SubNSegmentCUDA(float *srcPtr, int segLen, float *dstPtr,float lambda);
void CopyMatrixOrVecCUDA(float * src , float *dest, int dim);
void computeDotProductCUDA(float * vectorL, float * vectorR,int dim,float  *result);

//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for forward propagation*/
//-------------------------------------------------------------------------------------------------------------------
void CopyNSegmentCUDA(float *srcPtr, int segLen, float *dstPtr);
void HNBlasTNgemmCUDA(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C);
void ApplySigmoidActCUDA(float *srcPtr, int len, float *dstPtr);
void ApplyTanHActCUDA(float *srcPtr, int len, float *dstPtr);
void ApplySoftmaxActCUDA(float *srcPtr, int row, int col, float *dstPtr);
//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for back propagation*/
//-------------------------------------------------------------------------------------------------------------------
void sumColsOfMatrixCUDA(float *dyFeatMat,float *dbFeatMat,int dim,int batchsamples);
void  computeSigmoidDrvCUDA(float *srcPtr, int len, float *dstPtr);
void  computeTanHDrvCUDA(float *srcPtr, int len, float *dstPtr);
void HNBlasNNgemmCUDA(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C);
void HNBlasNTgemmCUDA(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C);

//-------------------------------------------------------------------------------------------------------------------
/*The following routines are used for computing the hessian of the loss function with respect to network outputs*/
//-------------------------------------------------------------------------------------------------------------------
 void AddElementstoDiagonalOfMatrix(float * lhs , float * rhs , int dim, float * dst);