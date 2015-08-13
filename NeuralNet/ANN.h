
typedef void * Ptr;
typedef enum {FALSE, TRUE} Boolean;
typedef enum {XENT, SSE} ObjFuncKind;
typedef enum {REGRESSION, CLASSIFICATION} OutFuncKind;
typedef enum {HIDDEN,OUTPUT} LayerRole;
typedef enum {SIGMOID,IDENTITY,TANH,SOFTMAX} ActFunKind;


typedef struct _LayerElem *LELink;
typedef struct _ANNdef *ADLink;
typedef struct _FeatElem *FELink;
typedef struct _ErrorElem *ERLink;
typedef struct _TrainInfo *TRLink;
typedef struct _GaussNewtonProdInfo *GNProdInfo;
typedef struct _ConjugateGradientInfo *CGInfo;
typedef struct _LayerRpropInfo *RLink;
typedef struct _MSI *MSLink;

typedef struct _NMatrix{
	int row;
	int col;
	float *elems;
	float *deviceElems;
}NMatrix;

typedef struct _NFVec{
	int len;
	float *elems;
	float *deviceElems;
}NFVector;

typedef struct _NIntVec{
	int len;
	int *elems;
	int *deviceElems;
}NIntVector;

/*model set info -this struct is needed to compare the error on the validation dataset between two epochs*/
typedef struct _MSI{
float crtVal;
float bestValue;
float prevCrtVal;
}MSI;

/**this struct stores the search directions,residues and dw for each iteration of CG**/
typedef struct _ConjugateGradientInfo{
   NMatrix * delweightsUpdate;
   NFVector * delbiasUpdate;
   NMatrix * residueUpdateWeights;
   NFVector * residueUpdateBias;
   NMatrix * searchDirectionUpdateWeights;
   NFVector * searchDirectionUpdateBias;
 }ConjuageGradientInfo;



/** this struct stores the directional error derivatives with respect to weights and biases**/
typedef struct _GaussNewtonProdInfo{
	NMatrix * vweights;
   	NFVector * vbiases;
   	NMatrix * Ractivations;
}GaussNewtonProductInfo ;

typedef struct _LayerRpropInfo{
	NMatrix * stepWght;
	NFVector * stepBias;
	NMatrix * delWght;
	NFVector *delbias;

}LayerRpropInfo;

typedef struct _TrainInfo{
	NMatrix * dwFeatMat; /* dE/dw matrix*/
	NFVector * dbFeaMat; /* dE/db  vector */
	NMatrix * updatedWeightMat; /* stores the velocity in the weight space or accumulates  gradeints of weights*/
	NFVector * updatedBiasMat;/* stores the velocity in the bias space  or accumulates gradients of biases*/
	NMatrix * bestWeightParamsHF;
	NFVector * bestBiasParamsHF;
}TrainInfo;

typedef struct _ErrorElem{
	NMatrix * dxFeatMat;
	NMatrix * dyFeatMat;
}ErrElem;

typedef struct _FeatElem{
	NMatrix * xfeatMat;/* is a BatchSample size by feaDim matix */
	NMatrix * yfeatMat; /* is BatchSample Size by node matrix*/
}FeaElem;

/*structure for individual layers*/
typedef struct _LayerElem{
	int  id; /* each layer has a unique layer id */
	int  dim ; /* number of units in the hidden layer */
	LayerRole role; /* the type of layer : input,hidden or output */
	ActFunKind actfuncKind; /*each layer is allowed to have its own non-linear activation function */
	LELink src; /* pointer to the input layer */
	int srcDim; /* the number of units in the input layer */
	NMatrix * weights;/* the weight matrix of the layer should number of nodes by input dim*/
	NFVector * bias; /* the bias vector */
	FELink feaElem; /* stores the  input activations coming into the layer as well the output activations going out from the layer */
	ERLink errElem;
	TRLink traininfo;/*struct that stores the error derivatives with respect to weights and biases */
	RLink rProp; /*struct that stores individual step updates when we employ Rprop*/ 
	GNProdInfo gnInfo;
	CGInfo  cgInfo;
	NMatrix * bestweights;
	NFVector * bestBias;
}LayerElem;

/*structure for ANN*/
typedef struct _ANNdef{
	int layerNum;
	LELink *layerList;/* list of layers*/
	OutFuncKind target; /* the activation function of the final output layer */
	ObjFuncKind errorfunc; /* the error function of the ANN */
	NMatrix * labelMat ; /* the target labels : BatchSample by targetDim matrix*/
}ANNDef;

//-------------------------------------------------------------------------------------------------------------------
/**This section of the code deals with parsing Command Line arguments**/
//-------------------------------------------------------------------------------------------------------------------
void cleanString(char *Name);
void loadLabels(NMatrix *labelMat, NIntVector *labels,char*filepath,char *datatype);
void loadMatrix(NMatrix * matrix,char *filepath, char *datatype);
void parseCfg(char * filepath);
void parseCMDargs(int argc, char *argv[]);

//-------------------------------------------------------------------------------------------------------------------
/**This section of the code deals with handling the batch sizes of the data**/
//-------------------------------------------------------------------------------------------------------------------
void setBatchSize(int sampleSize);
/**load minibatch into neural net for HF training **/
void setBatchSizetoHFminiBatch();
void loadMiniBatchintoANN();
/**load entire batch into the neural net**/
void loadDataintoANN(NMatrix* samples, NMatrix * labels);

//-------------------------------------------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
//-------------------------------------------------------------------------------------------------------------------
void shuffle(int *array, size_t n);
void setUpMinibatchforHF(ADLink anndef);
void setUpForHF(ADLink anndef);
void reinitLayerFeaMatrices(ADLink anndef);
void reinitLayerErrFeaMatrices(ADLink anndef);
void initialiseErrElems(ADLink anndef);
float drand();
float genrandWeight(float limit);
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(NMatrix * weightMat,int length,int srcDim,ActFunKind actfunc);
void initialiseLayer(LELink layer,int i, LELink srcLayer);
void  initialiseDNN();
void initialise();

//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents code that creates matrices in host and device(In case of GPU computing)*/
//-------------------------------------------------------------------------------------------------------------------
NMatrix * CreateMatrix(int row , int col);
NFVector * CreateFloatVec( int len);
NIntVector * CreateIntVec( int len);
void DisposeMatrix(NMatrix *matrix);
void DisposeFloatVec(NFVector *vector);
void DisposeIntVec(NIntVector *vector);
void initialiseWithZeroMatrix(NMatrix * matrix, int dim,size_t size);
void initialiseWithZeroFVector(NFVector * matrix, int dim,size_t size);
void initialiseWithZeroIVector(NIntVector * matrix, int dim,size_t size);
void setValueInMatrix( NMatrix *matrix, float value);
void setValueInVec( NFVector *vector, float value);
//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents auxilary functions that are required*/
//-------------------------------------------------------------------------------------------------------------------
void CopyMatrix (NMatrix *src,int lstartpos, NMatrix *dest,int rstartpos,int dim);
void CopyVec (NFVector *src,int lstartpos, NFVector *dest,int rstartpos,int dim);
void CopyVecToMat (NFVector *src,int lstartpos, NMatrix *dest,int rstartpos,int dim);
void CopyMatrixToVec (NMatrix *src,int lstartpos, NFVector *dest,int rstartpos,int dim);

/* this function allows the addition of  two matrices or two vectors*/
void addMatrix(NMatrix * weights, NMatrix * dwFeatMat,int dim, float lambda);
void addVec(NFVector * weights, NFVector * dwFeatMat,int dim, float lambda);
void scaleMatrix(NMatrix * Mat, float scale ,int dim);
void scaleVec(NFVector * Mat, float scale ,int dim);
void subtractMatrix(NMatrix *dyfeat, NMatrix* labelMat, int dim, float lambda);
float computeTanh(float x);
float computeSigmoid(float x);
float max(float a, float b);
float min(float a, float b);
void HNBlasNNgemm(int srcDim, int batchsamples, int dim, float alpha, NMatrix *weights, NMatrix *dyFeatMat, float beta, NMatrix *dxFeatMat);
void HNBlasNTgemm(int srcDim, int dim,  int batchsamples, float alpha , NMatrix* xfeatMat, NMatrix * dyFeatMat, float beta, NMatrix * dwFeatMat);
float computeDotProductMatrix(NMatrix * vectorL, NMatrix * vectorR,int dim);
float computeDotProductVector(NFVector * vectorL, NFVector * vectorR,int dim);

//-------------------------------------------------------------------------------------------------------------------
/*this section of the code implements the  forward propagation of a deep neural net **/
//-------------------------------------------------------------------------------------------------------------------
void computeNonLinearActOfLayer(LELink layer);
/* Yfeat is batchSamples by nodeNum matrix(stored as row major)  = X^T(row-major)-batch samples by feaMat * W^T(column major) -feaMat By nodeNum */
void computeLinearActivation(LELink layer);
void fwdPassOfDNN(ADLink anndef);
void computeActivationOfOutputLayer(ADLink anndef);
//------------------------------------------------------------------------------------------------------
/*This section of the code implements the back-propation algorithm  to compute the error derivatives*/
//-------------------------------------------------------------------------------------------------------
void sumColsOfMatrix(NMatrix *dyFeatMat,NFVector *dbFeatMat,int dim,int batchsamples);
void computeActivationDrv (LELink layer);
/**compute del^2L J where del^2L is the hessian of the cross-entropy softmax with respect to output acivations **/ 
void computeLossHessSoftMax(LELink layer);
/*compute del^2L*J where L can be any convex loss function**/
void computeHessOfLossFunc(LELink layer, ADLink anndef);
void calcOutLayerBackwardSignal(LELink layer,ADLink anndef );
/**function computes the error derivatives with respect to the weights and biases of the neural net*/
void backPropBatch(ADLink anndef,Boolean doHessVecProd);

//------------------------------------------------------------------------------------------------------
/*This section implements gradient descent learning net**/
//------------------------------------------------------------------------------------------------------
void cacheParameters(ADLink anndef);
Boolean initialiseParameterCaches(ADLink anndef);
void perfBinClassf(NMatrix *yfeatMat, int *predictions,int dataSize);
float computeLogLikelihood(NMatrix  *  output, int batchsamples, int dim , NIntVector * labels);
/*The function finds the most active node in the output layer for each sample*/
void findMaxElement(float * matrix, int row, int col, int * vec);
/** the function calculates the percentage of the data samples correctly labelled by the DNN*/
float updatateAcc(NIntVector * labels, LELink layer,int dataSize);
void updateNeuralNetParams(ADLink anndef, float lrnrate, float momentum, float weightdecay);
void updateLearningRate(int currentEpochIdx, float * lrnRate);
Boolean terminateSchedNotTrue(int currentEpochIdx,float lrnrate);
//------------------------------------------------------------------------------------------------------
/** batch gradient descent training*/
//------------------------------------------------------------------------------------------------------
void TrainDNNGD();

//------------------------------------------------------------------------------------------------------
/* batch Rprop training*/
//------------------------------------------------------------------------------------------------------
void IterOfRpropLayer(LELink layer,float cost, float oldcost);
void updateWithRprop(ADLink anndef,float cost, float oldcost);
void TrainDNNRprop();

//------------------------------------------------------------------------------------------------------
/**this segment of the code is reponsible for accumulating the gradients **/
//------------------------------------------------------------------------------------------------------
void setHook(Ptr m, Ptr ptr,int incr);
Ptr getHook(Ptr m,int incr);
void accumulateLayerGradient(LELink layer,float weight);
void accumulateGradientsofANN(ADLink anndef);

//------------------------------------------------------------------------------------------------------
//additional functions to check CG sub-routines just in case 
//------------------------------------------------------------------------------------------------------
void normOfWeights(ADLink anndef);
void normaliseSearchDirections(ADLink anndef);
void normaliseResidueDirections(ADLink anndef, float * magnitudeOfGradient);
void computeNormOfGradient(ADLink anndef);
void computeNormOfAccuGradient(ADLink anndef);
void normOfVweights(ADLink anndef);
void printGVoutput(ADLink anndef);
void computeSearchDirDotProduct(ADLink anndef);
void displaySearchDirection(ADLink anndef);
void displayResidueDirection(ADLink anndef);
void displayVweights(ADLink anndef);
void displaydelWs(ADLink anndef);
void normofGV(ADLink anndef);
void normofDELW(ADLink anndef);

//------------------------------------------------------------------------------------------------------
/* This section of the code implements HF training*/
//------------------------------------------------------------------------------------------------------
void resetdelWeights(ADLink anndef);
void getBestParamsCG(ADLink anndef);
void cacheParamsCG(ADLink anndef);
void updateNeuralNetParamsHF( ADLink anndef);
void backtrackNeuralNetParamsCG(ADLink anndef);

//-------------------------------------------------------
/**This section of the code implements the small sub routinesof the  conjugate Gradient algorithm **/
void updateParameterDirection(ADLink anndef,float beta);
void updateResidue(ADLink anndef);
float  computeQuadfun( ADLink anndef);
void updatedelParameters(float alpha);
void computeSearchDirMatrixProduct( ADLink anndef, float * searchVecMatrixVecProductResult);
void computeResidueDotProduct(ADLink anndef, float * residueDotProductResult);
void addTikhonovDamping(ADLink anndef);
//------------------------------------------------------
/* the following routines compute the directional derivative using forward differentiation*/
void updateRactivations(LELink layer);
void computeRactivations(LELink layer);
void computeVweightsProjection(LELink layer);
void computeDirectionalErrDrvOfLayer(LELink layer, int layerid);
void computeDirectionalErrDerivativeofANN(ADLink anndef);
void setParameterDirections(NMatrix * weights, NFVector *  bias, LELink layer);
void setSearchDirectionCG(ADLink anndef, Boolean Parameter);
//-----------------------------------------------------
void initialiseResidueaAndSearchDirection(ADLink anndef);
void runConjugateGradient();
void TrainDNNHF();

//-------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------

void freeMemoryfromANN();
void printLayerMatrices(ADLink anndef);
void printLayerDWMatrices(ADLink anndef);
void printArray(float * vector , int dim);
void printIArray(int *vector , int dim);
void printVector(NFVector * vector , int dim);
void UnitTest_computeGradientDotProd();
void printWeights(ADLink anndef, int i);
void printYfeat(ADLink anndef, int id);
void printMatrix(NMatrix * matrix,int row,int col);
void printDBWeights(ADLink anndef, int i);

/*This function is used to check the correctness of various routines */
void unitTestofIterRprop(int iter);
void unitTests();
