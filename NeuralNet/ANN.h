
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
typedef struct _MSI *MSLink;



/*model set info -this struct is needed to compare the error on the validation dataset between two epochs*/
typedef struct _MSI{
double crtVal;
double bestValue;
double prevCrtVal;
}MSI;

/**this struct stores the search directions,residues and dw for each iteration of CG**/
typedef struct _ConjugateGradientInfo{
   double *delweightsUpdate;
   double *delbiasUpdate;
   double *residueUpdateWeights;
   double * residueUpdateBias;
   double *searchDirectionUpdateWeights;
   double *searchDirectionUpdateBias;
 }ConjuageGradientInfo;



/** this struct stores the directional error derivatives with respect to weights and biases**/
typedef struct _GaussNewtonProdInfo{
	double *vweights;
   	double *vbiases;
   	double *Ractivations;
}GaussNewtonProductInfo ;

typedef struct _TrainInfo{
	double *dwFeatMat; /* dE/dw matrix*/
	double *dbFeaMat; /* dE/db  vector */
	double *updatedWeightMat; /* stores the velocity in the weight space or accumulates  gradeints of weights*/
	double *updatedBiasMat;/* stores the velocity in the bias space  or accumulates gradients of biases*/
	double *bestWeightParamsHF;
	double *bestBiasParamsHF;
}TrainInfo;

typedef struct _ErrorElem{
	double *dxFeatMat;
	double *dyFeatMat;
}ErrElem;

typedef struct _FeatElem{
	double *xfeatMat;/* is a BatchSample size by feaDim matix */
	double *yfeatMat; /* is BatchSample Size by node matrix*/
}FeaElem;

/*structure for individual layers*/
typedef struct _LayerElem{
	int  id; /* each layer has a unique layer id */
	int  dim ; /* number of units in the hidden layer */
	LayerRole role; /* the type of layer : input,hidden or output */
	ActFunKind actfuncKind; /*each layer is allowed to have its own non-linear activation function */
	LELink src; /* pointer to the input layer */
	int srcDim; /* the number of units in the input layer */
	double *weights;/* the weight matrix of the layer should number of nodes by input dim*/
	double *bias; /* the bias vector */
	FELink feaElem; /* stores the  input activations coming into the layer as well the output activations going out from the layer */
	ERLink errElem;
	TRLink traininfo;/*struct that stores the error derivatives with respect to weights and biases */
	GNProdInfo gnInfo;
	CGInfo  cgInfo;
	double *bestweights;
	double *bestBias;
}LayerElem;

/*structure for ANN*/
typedef struct _ANNdef{
	int layerNum;
	LELink *layerList;/* list of layers*/
	OutFuncKind target; /* the activation function of the final output layer */
	ObjFuncKind errorfunc; /* the error function of the ANN */
	double *labelMat ; /* the target labels : BatchSample by targetDim matrix*/
}ANNDef;

//-------------------------------------------------------------------------------------------------------------------
/**This section of the code deals with parsing Command Line arguments**/
//-------------------------------------------------------------------------------------------------------------------
void cleanString(char *Name);
void loadLabels(double *labelMat, int *labels,char*filepath,char *datatype);
void loadMatrix(double *matrix,char *filepath, char *datatype);
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
void loadDataintoANN(double *samples, double *labels);

//-------------------------------------------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
//-------------------------------------------------------------------------------------------------------------------
void shuffle(int *array, size_t n);
void setUpMinibatchforHF(ADLink anndef);
void setUpForHF(ADLink anndef);
void reinitLayerFeaMatrices(ADLink anndef);
void reinitLayerErrFeaMatrices(ADLink anndef);
void initialiseErrElems(ADLink anndef);
void initialiseWithZero(double * matrix, int dim);
double drand();
double genrandWeight(double limit);
void initialiseBias(double *biasVec,int dim, int srcDim,ActFunKind actfunc);
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(double *weightMat,int length,int srcDim,ActFunKind actfunc);
void initialiseLayer(LELink layer,int i, LELink srcLayer);
void  initialiseDNN();
void initialise();

//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents auxilary functions that are required*/
//-------------------------------------------------------------------------------------------------------------------
/**this funciton copies one matrix/array into another*/
void copyMatrixOrVec(double *src, double *dest,int dim);
/* this function allows the addition of  two matrices or two vectors*/
void addMatrixOrVec(double *weightMat, double* dwFeatMat, int dim,double lambda);
void scaleMatrixOrVec(double* weightMat, double learningrate,int dim);
void subtractMatrix(double *dyfeat, double* labels, int dim, double lambda);
double computeTanh(double x);
double computeSigmoid(double x);

//-------------------------------------------------------------------------------------------------------------------
/*this section of the code implements the  forward propagation of a deep neural net **/
//-------------------------------------------------------------------------------------------------------------------
void computeNonLinearActOfLayer(LELink layer);
/* Yfeat is batchSamples by nodeNum matrix(stored as row major)  = X^T(row-major)-batch samples by feaMat * W^T(column major) -feaMat By nodeNum */
void computeLinearActivation(LELink layer);
void loadDataintoANN(double *samples, double *labels);
void fwdPassOfDNN(ADLink anndef);
void computeActivationOfOutputLayer(ADLink anndef);
//------------------------------------------------------------------------------------------------------
/*This section of the code implements the back-propation algorithm  to compute the error derivatives*/
//-------------------------------------------------------------------------------------------------------
void sumColsOfMatrix(double *dyFeatMat,double *dbFeatMat,int dim,int batchsamples);
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
void fillCache(LELink layer,int dim,Boolean weights);
void cacheParameters(ADLink anndef);
Boolean initialiseParameterCaches(ADLink anndef);
void perfBinClassf(double *yfeatMat, double *predictions,int dataSize);
double computeLogLikelihood(double* output, int batchsamples, int dim , int* labels);
/*The function finds the most active node in the output layer for each sample*/
void findMaxElement(double *matrix, int row, int col, double *vec);
/** the function calculates the percentage of the data samples correctly labelled by the DNN*/
double updatateAcc(int *labels, LELink layer,int dataSize);
void updateNeuralNetParams(ADLink anndef, double lrnrate, double momentum, double weightdecay);
void updateLearningRate(int currentEpochIdx, double *lrnRate);
Boolean terminateSchedNotTrue(int currentEpochIdx,double lrnrate);
void TrainDNNGD();

//------------------------------------------------------------------------------------------------------
/**this segment of the code is reponsible for accumulating the gradients **/
//------------------------------------------------------------------------------------------------------
void setHook(Ptr m, Ptr ptr,int incr);
Ptr getHook(Ptr m,int incr);
void accumulateLayerGradient(LELink layer,double weight);
void accumulateGradientsofANN(ADLink anndef);

//------------------------------------------------------------------------------------------------------
//additional functions to check CG sub-routines just in case 
//------------------------------------------------------------------------------------------------------
void normOfWeights(ADLink anndef);
void normaliseSearchDirections(ADLink anndef);
void normaliseResidueDirections(ADLink anndef, double* magnitudeOfGradient);
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
void updateParameterDirection(ADLink anndef,double beta);
void updateResidue(ADLink anndef);
double  computeQuadfun( ADLink anndef);
void updatedelParameters(double alpha);
void computeSearchDirMatrixProduct( ADLink anndef,double * searchVecMatrixVecProductResult);
void computeResidueDotProduct(ADLink anndef, double * residueDotProductResult);
void addTikhonovDamping(ADLink anndef);
//------------------------------------------------------
/* the following routines compute the directional derivative using forward differentiation*/
void updateRactivations(LELink layer);
void computeRactivations(LELink layer);
void computeVweightsProjection(LELink layer);
void computeDirectionalErrDrvOfLayer(LELink layer, int layerid);
void computeDirectionalErrDerivativeofANN(ADLink anndef);
void setParameterDirections(double * weights, double* bias, LELink layer);
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
void printVector(double * vector , int dim);
void UnitTest_computeGradientDotProd();
void printWeights(ADLink anndef, int i);
void printYfeat(ADLink anndef, int id);
void printMatrix(double *matrix,int row,int col);
void printDBWeights(ADLink anndef, int i);

/*This function is used to check the correctness of various routines */
void unitTests();
