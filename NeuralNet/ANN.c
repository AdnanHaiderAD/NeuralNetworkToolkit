#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ANN.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>




#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
	#define M_PI 3.14159265358979323846
	#endif
#ifndef CACHESIZE
	#define CACHESIZE 100
#endif

#ifdef CBLAS
#include "../CBLAS/include/cblas.h"
#endif

#ifdef CUDA
#include "../Cuda/ANNCUDA.cuh"
#endif	
/*hyper-parameters deep Neural Net training initialised with default values*/
static float weightdecay = 1;
static float momentum = 0; 
static int  maxEpochNum = 5; 
static float initLR = 0.05; 
static float minLR = 0.0001;

/*hyper-parameters specific to Rprop training*/
static float init_delta = 0.01;//default
static float delta_max = 5; //default
static float delta_min = -2; //default
static float eta_plus = 1.05; //default
static  float eta_minus = 0.5 ;//default

/*hyper parameters specific to HF training */
static  int maxNumOfCGruns = 50;
static float samplingRateHf = 0.4;

/*training data set and validation data set*/
static int BATCHSAMPLES; //the number of samples to load into the DNN
static NMatrix * inputData = NULL;
static NMatrix * miniBatchforHF = NULL;
static NMatrix * labelMat = NULL;
static NIntVector * labels = NULL;
static NIntVector* minBatchLabels = NULL ;
static NMatrix * validationData = NULL;
static NIntVector * validationLabelIdx = NULL;
static int trainingDataSetSize;
static int validationDataSetSize;


/*configurations for DNN architecture*/
static Boolean doHF = FALSE;
static Boolean useGNMatrix = FALSE;
static Boolean doRprop = FALSE;
static MSLink modelSetInfo = NULL;
static ActFunKind *actfunLists;
static int *hidUnitsPerLayer;
static int numLayers;
static int inputDim;
static int targetDim;
static OutFuncKind target;
static ObjFuncKind errfunc; 
static ADLink anndef = NULL;

//-------------------------------------------------------------------------------------------------------------------
/**This section of the code deals with parsing Command Line arguments**/
//-------------------------------------------------------------------------------------------------------------------
void cleanString(char *Name){
	char *pos;
	if ((pos=strchr(Name, '\n')) != NULL)
	    *pos = '\0';
}
void loadLabels(NMatrix *labelMat, NIntVector *labels,char*filepath,char *datatype){
	FILE *fp;
	int i,c;
	char *line = NULL;
	size_t len = 0;
	int id ;
	int samples = 0;
	samples  = strcmp(datatype,"train")== 0 ? trainingDataSetSize: validationDataSetSize;
	fp = fopen(filepath,"r");
	i = 0;
	while(getline(&line,&len,fp)!=-1){
		cleanString(line);
		//extracting labels 
		if (strcmp(datatype,"train")==0){
			id  = strtod(line,NULL);
			labels->elems[i] = (int) id;
			labelMat->elems[i*targetDim+id] = 1;
			if (i> trainingDataSetSize){
				printf("Error! : the number of training labels doesnt match the size of the training set \n");
				exit(0);
			}
		}else if(strcmp(datatype,"validation")==0){
			id  = strtod(line,NULL);
			labels->elems[i] = (int) id;
			if(i > validationDataSetSize){
				printf("Error! : the number of validation target labels doesnt match the size of the validation set \n");
				exit(0);
			}
		}
		i+=1;
	}
	free(line);
	fclose(fp);	
	#ifdef CUDA
	if (strcmp(datatype,"train")==0){
		SyncHost2Dev(labelMat->elems,labelMat->deviceElems,sizeof(float)*labelMat->row*labelMat->col);
	}
	SyncHost2Dev(labels->elems,labels->deviceElems,sizeof(int)*labels->len);
	#endif	
}

void loadMatrix(NMatrix *matrix,char *filepath, char *datatype){
	FILE *fp;
	int i;
	char *line = NULL;
	size_t len = 0;
	char* token;
	
	fp = fopen(filepath,"r");
	i = 0;
	while(getline(&line,&len,fp)!=-1){
		token = strtok(line,",");
		while (token != NULL){
			matrix->elems[i] = strtod(token,NULL);
			token = strtok(NULL,",");
			if (strcmp(datatype,"train")== 0){
				if (i > trainingDataSetSize*inputDim){
					printf("Error: either the size of the training set or the dim of the  feature vectors have been incorrectly specified in config file \n");
					printf("correct size should be %d \n  but is %d ",trainingDataSetSize*inputDim,i);
					exit(0);
				}
			}else if (strcmp(datatype,"validation")==0){
				if (i > validationDataSetSize*inputDim){
					printf("Error: either the size of the   validation dataset or the dim of the  target vectors have been incorrectly specified in config file \n");
					exit(0);
				}
			}
			i+=1;
		}
	}	
	free(line);
	fclose(fp);
	#ifdef CUDA
	SyncHost2Dev(matrix->elems,matrix->deviceElems,sizeof(float)*matrix->row*matrix->col);
	#endif
}

void parseCfg(char * filepath){
	FILE *fp;
	char *line = NULL;
	size_t len = 0;
	char* token;
	char* list;
	char *pos;

	fp = fopen(filepath,"r");
	while(getline(&line,&len,fp)!=-1){
		token = strtok(line," : ");
		if (strcmp(token,"momentum")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			momentum = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"weightdecay")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			weightdecay = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;	
		}
		if (strcmp(token,"minLR")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			minLR = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue	;
		}
		if (strcmp(token,"maxEpochNum")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			maxEpochNum = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"initLR")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			initLR =  strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		
		if (strcmp(token,"numLayers")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			numLayers = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"inputDim")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			inputDim = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"targetDim")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			targetDim = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"trainingDataSetSize")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			trainingDataSetSize = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"validationDataSetSize")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			validationDataSetSize = (int)strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"Errfunc")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			if (strcmp("XENT",token)==0){
				errfunc = XENT;
				target = CLASSIFICATION;
			}else if (strcmp("SSE",token)==0){
				printf("ITS Not XENT\n");
				errfunc = SSE ;
				target = REGRESSION ;
			}	
			continue;
		}
		if (strcmp(token,"doHF")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			if (strcmp("TRUE",token)==0){
				doHF = TRUE;
			}else {
				doHF = FALSE;
			}
			continue;
		}
		if (strcmp(token,"doRprop")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			if (strcmp("TRUE",token)==0){
				doRprop = TRUE;
			}else {
				doRprop = FALSE;
			}
			continue;
		}
		if (strcmp(token,"useGNMatrix")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			if (strcmp("TRUE",token)==0){
				useGNMatrix = TRUE;
			}else {
				useGNMatrix = FALSE;
			}
			continue;
		}
		
		if (strcmp(token,"samplingRateHf")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			samplingRateHf = (float) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"delta_min")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			samplingRateHf = (float) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"delta_max")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			samplingRateHf = (float) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"eta_minus")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			samplingRateHf = (float) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"eta_plus")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			samplingRateHf = (float) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"init_delta")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			samplingRateHf = (float) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"hiddenUnitsPerLayer")==0){
			hidUnitsPerLayer = malloc(sizeof(int)*numLayers);
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			list = token;
			while(token != NULL){
				token = strtok(NULL,":");
			}	
			token = strtok(list,",");
			cleanString(token);
			int count = 0;
			while(token !=NULL){
				*(hidUnitsPerLayer+count) = (int) strtod (token,NULL);
				count+=1;
				token = strtok(NULL,",");
				if (token == NULL) break;
				cleanString(token);
			}
			continue;
		}
		if (strcmp(token,"activationfunctionsPerLayer")==0){
			actfunLists = malloc( sizeof(ActFunKind)*numLayers);
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			list = token;
			while(token !=NULL){
				token = strtok(NULL,":");
			}
			token = strtok(list,",");
			cleanString(token);
		   int count = 0;
			while(token !=NULL){
			if (strcmp(token,"SIGMOID")==0){
				*(actfunLists+count) = SIGMOID ;
				}else if (strcmp(token,"IDENTITY")==0){
					*(actfunLists+count) = IDENTITY;
				}else if (strcmp(token,"TANH")==0){
					*(actfunLists+count) = TANH ;
				}else if (strcmp(token,"SOFTMAX")==0){
					*(actfunLists+count) = SOFTMAX;
			}		
				count+=1;
				token = strtok(NULL,",");
				if (token ==NULL) break;
				cleanString(token);
			}	
			continue;
		}
		
	}
	free(line);
	fclose(fp);
}

void parseCMDargs(int argc, char *argv[]){
	int i;
	if (strcmp(argv[1],"-C")!=0){
		printf("the first argument to ANN must be the config file \n");
		exit(0);
	}
	for (i = 1 ; i < argc;i++){
		if (strcmp(argv[i],"-C") == 0){
			++i;
			printf("parsing cfg\n");
			parseCfg(argv[i]);
			//parse the config file to set the configurations for DNN architecture
			printf("config file %s has been successfully parsed \n",argv[i]);
			continue;
		}
	   if(strcmp(argv[i],"-S") == 0){
	   		++i;
			//load the input batch for training
			printf("parsing training data file \n");
			inputData = CreateMatrix(trainingDataSetSize,inputDim);
			initialiseWithZeroMatrix(inputData,trainingDataSetSize*inputDim,sizeof(float)*trainingDataSetSize*inputDim);
			printf("row  %d col %d  of training data \n", inputData->row,inputData->col);
			loadMatrix(inputData,argv[i],"train");
			printf("training samples from %s have been successfully loaded \n",argv[i]);
			continue;
		}
		if(strcmp(argv[i],"-L")==0){
			++i;
			//load the training labels or outputs in case of regression
			printf("parsing training-labels file with trainingDataSize %d \n", trainingDataSetSize);
			labelMat = CreateMatrix(trainingDataSetSize,targetDim);
			initialiseWithZeroMatrix(labelMat,trainingDataSetSize*targetDim, sizeof(float)*trainingDataSetSize*targetDim);
			
			labels = CreateIntVec(trainingDataSetSize);
			initialiseWithZeroIVector(labels,trainingDataSetSize,sizeof(int)*trainingDataSetSize);
			loadLabels(labelMat,labels,argv[i],"train");
			printf("training labels from %s have been successfully loaded \n",argv[i]);
			continue;
		} 
		if(strcmp(argv[i],"-v")==0){
			++i;
			//load the validation training samples 
			printf("parsing validation-data file \n");
			validationData = CreateMatrix(validationDataSetSize,inputDim);
			initialiseWithZeroMatrix(validationData,validationDataSetSize*inputDim,sizeof(float)*validationDataSetSize*inputDim);
			loadMatrix(validationData,argv[i],"validation");
			printf("samples from validation file %s have been successfully loaded \n",argv[i]);
			continue;
		}
		if(strcmp(argv[i],"-vl")==0){
			++i;
			//load the validation training labels or expected outputs
			printf("parsing validation-data-label file \n");
			validationLabelIdx = CreateIntVec(validationDataSetSize); 
			initialiseWithZeroIVector(validationLabelIdx,validationDataSetSize,sizeof(int)*validationDataSetSize);
			loadLabels(NULL,validationLabelIdx,argv[i],"validation");
			printf("validation labels from %s have been successfully loaded\n",argv[i]);
			continue;
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
/**This section of the code deals with handling the batch sizes of the data**/
//-------------------------------------------------------------------------------------------------------------------
void setBatchSize(int sampleSize){
	//BATCHSAMPLES = sampleSize < CACHESIZE ? sampleSize : CACHESIZE;
	BATCHSAMPLES = sampleSize;
}
/**load minibatch into neural net for HF training **/
 void setBatchSizetoHFminiBatch(){
 	BATCHSAMPLES = (int)(samplingRateHf * BATCHSAMPLES);
 	printf( "BATCH SIZE %d ",BATCHSAMPLES);
}

void loadMiniBatchintoANN(){
	anndef->layerList[0]->feaElem->xfeatMat = miniBatchforHF;
}
/**load entire batch into the neural net**/
void loadDataintoANN(NMatrix *samples, NMatrix *labelMat){
	anndef->layerList[0]->feaElem->xfeatMat = samples;
	anndef->labelMat = labelMat;
}   
//-------------------------------------------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
//-------------------------------------------------------------------------------------------------------------------
void setUpForRprop(ADLink anndef){
	LELink layer;
	int i,srdim,dim;
	printf("Setting additional structures for Rprop training \n");
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		//set up structure to accumulate gradients
		if (layer->traininfo->updatedWeightMat == NULL && layer->traininfo->updatedBiasMat == NULL){
			layer->traininfo->updatedWeightMat = CreateMatrix(layer->dim ,layer->srcDim);
			layer->traininfo->updatedBiasMat = CreateFloatVec(layer->dim);
			
			initialiseWithZeroMatrix(layer->traininfo->updatedWeightMat,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
			initialiseWithZeroFVector(layer->traininfo->updatedBiasMat,layer->dim,sizeof(float)*layer->dim);

			assert(layer->rProp==NULL); 
			layer->rProp = (RLink) malloc(sizeof(LayerRpropInfo));

			layer->rProp->stepWght = CreateMatrix(layer->dim ,layer->srcDim);
			layer->rProp->stepBias = CreateFloatVec(layer->dim);
			setValueInMatrix(layer->rProp->stepWght,init_delta);
			setValueInVec(layer->rProp->stepBias,init_delta);

			layer->rProp->delWght =  CreateMatrix(layer->dim ,layer->srcDim);
			layer->rProp->delbias =  CreateFloatVec(layer->dim);
			initialiseWithZeroMatrix(layer->rProp->delWght,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
			initialiseWithZeroFVector(layer->rProp->delbias,layer->dim,sizeof(float)*layer->dim);
		}
		else if (layer->traininfo->updatedWeightMat == NULL || layer->traininfo->updatedBiasMat == NULL){
			printf("Error something went wrong during the initialisation of updatedWeightMat and updateBiasMat in the layer %d \n",i);
			exit(0);
		}


	}
}	

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
void setUpMinibatchforHF(ADLink anndef){
	int i, index;
	int minibatchSize =  (int)(samplingRateHf * BATCHSAMPLES);
	printf( "BATCH SIZE %d ",minibatchSize);
 	if (miniBatchforHF == NULL){
 		miniBatchforHF = CreateMatrix(minibatchSize,inputDim);
 		minBatchLabels = CreateIntVec(minibatchSize);
 	}
 	initialiseWithZeroMatrix(miniBatchforHF,minibatchSize,sizeof(float)*minibatchSize*inputDim);
 	initialiseWithZeroIVector (minBatchLabels,minibatchSize,sizeof(float)*minibatchSize);
 	int * randomIndices = malloc(sizeof(int)*BATCHSAMPLES);
 	for(i = 0; i < BATCHSAMPLES;i++)randomIndices[i] =i;
 	shuffle(randomIndices,BATCHSAMPLES);	
 	for (i = 0 ; i<minibatchSize;i++){
 		index = randomIndices[i];
 		minBatchLabels->elems[i] = labels->elems[index];
 		CopyMatrix (inputData, (index*inputDim),miniBatchforHF,(i*inputDim),inputDim);
 		
 	}
	free(randomIndices);
	#ifdef CUDA
		SyncHost2Dev(miniBatchforHF->elems,miniBatchforHF->deviceElems,sizeof(float)*minibatchSize);
	#endif
 }

void setUpForHF(ADLink anndef){
	LELink layer;
	int i,srdim,dim;
	printf("Setting additional structures for HF training \n");
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		//set up structure to accumulate gradients
		if (layer->traininfo->updatedWeightMat == NULL && layer->traininfo->updatedBiasMat == NULL){
			layer->traininfo->updatedWeightMat = CreateMatrix(layer->dim ,layer->srcDim);
			layer->traininfo->updatedBiasMat = CreateFloatVec(layer->dim);
			
			initialiseWithZeroMatrix(layer->traininfo->updatedWeightMat,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
			initialiseWithZeroFVector(layer->traininfo->updatedBiasMat,layer->dim,sizeof(float)*layer->dim);

			assert(layer->traininfo->bestWeightParamsHF==NULL); 
			assert( layer->traininfo->bestBiasParamsHF==NULL);

			layer->traininfo->bestWeightParamsHF = CreateMatrix(layer->dim ,layer->srcDim);
			layer->traininfo->bestBiasParamsHF = CreateFloatVec(layer->dim);
			initialiseWithZeroMatrix(layer->traininfo->bestWeightParamsHF,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
			initialiseWithZeroFVector(layer->traininfo->bestBiasParamsHF,layer->dim,sizeof(float)*layer->dim);
		}
		else if (layer->traininfo->updatedWeightMat == NULL || layer->traininfo->updatedBiasMat == NULL){
			printf("Error something went wrong during the initialisation of updatedWeightMat and updateBiasMat in the layer %d \n",i);
			exit(0);
		}

		layer->cgInfo = malloc(sizeof(ConjuageGradientInfo));
		layer->cgInfo->delweightsUpdate = CreateMatrix(layer->dim,layer->srcDim);
		layer->cgInfo->residueUpdateWeights = CreateMatrix(layer->dim,layer->srcDim);
		layer->cgInfo->searchDirectionUpdateWeights = CreateMatrix(layer->dim,layer->srcDim);
		
		initialiseWithZeroMatrix(layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
		initialiseWithZeroMatrix(layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
		initialiseWithZeroMatrix(layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);

		layer->cgInfo->delbiasUpdate = CreateFloatVec(layer->dim);
		layer->cgInfo->searchDirectionUpdateBias = CreateFloatVec(layer->dim);
		layer->cgInfo->residueUpdateBias = CreateFloatVec(layer->dim);

		initialiseWithZeroFVector(layer->cgInfo->delbiasUpdate,layer->dim,sizeof(float)*layer->dim);
		initialiseWithZeroFVector(layer->cgInfo->residueUpdateBias,layer->dim,sizeof(float)*layer->dim);
		initialiseWithZeroFVector(layer->cgInfo->searchDirectionUpdateBias,layer->dim,sizeof(float)*layer->dim);


		if (useGNMatrix){
			layer->gnInfo = malloc (sizeof(GaussNewtonProductInfo));
			layer->gnInfo->vweights = CreateMatrix(layer->dim,layer->srcDim);
			layer->gnInfo->vbiases = CreateFloatVec(layer->dim);
			layer->gnInfo->Ractivations = CreateMatrix(BATCHSAMPLES,layer->dim);
			
			initialiseWithZeroFVector(layer->gnInfo->vbiases,layer->dim,sizeof(float)*layer->dim);
			initialiseWithZeroMatrix(layer->gnInfo->vweights,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
			initialiseWithZeroMatrix(layer->gnInfo->Ractivations,layer->dim*BATCHSAMPLES,sizeof(float)*layer->dim*BATCHSAMPLES);

		}
	}
}

void reinitLayerFeaMatrices(ADLink anndef){
	LELink layer;
	int i;
	for (i = 0 ; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		if (layer->feaElem->yfeatMat != NULL){
			DisposeMatrix(layer->feaElem->yfeatMat);
			layer->feaElem->yfeatMat = CreateMatrix(BATCHSAMPLES,layer->dim);
			initialiseWithZeroMatrix(layer->feaElem->yfeatMat,layer->dim*BATCHSAMPLES,sizeof(float)*layer->dim*BATCHSAMPLES);
			layer->feaElem->xfeatMat = (layer->src != NULL) ? layer->src->feaElem->yfeatMat : NULL;
		}
	}
}
void reinitLayerErrFeaMatrices(ADLink anndef){
	LELink layer,srcLayer;;
	int i;
	for (i = 0 ; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		if (layer->errElem->dxFeatMat != NULL){
			DisposeMatrix(layer->errElem->dxFeatMat);
			layer->errElem->dxFeatMat = CreateMatrix(BATCHSAMPLES,layer->srcDim); 
			initialiseWithZeroMatrix(layer->errElem->dxFeatMat,layer->srcDim*BATCHSAMPLES,sizeof(float)*layer->srcDim*BATCHSAMPLES);
			if ( i!=0){
			srcLayer = layer->src;
			srcLayer->errElem->dyFeatMat = layer->errElem->dxFeatMat;
			}
		}		
	}
}
void initialiseErrElems(ADLink anndef){
	int i;
	LELink layer,srcLayer;
	for (i = 0; i < anndef->layerNum ;i++){
		layer = anndef->layerList[i];
		layer->errElem = (ERLink) malloc (sizeof(ErrElem));
		layer->errElem->dxFeatMat = CreateMatrix(BATCHSAMPLES,layer->srcDim);
		if ( i!=0){
			srcLayer = layer->src;
			srcLayer->errElem->dyFeatMat = layer->errElem->dxFeatMat;
		}	
	}
	
}
float drand(){	
return (float) rand()/(RAND_MAX);
}
float genrandWeight(float limit){
	return  -limit + (2*limit)*drand()  ;
}

/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(NMatrix *weights,int dim,int srcDim, ActFunKind actfunc){
	int i,j;
	float randm;
	//this is not an efficient way of doing but it allows better readibility
	for (i = 0; i < dim; i++){
		for(j = 0; j < srcDim;j++){
			/* bengio;s proposal for a new tpye of initialisation to ensure 
			the variance of error derivatives are comparable accross layers*/
			if (actfunc == SIGMOID){
				 weights->elems[i*srcDim +j] = 4* genrandWeight(sqrt(6)/sqrt(dim+srcDim+1));
			}else{
				 weights->elems[i*srcDim +j] = genrandWeight(sqrt(6)/sqrt(dim+srcDim+1));
			}
			
		}
	}
	#ifdef CUDA
	SyncHost2Dev(weights->elems,weights->deviceElems,sizeof(float)*dim*srcDim);
	#endif
}
void initialiseLayer(LELink layer,int i, LELink srcLayer){
	int srcDim,numOfElems;
	srcDim = srcLayer != NULL ? srcLayer->dim : inputDim;
	layer->id = i;
	layer->src = srcLayer;
	layer->srcDim = srcDim;
	//setting the layer's role
	if (i == (anndef->layerNum-1)){
		layer->role = OUTPUT;
		layer->dim = targetDim;
	}else{
		layer->role = HIDDEN;
		layer->dim = hidUnitsPerLayer[i];
	}
	layer->actfuncKind = actfunLists[i];
	//for binary classification
	if (layer->role==OUTPUT && layer->dim == 2 && anndef->target==CLASSIFICATION){
		layer->dim = 1; 
	}
	//initialise weights and biases: W is node by feadim Matrix 
	numOfElems = (layer->dim) * (layer->srcDim);
	layer->weights = CreateMatrix(layer->dim,layer->srcDim);
	assert(layer->weights!=NULL);
	
	layer->bias = CreateFloatVec(layer->dim);
	assert(layer->bias!=NULL);
	//initialise weights of outer layer
	if (i ==(numLayers-1)){
		initialiseWithZeroMatrix(layer->weights,layer->dim * layer->srcDim,sizeof(float)*layer->dim * layer->srcDim);
	}else{
		//initialise weights of hidden layers
		//initialiseWithZeroMatrix(layer->weights,layer->dim * layer->srcDim,sizeof(float)*layer->dim * layer->srcDim);
		initialiseWeights(layer->weights,layer->dim,layer->srcDim,layer->actfuncKind);
	}
	initialiseWithZeroFVector(layer->bias,layer->dim, sizeof(float)*layer->dim);
	
	//initialise feaElems
	layer->feaElem = (FELink) malloc(sizeof(FeaElem));
	assert(layer->feaElem!=NULL);
	layer->feaElem->yfeatMat = CreateMatrix(BATCHSAMPLES,layer->dim); 
	initialiseWithZeroMatrix(layer->feaElem->yfeatMat,layer->dim*BATCHSAMPLES, sizeof(float)*layer->dim*BATCHSAMPLES);
	layer->feaElem->xfeatMat = (srcLayer != NULL) ? srcLayer->feaElem->yfeatMat : NULL;
	
	//intialise traininfo and allocating extra memory for setting hooks
	layer->traininfo = (TRLink) malloc(sizeof(TrainInfo) * sizeof(float)*(numOfElems*4));
	assert(layer->traininfo!= NULL);
	layer->traininfo->dwFeatMat = CreateMatrix(layer->dim,layer->srcDim);
	initialiseWithZeroMatrix(layer->traininfo->dwFeatMat,numOfElems, sizeof(float)*numOfElems);

	layer->traininfo->dbFeaMat = CreateFloatVec(layer->dim);
	initialiseWithZeroFVector(layer->traininfo->dbFeaMat,layer->dim, sizeof(float)*layer->dim);
	layer->traininfo->updatedWeightMat = NULL;
	layer->traininfo->updatedBiasMat = NULL;
	layer->traininfo->bestWeightParamsHF = NULL;
	layer->traininfo->bestBiasParamsHF = NULL;

	layer->cgInfo =NULL; 
	layer->gnInfo = NULL;
	layer->rProp = NULL;


	if (momentum > 0) {
		layer->traininfo->updatedWeightMat = CreateMatrix(layer->dim,layer->srcDim);
		layer->traininfo->updatedBiasMat = CreateFloatVec(layer->dim);
		initialiseWithZeroMatrix(layer->traininfo->updatedWeightMat,numOfElems,sizeof(float)*numOfElems);
		initialiseWithZeroFVector(layer->traininfo->updatedBiasMat,layer->dim,sizeof(float)*layer->dim);
	}
}
void initialiseDNN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);
	anndef->target = target;
	anndef->layerNum = numLayers;
	anndef->layerList = (LELink *) malloc (sizeof(LELink)*numLayers);
	assert(anndef->layerList!=NULL);
	/*initialise the seed only once then initialise weights and bias*/
	srand((unsigned int)time(NULL));
	//initilaise layers
	for(i = 0; i<anndef->layerNum; i++){
		anndef->layerList[i] = (LELink) malloc (sizeof(LayerElem));
		assert (anndef->layerList[i]!=NULL);
		if (i == 0 ){
			initialiseLayer(anndef->layerList[i],i, NULL);
		}else{
			initialiseLayer(anndef->layerList[i],i, anndef->layerList[i-1]);
		}	
	}
	anndef->errorfunc = errfunc;
	//initialise ErrElems of layers for back-propagation
	initialiseErrElems(anndef);
	if (doHF) setUpForHF(anndef);
	else if (doRprop) setUpForRprop(anndef);
}
void initialise(){
	printf("initialising DNN\n");
	setBatchSize(trainingDataSetSize);
	initialiseDNN();
	printf("successfully initialised DNN\n");
}

//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents code that creates matrices in host and device(In case of GPU computing)*/
//-------------------------------------------------------------------------------------------------------------------
NMatrix * CreateMatrix(int row , int col){
	NMatrix *matrix;
	matrix  = malloc(sizeof(NMatrix));
	
	matrix->row = row ;
	matrix ->col = col ;
	matrix->elems = malloc(sizeof(float)*row*col);
	#ifdef CUDA
	
	cudaMalloc(&matrix->deviceElems, sizeof(float)*row*col);
	#endif
	return matrix;
	
}
NFVector * CreateFloatVec(int len){
	NFVector * vector;
	vector  = malloc(sizeof(NFVector));
	vector->len = len ;
	vector->elems = malloc(sizeof(float)*len);
	#ifdef CUDA
	cudaMalloc(&vector->deviceElems, sizeof(float)*len);
	#endif
	return  vector ;
}
NIntVector * CreateIntVec( int len){
	NIntVector * vector;
	vector  = malloc(sizeof(NIntVector));
	vector->len = len ;
	vector->elems = malloc(sizeof(int)*len);
	#ifdef CUDA
	cudaMalloc(&vector->deviceElems, sizeof(int)*len);
	#endif
	return vector;
}
void DisposeMatrix(NMatrix *matrix){
	assert (matrix!=NULL);
	if (matrix->elems!=NULL){
		free(matrix->elems) ;
		
	}	
	#ifdef CUDA
	DevDispose(matrix->deviceElems);
	#endif 
	free(matrix);
}
void DisposeFloatVec(NFVector *vector){
	assert (vector!=NULL);
	if (vector->elems!=NULL)free(vector->elems) ;
	#ifdef CUDA
	DevDispose(vector->deviceElems);
	#endif 
	free(vector);
}
void DisposeIntVec(NIntVector *vector){
	assert (vector!=NULL);
	if (vector->elems!=NULL)free(vector->elems) ;
	#ifdef CUDA
	DevDispose(vector->deviceElems);
	#endif 
	free(vector);
}
void initialiseWithZeroMatrix(NMatrix * matrix, int dim,size_t size){
	int i;
	for (i = 0; i< dim;i++){
		matrix->elems[i] = 0;
	}
	#ifdef CUDA
	initialiseDeviceArrayWithZero(matrix->deviceElems,size);
	#endif
}
void initialiseWithZeroFVector(NFVector * matrix, int dim,size_t size){
	int i;
	for (i = 0; i< dim;i++){
		matrix->elems[i] = 0;
	}
	#ifdef CUDA
	initialiseDeviceArrayWithZero(matrix->deviceElems,size);
	#endif
}
void initialiseWithZeroIVector(NIntVector * matrix, int dim,size_t size){
	int i;
	for (i = 0; i< dim;i++){
		matrix->elems[i] = 0;
	}
	#ifdef CUDA
	initialiseDeviceArrayWithZero(matrix->deviceElems,size);
	#endif
}
void setValueInMatrix( NMatrix *matrix, float value){
	#ifdef CUDA
		setValueCUDA(matrix->deviceElems,matrix->row*matrix->col,value);
	#else
	int i;
	for(i = 0 ; i <matrix->row*matrix->col;i++){
		matrix->elems[i] = value;
	}
	#endif

}
void setValueInVec( NFVector *vector, float value){
	#ifdef CUDA
		setValueCUDA(vector->deviceElems,vector->len,value);
	#else
	int i;
	for(i = 0 ; i <vector->len;i++){
		vector->elems[i] = value;
	}
	#endif

}


//-------------------------------------------------------------------------------------------------------------------
/* this section of the code presents library routines that are frequently used*/
//-------------------------------------------------------------------------------------------------------------------
void CopyMatrix (NMatrix *src,int lstartpos, NMatrix *dest,int rstartpos,int dim){

	#ifdef CBLAS
	cblas_scopy(dim, src->elems+lstartpos, 1,dest->elems+rstartpos, 1);
	#else
		#ifdef CUDA
			CopyMatrixOrVecCUDA(src->deviceElems+lstartpos, dest->deviceElems+rstartpos, dim);
		#else
			memcpy(dest,src,sizeof(float)*dim);	
		#endif		
	#endif
}
void CopyVec (NFVector *src,int lstartpos, NFVector *dest,int rstartpos,int dim){

	#ifdef CBLAS
	cblas_scopy(dim, src->elems+lstartpos, 1,dest->elems+rstartpos, 1);
	#else
		#ifdef CUDA
			 CopyMatrixOrVecCUDA(src->deviceElems+lstartpos, dest->deviceElems+rstartpos, dim);
		#else
		memcpy(dest,src,sizeof(float)*dim);	
		#endif	
	#endif
}
void CopyVecToMat (NFVector *src,int lstartpos, NMatrix *dest,int rstartpos,int dim){

	#ifdef CBLAS
	cblas_scopy(dim, src->elems+lstartpos, 1,dest->elems+rstartpos, 1);
	#else
		#ifdef CUDA
			 CopyMatrixOrVecCUDA(src->deviceElems+lstartpos, dest->deviceElems+rstartpos, dim);
		#else
		memcpy(dest,src,sizeof(float)*dim);
		#endif	
	#endif
}
void CopyMatrixToVec (NMatrix *src,int lstartpos, NFVector *dest,int rstartpos,int dim){

	#ifdef CBLAS
	cblas_scopy(dim, src->elems+lstartpos, 1,dest->elems+rstartpos, 1);
	#else
		#ifdef CUDA
			 CopyMatrixOrVecCUDA(src->deviceElems+lstartpos, dest->deviceElems+rstartpos, dim);
		#else
		memcpy(dest,src,sizeof(float)*dim);	
		#endif
	#endif
}

/* this function allows the addition of  two matrices or two vectors*/
void addMatrix(NMatrix * weights, NMatrix * dwFeatMat,int dim, float lambda){
	//blas routine
	#ifdef CBLAS
		cblas_saxpy(dim,lambda,weights->elems,1,dwFeatMat->elems,1);
	#else
		#ifdef CUDA
			AddNSegmentCUDA(weights->deviceElems,dim,dwFeatMat->deviceElems,lambda);
		#else
			int i;
			for (i =0;i<dim;i++){
				dwFeatMat->elems[i] = dwFeatMat->elems[i] + (weights->elems[i] * lambda);
			}
		#endif	
	#endif	
}
void addVec(NFVector * weights, NFVector * dwFeatMat,int dim, float lambda){
	//blas routine
	#ifdef CBLAS
		cblas_saxpy(dim,lambda,weights->elems,1,dwFeatMat->elems,1);
	#else
		#ifdef CUDA
			AddNSegmentCUDA(weights->deviceElems,dim,dwFeatMat->deviceElems,lambda);
		#else
			int i;
			for (i =0;i<dim;i++){
				dwFeatMat->elems[i] = dwFeatMat->elems[i] + (weights->elems[i] * lambda);
			}
		#endif	
	#endif	
}
/*multipy a vector or a matrix with a scalar*/
void scaleMatrix(NMatrix * Mat, float scale ,int dim){
	//blas routine
	#ifdef CBLAS
		cblas_sscal(dim,scale,Mat->elems,1);
	#else
		#ifdef CUDA
			ScaleNSegmentCUDA(dim, scale,Mat->deviceElems);	
		#else
			int i;
			for (i =0;i<dim;i++){
				Mat->elems[i] = Mat->elems[i]*scale;	
			}
		#endif	
	#endif	
}
void scaleVec(NFVector * Mat, float scale ,int dim){
	//blas routine
	#ifdef CBLAS
		cblas_sscal(dim,scale,Mat->elems,1);
	#else
		#ifdef CUDA
			ScaleNSegmentCUDA(dim, scale,Mat->deviceElems);	
		#else
			int i;
			for (i =0;i<dim;i++){
				Mat->elems[i] = Mat->elems[i]*scale;	
			}
		#endif	
	#endif	
}

void subtractMatrix(NMatrix *dyfeat, NMatrix* labelMat, int dim, float lambda){
	//blas routine
	#ifdef CBLAS
		cblas_saxpy(dim,-lambda,labelMat->elems,1,dyfeat->elems,1);
	#else
		#ifdef CUDA
		SubNSegmentCUDA(labelMat->deviceElems,dim,dyfeat->deviceElems,lambda);
		#else
		//CPU version
		int i;
		for (i = 0; i<dim;i++){
			dyfeat->elems[i] = dyfeat->elems[i]-lambda*labelMat->elems[i];
		}
		#endif
	#endif
}

float computeTanh(float x){
	return 2*(computeSigmoid(2*x))-1;
}
float computeSigmoid(float x){
	float result;
	result = 1/(1+ exp(-x));
	return result;
}

float max(float a, float b){
	if (a > b) return a;
	else return b;
}
float min(float a, float b){
	if (a < b) return a;
	else return b;
}

void HNBlasNNgemm(int srcDim, int batchsamples, int dim, float alpha, NMatrix *weights, NMatrix *dyFeatMat, float beta, NMatrix *dxFeatMat){
	#ifdef CBLAS
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, srcDim,batchsamples, dim, alpha, weights->elems, srcDim, dyFeatMat->elems, dim, beta,dxFeatMat->elems,srcDim);
	#else 
		#ifdef CUDA
			HNBlasNNgemmCUDA(srcDim, batchsamples, dim, alpha, weights->deviceElems, dyFeatMat->deviceElems, beta,dxFeatMat->deviceElems);	
		#endif

	#endif
}

void HNBlasNTgemm(int srcDim, int dim,  int batchsamples, float alpha , NMatrix* xfeatMat, NMatrix * dyFeatMat, float beta, NMatrix * dwFeatMat){
	#ifdef CBLAS
		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, srcDim, dim, batchsamples, alpha, xfeatMat->elems, srcDim, dyFeatMat->elems, dim, beta,dwFeatMat->elems, srcDim);
	#else 
		#ifdef CUDA
			HNBlasNTgemmCUDA(srcDim, dim, BATCHSAMPLES, alpha, xfeatMat->deviceElems,dyFeatMat->deviceElems,beta, dwFeatMat->deviceElems);
		#endif
	#endif	
}
 
float computeDotProductMatrix(NMatrix * vectorL, NMatrix * vectorR,int dim){
	float result = 0;
	#ifdef CBLAS
		result = cblas_sdot(dim,vectorL->elems,1,vectorR->elems,1);
		#else
			#ifdef CUDA
			result = computeDotProductCUDA(vectorL->deviceElems,vectorR->deviceElems,dim,result);
			#else
			printf("CPU\n");
			int i;
			for (i = 0 ; i < dim; i++){
				result+= vectorL->elems[i] * vectorR->elems [i];
			}
			#endif
	#endif
	return result;
} 
float computeDotProductVector(NFVector * vectorL, NFVector * vectorR,int dim){
	float result = 0;
	#ifdef CBLAS
		result = cblas_sdot(dim,vectorL->elems,1,vectorR->elems,1);
      	#else
		#ifdef CUDA
			result = computeDotProductCUDA(vectorL->deviceElems,vectorR->deviceElems,dim,result);
		#else
			int i;
			for (i = 0 ; i < dim; i++){
				result+= vectorL->elems[i] * vectorR->elems [i];
			}
		#endif
	#endif
	return result;
} 
//-------------------------------------------------------------------------------------------------------------------
/*this section of the code implements the  forward propgation of a deep neural net **/
//-------------------------------------------------------------------------------------------------------------------

/*computing non-linear activation*/
void computeNonLinearActOfLayer(LELink layer){
	int i,j ;
	float sum;
	float maximum;

	switch(layer->role){
		case HIDDEN:
			switch(layer->actfuncKind){
				case SIGMOID:
				#ifdef CUDA
					ApplySigmoidActCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->feaElem->yfeatMat->deviceElems); 
				#else	
					for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
						layer->feaElem->yfeatMat->elems[i] = computeSigmoid(layer->feaElem->yfeatMat->elems[i]);
					}
				#endif	

				break;
			case TANH:
				#ifdef CUDA
					ApplyTanHActCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->feaElem->yfeatMat->deviceElems); 
				# else	
					for(i = 0; i< layer->dim*BATCHSAMPLES; i++){
						layer->feaElem->yfeatMat->elems[i] = computeTanh(layer->feaElem->yfeatMat->elems[i]);
					}
				#endif

				break;	
			default:
				break;	
			}
			break;
		case OUTPUT:
			switch(layer->actfuncKind){
				case SIGMOID:
					if (layer->dim==1){
						#ifdef CUDA
							ApplySigmoidActCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->feaElem->yfeatMat->deviceElems); 
						#else	
							for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
								layer->feaElem->yfeatMat->elems[i] = computeSigmoid(layer->feaElem->yfeatMat->elems[i]);
							}
						#endif

					}else{
						printf("ERROR to perform binary classification,the number of non-zero output nodes must be <=2");
						exit(0);
					}
					break;	
				case SOFTMAX:
				//softmax activation
					#ifdef CUDA
						ApplySoftmaxActCUDA(layer->feaElem->yfeatMat->deviceElems, BATCHSAMPLES,layer->dim, layer->feaElem->yfeatMat->deviceElems);
					#else
						for (i = 0;i < BATCHSAMPLES;i++){
							sum = 0;
							maximum = 0;
							for (j = 0; j<layer->dim;j++){
								float value = layer->feaElem->yfeatMat->elems[i*layer->dim+j];
								if (value>maximum) maximum = value;
							}
							for (j = 0; j<layer->dim;j++){
								float value = layer->feaElem->yfeatMat->elems[i*layer->dim+j];
								layer->feaElem->yfeatMat->elems[i*layer->dim+j] = exp(value-maximum);
								sum+= exp(value-maximum);
							}
							for (j =0; j<layer->dim;j++){
								layer->feaElem->yfeatMat->elems[i*layer->dim+j]= layer->feaElem->yfeatMat->elems[i*layer->dim+j]/sum ;
							}
						}
					#endif	
					break;
				default:
					break;	
			}
			break;
		default:
			break;
	}
	
}
/* Yfeat is batchSamples by nodeNum matrix(stored as row major)  = X^T(row-major)-batch samples by feaMat * W^T(column major) -feaMat By nodeNum */
void computeLinearActivation(LELink layer){
	#ifdef CBLAS
		int i,off;
		for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
			cblas_scopy(layer->dim, layer->bias->elems, 1, layer->feaElem->yfeatMat->elems + off, 1);
		}
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights->elems, layer->srcDim, layer->feaElem->xfeatMat->elems, layer->srcDim, 1, layer->feaElem->yfeatMat->elems, layer->dim);
	#else
		#ifdef CUDA
		int i,off;
		for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
			CopyNSegmentCUDA(layer->bias->deviceElems, layer->dim, layer->feaElem->yfeatMat->deviceElems+off);
		}
		HNBlasTNgemmCUDA(layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights->deviceElems,layer->feaElem->xfeatMat->deviceElems, 1, layer->feaElem->yfeatMat->deviceElems);
		#endif
	#endif
}

/*forward pass*/
void fwdPassOfANN(ADLink anndef){
	LELink layer;
	int i;
	switch(anndef->target){
			case REGRESSION:
				for (i = 0; i< anndef->layerNum-1;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					computeNonLinearActOfLayer(layer);
				}
				computeLinearActivation(anndef->layerList[anndef->layerNum-1]);
				break;
			case CLASSIFICATION:
				for (i = 0; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					//printMatrix(miniBatchforHF,BATCHSAMPLES,layer->srcDim);
					//printf("DONE \n");
					//printMatrix(layer->feaElem->xfeatMat,BATCHSAMPLES,layer->srcDim);
					computeLinearActivation(layer);
					/*printf("activation of layer %d before >>>>>",i);
					printMatrix(layer->feaElem->yfeatMat,BATCHSAMPLES,layer->dim);
					printf("activation of layer %d after >>>>>",i);*/
					computeNonLinearActOfLayer(layer);
				    //printMatrix(layer->feaElem->yfeatMat,BATCHSAMPLES,layer->dim);
	
				}
				break;	
	}
}

void computeActivationOfOutputLayer(ADLink anndef){
	LELink layer;
	layer = anndef->layerList[anndef->layerNum-1];
	computeLinearActivation(layer);
	computeNonLinearActOfLayer(layer);
}
//------------------------------------------------------------------------------------------------------
/*This section of the code implements the back-propation algorithm  to compute the error derivatives*/
//-------------------------------------------------------------------------------------------------------
void sumColsOfMatrix(NMatrix *dyFeatMat,NFVector *dbFeatMat,int dim,int batchsamples){
	#ifdef CBLAS
		int i;
		float* ones = malloc (sizeof(float)*batchsamples);
		for (i = 0; i<batchsamples;i++){
			ones[i] = 1;
		}
		//multiply node by batchsamples with batchsamples by 1
		#ifdef CBLAS
		cblas_sgemv(CblasColMajor,CblasNoTrans, dim,batchsamples,1,dyFeatMat->elems,dim,ones,1,0,dbFeatMat->elems,1);
		#endif
		free (ones);
	#else
		#ifdef CUDA
		sumColsOfMatrixCUDA(dyFeatMat->deviceElems,dbFeatMat->deviceElems, dim,batchsamples);
		#endif	
	#endif
}

void computeActivationDrv (LELink layer){
	int i;
	switch (layer->actfuncKind){
		case SIGMOID:
			#ifdef CUDA
				computeSigmoidDrvCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->errElem->dyFeatMat->deviceElems);
			#else 				
			//CPU verion
		  		for (i = 0; i<layer->dim*BATCHSAMPLES;i++){
					layer->errElem->dyFeatMat->elems[i] = layer->errElem->dyFeatMat->elems[i] *(layer->feaElem->yfeatMat->elems[i]*(1-layer->feaElem->yfeatMat->elems[i]));
				}
			#endif	

			break;
		case TANH:
			#ifdef CUDA
				computeTanHDrvCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->errElem->dyFeatMat->deviceElems);
			#else 				
			//CPU verion
				for (i = 0; i<layer->dim*BATCHSAMPLES;i++){
					layer->errElem->dyFeatMat->elems[i] = layer->errElem->dyFeatMat->elems[i] *( 1- layer->feaElem->yfeatMat->elems[i]*layer->feaElem->yfeatMat->elems[i]);
				}
			#endif	
			break;
		default:
			break;	
	}
}
/**compute del^2L J where del^2L is the hessian of the cross-entropy softmax with respect to output acivations **/ 
void computeLossHessSoftMax(LELink layer){
	int i,j;
	NFVector * RactivationVec;
	NMatrix * yfeatVec;
	NMatrix * diaP;
	NFVector * result;
	RactivationVec = CreateFloatVec(layer->dim);
	yfeatVec = CreateMatrix(layer->dim,1);
	diaP = CreateMatrix(layer->dim,layer->dim);
	result = CreateFloatVec(layer->dim);
	
	// under the assumption then we call this function after we have already computed gradients then we might need to reset yfeatMat 
	computeLinearActivation(layer);
	computeNonLinearActOfLayer(layer);
	for (i = 0 ; i< BATCHSAMPLES; i++){
		initialiseWithZeroFVector(RactivationVec,layer->dim,sizeof(float)*layer->dim);
		initialiseWithZeroMatrix(yfeatVec,layer->dim,sizeof(float)*layer->dim);
		initialiseWithZeroMatrix(diaP,layer->dim*layer->dim,sizeof(float)*layer->dim*layer->dim);
		initialiseWithZeroFVector(result,layer->dim,sizeof(float)*layer->dim);
		/**extract error directional derivative for a single sample*/ 
		CopyMatrixToVec (layer->gnInfo->Ractivations,i*(layer->dim),RactivationVec,0,layer->dim);
		CopyMatrix (layer->feaElem->yfeatMat,i*(layer->dim),yfeatVec,0,layer->dim);
		//compute dia(yfeaVec - yfeacVec*yfeaVec)'

		HNBlasNNgemm(layer->dim, layer->dim, 1, -1,yfeatVec , yfeatVec, 0, diaP);
		#ifdef CBLAS
		for (j = 0; j<layer->dim;j++){
			diaP->elems[j*(layer->dim+1)] += yfeatVec->elems[j];
		}
		//multiple hessian of loss function of particular sample with Jacobian 
		cblas_sgemv(CblasColMajor,CblasNoTrans,layer->dim,layer->dim,1,diaP->elems,layer->dim,RactivationVec->elems,1,0,result->elems,1);
		#else
			#ifdef CUDA
				const float alpha = 1;
				const float beta = 0;
				AddElementstoDiagonalOfMatrix(diaP->deviceElems,yfeatVec->deviceElems,layer->dim,diaP->deviceElems);
				computeMatVecProductCUDA(layer->dim,layer->dim,alpha,diaP->deviceElems,layer->dim,RactivationVec->deviceElems,1,beta,result->deviceElems,1);
			#endif

		#endif
		CopyVecToMat (result,0,layer->feaElem->yfeatMat,i*(layer->dim),layer->dim);
		
	}
	DisposeFloatVec(result);
	DisposeMatrix(yfeatVec);
	DisposeFloatVec(RactivationVec);
	DisposeMatrix(diaP);

}
/*compute del^2L*J where L can be any convex loss function**/
void computeHessOfLossFunc(LELink outlayer, ADLink anndef){
	switch(anndef->errorfunc){
		case XENT:
			switch (outlayer->actfuncKind){
				case SOFTMAX:
					computeLossHessSoftMax(outlayer);
				default:
					break;
			}
		case SSE :
			break;
	}
}
void calcOutLayerBackwardSignal(LELink layer,ADLink anndef ){
	switch(anndef->errorfunc){
		case (XENT):
			switch(layer->actfuncKind){
				case SIGMOID:
					subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES,1);
					break;
				case SOFTMAX:
					subtractMatrix(layer->feaElem->yfeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES,1);
					break;
				case TANH:
					break;
				default:
					break;
			}
		break;	
		case (SSE):
			subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES,1);	
			break;
	}
}
/**function computes the error derivatives with respect to the weights and biases of the neural net*/
void backPropBatch(ADLink anndef,Boolean doHessVecProd){
	int i;
	LELink layer;
	for (i = (anndef->layerNum-1); i>=0;i--){
		layer = anndef->layerList[i];
		if (layer->role ==OUTPUT){
			if(!doHessVecProd){
				calcOutLayerBackwardSignal(layer,anndef);
			}else{
				if(useGNMatrix){
					printf("computing loss of hessian\n");
					computeHessOfLossFunc(layer,anndef);
					printf("finished computing loss of hessian\n");
				}
			}

			layer->errElem->dyFeatMat = layer->feaElem->yfeatMat;
			/**normalisation because the cost function is mean log-likelihood*/
			scaleMatrix(layer->feaElem->yfeatMat,(float)1/BATCHSAMPLES,BATCHSAMPLES*layer->dim);
			//printMatrix(layer->errElem->dyFeatMat,BATCHSAMPLES, layer->dim);
		}else{
			// from previous iteration dxfeat that is dyfeat now is dE/dZ.. computing dE/da
			computeActivationDrv(layer); 
		}
		
		//compute dxfeatMat: the result  should be an array [ b1 b2..] where b1 is one of dim srcDim
		HNBlasNNgemm(layer->srcDim, BATCHSAMPLES, layer->dim, 1, layer->weights, layer->errElem->dyFeatMat, 0, layer->errElem->dxFeatMat);	
		//compute derivative with respect to weights: the result  should be an array of array of [ n1 n2] where n1 is of length srcDim
		HNBlasNTgemm(layer->srcDim, layer->dim, BATCHSAMPLES, 1, layer->feaElem->xfeatMat,layer->errElem->dyFeatMat,0, layer->traininfo->dwFeatMat);
 		//compute dE/db
		sumColsOfMatrix(layer->errElem->dyFeatMat,layer->traininfo->dbFeaMat, layer->dim, BATCHSAMPLES);
	}
}
//------------------------------------------------------------------------------------------------------
/*This section implements gradient descent learning net**/
//------------------------------------------------------------------------------------------------------
void cacheParameters(ADLink anndef){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		CopyVec (layer->bias,0,layer->bestBias,0,layer->dim);
		CopyMatrix (layer->weights,0,layer->bestweights,0,layer->dim*layer->srcDim);
	}	
	printf("successfully cached best parameters \n");
}

Boolean initialiseParameterCaches(ADLink anndef){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		layer->bestweights = CreateMatrix(layer->dim,layer->srcDim);
		layer->bestBias = CreateFloatVec(layer->dim);
	}	
	printf("successfully intialised caches \n");
	return TRUE;
	 
}
void perfBinClassf(NMatrix *yfeatMat, int *predictions,int dataSize){
	int i;
	for (i = 0; i< dataSize;i++){
		predictions[i] = yfeatMat->elems[i]>0.5 ? 1 :0;
		printf("Predictions %d  %d  and yfeat is %lf and real predict is %d \n",i,predictions[i],yfeatMat->elems[i],validationLabelIdx->elems[i]);
	}
}
/**compute the negative mean log likelihood of the training data**/
float computeLogLikelihood(NMatrix * output, int batchsamples, int dim , NIntVector * labels){
	int i,index;
	float lglikelihood = 0;
	#ifdef CUDA
		SyncDev2Host(output->deviceElems,output->elems,sizeof(float)*batchsamples*dim);
	#endif

	for (i = 0; i < batchsamples ;i++){
		index =  labels->elems[i];
		 lglikelihood += log(output->elems[i*dim+index]);
	}
	return -1*(lglikelihood/batchsamples);
}
/*The function finds the most active node in the output layer for each sample*/
void findMaxElement(float *matrix, int row, int col, int *vec){
	int maxIdx, i, j;
  	float maxVal;
 	for (i = 0; i < row; ++i) {
      maxIdx = 0;
      maxVal = matrix[i * col + 0];
      for (j = 0; j < col; ++j) {
         if (maxVal < matrix[i * col + j]) {
            maxIdx = j;
            maxVal = matrix[i * col + j];
          }
      }
      vec[i] = maxIdx;
   }
}
/** the function calculates the average error*/
float updatateAcc(NIntVector * labels, LELink layer,int dataSize){
	int i, dim;
	float holdingVal;
	int accCount=0;
	dim = layer->dim;
	#ifdef CUDA
		SyncDev2Host(layer->feaElem->yfeatMat->deviceElems,layer->feaElem->yfeatMat->elems,sizeof(float)*dataSize*layer->dim);
	#endif

	if (anndef->target==CLASSIFICATION){
		int *predictions = malloc(sizeof(int)*dataSize);
		if (layer->dim >1){
			findMaxElement(layer->feaElem->yfeatMat->elems,dataSize,dim,predictions);
		}else{
			perfBinClassf(layer->feaElem->yfeatMat,predictions,dataSize);
		}
		for (i = 0; i<dataSize;i++){
			if (abs(predictions[i]-labels->elems[i])>0){
				accCount+=1;
			}	
		}
		free(predictions);
	}
	/*else{
		subtractMatrix(layer->feaElem->yfeatMat, labels, dataSize,1);
		for (i = 0;i<dataSize*layer->dim;i++){
			holdingVal = layer->feaElem->yfeatMat[i];
			accCount+= holdingVal*holdingVal;
		}
	}*/		
		
	return  (float) accCount/dataSize;
}
void updateNeuralNetParams(ADLink anndef, float lrnrate, float momentum, float weightdecay){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		//if we have a regularised error function: 
		if (weightdecay > 0){
			//printf("SHIOULD NOTT REACH HER \n");
			/** here we are computing delE/w + lambda w and then later we add leanring rate -mu(delE/w + lambda w)**/
			addMatrix(layer->weights,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim,weightdecay);
			addVec(layer->bias,layer->traininfo->dbFeaMat,layer->dim,weightdecay);
		}
		if (momentum > 0 ){
			scaleMatrix(layer->traininfo->updatedWeightMat,momentum,layer->dim*layer->srcDim);
			scaleVec(layer->traininfo->updatedBiasMat,momentum,layer->dim);
			addMatrix(layer->traininfo->dwFeatMat,layer->traininfo->updatedWeightMat,layer->dim*layer->srcDim,lrnrate);
			addVec(layer->traininfo->dbFeaMat,layer->traininfo->updatedBiasMat,layer->dim,1-momentum);
			//updating parameters: first we need to descale the lambda from weights and bias
			addMatrix(layer->traininfo->updatedWeightMat,layer->weights,layer->dim*layer->srcDim,1);
			addVec(layer->traininfo->updatedBiasMat,layer->bias,layer->dim,1);
		}else{
			//updating parameters: first we need to descale the lambda from weights and bias
			addMatrix(layer->traininfo->dwFeatMat,layer->weights,layer->dim*layer->srcDim,-1*lrnrate);
			addVec(layer->traininfo->dbFeaMat,layer->bias,layer->dim,-1*lrnrate);
		}
	}
		
}
void updateLearningRate(int currentEpochIdx, float *lrnrate){
	float crtvaldiff;
	if (currentEpochIdx == 0) {
		*lrnrate = initLR;
	}else if (modelSetInfo !=NULL){
		crtvaldiff = (modelSetInfo->crtVal - modelSetInfo->prevCrtVal);

		if (crtvaldiff > 0.000001){
			*lrnrate /=2;
			printf("Learning rate has been halved !! \n");
		}
	}
}

Boolean terminateSchedNotTrue(int currentEpochIdx,float lrnrate){
	if (currentEpochIdx == 0) return TRUE;
	if (currentEpochIdx >=0 && currentEpochIdx >= maxEpochNum){
		printf("Stoppped: max epoc has been reached \n");
		return FALSE;
	}
	if( (lrnrate) < minLR){
		printf("Stopped :lrnrate < minLR\n");
		return FALSE;
	}
	return TRUE; 
}
//------------------------------------------------------------------------------------------------------
/** batch gradient descent training*/
//------------------------------------------------------------------------------------------------------

void TrainDNNGD(){
	printf("TRAINING with BATCH Gradient Optmiser \n");
	
	clock_t start = clock();
	int currentEpochIdx;
	float learningrate;
	float min_validation_error ;
	
	currentEpochIdx = 0;
	learningrate = 0;
	min_validation_error = 1;

	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;

	//array to store the  training errors and validation error
	float * log_likelihood  = malloc(sizeof(float)*(maxEpochNum+1));
	float * zero_one_errorTraining =  malloc(sizeof(float)*(maxEpochNum+1));
	float * zero_one_error =  malloc(sizeof(float)*(maxEpochNum+1));
	
	//compute negative-loglikelihood of training data
	printf("computing the mean negative log-likelihood of training data \n");
	loadDataintoANN(inputData,labelMat);
	fwdPassOfANN(anndef);
	log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);
	modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
	if (modelSetInfo->crtVal < modelSetInfo->bestValue){
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	zero_one_errorTraining[currentEpochIdx] = updatateAcc(labels, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	

	//with the initialisation of weights,checking how well DNN performs on validation data
	printf("computing initial error on validation data\n");
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,NULL);
	fwdPassOfANN(anndef);
	zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	min_validation_error = zero_one_error[currentEpochIdx];
	printf("validation initial error %lf \n ",zero_one_error[currentEpochIdx] );
		
	initialiseParameterCaches(anndef);
	while(terminateSchedNotTrue(currentEpochIdx,learningrate)){
		printf("epoc number %d \n", currentEpochIdx);
		currentEpochIdx+=1;
		updateLearningRate(currentEpochIdx-1,&learningrate);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labelMat);
		printf("computing gradient at epoch %d\n", currentEpochIdx);
		fwdPassOfANN(anndef);
		backPropBatch(anndef,FALSE);
		updateNeuralNetParams(anndef,learningrate,momentum,weightdecay);
		computeActivationOfOutputLayer(anndef);

		log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);	
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
		printf("log_likelihood %lf \n ",modelSetInfo->crtVal );
		zero_one_errorTraining[currentEpochIdx] = updatateAcc(labels, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	

		printf("checking the performance of the neural net on the validation data\n");
		setBatchSize(validationDataSetSize);
		reinitLayerFeaMatrices(anndef);
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,NULL);
		fwdPassOfANN(anndef);
		zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			modelSetInfo->bestValue = modelSetInfo->crtVal;
		}	
		if (zero_one_error[currentEpochIdx]<min_validation_error){
				min_validation_error = zero_one_error[currentEpochIdx];
				cacheParameters(anndef);
		}
		
	}
	clock_t end = clock();
	printf("TRAINING ERROR >>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printArray(log_likelihood,maxEpochNum+1);
	printf("THE  VALIDATION RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printArray(zero_one_error,maxEpochNum+1);
	printf("The zero -one loss on training set >>>>>>>>>>>>>>>>>\n");
	printArray(zero_one_errorTraining,maxEpochNum+1);
	
	printf("The minimum error on the validation data set is %lf percent  and min log_likelihood is %lf \n",min_validation_error*100, modelSetInfo->bestValue);
	free(zero_one_error);
	free(log_likelihood);
	free(zero_one_errorTraining);
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
 	printf("total time taken %lf \n ", seconds);
	
}

//------------------------------------------------------------------------------------------------------
/* batch Rprop training*/
//------------------------------------------------------------------------------------------------------
void IterOfRpropLayer(LELink layer,float cost, float oldcost){
	#ifdef CUDA
	printf("Rprop is CUDA mode>>>>>>>>>>>>\n\n");
	singleIterationOfRprop_plus(layer->traininfo->dwFeatMat->deviceElems, layer->traininfo->updatedWeightMat->deviceElems, layer->rProp->stepWght->deviceElems, layer->rProp->delWght->deviceElems, layer->weights->deviceElems, cost, oldcost, eta_plus,eta_minus,delta_min, delta_max, layer->dim*layer->srcDim);
 	singleIterationOfRprop_plus(layer->traininfo->dbFeaMat->deviceElems, layer->traininfo->updatedBiasMat->deviceElems,layer->rProp->stepBias->deviceElems, layer->rProp->delbias->deviceElems, layer->bias->deviceElems, cost,  oldcost,  eta_plus, eta_minus, delta_min,delta_max,layer->dim);
 
	#else
 	
 	printf("Rprop is CPU mode>>>>>>>>>>>>>>>>>\n\n");
	//printf("Layer id %d >>>>>>\n",layer->id);
	int i ,j;
	float w,b;
	int col = layer->weights->col;
	int row = layer->weights->row;
	//printf("CACHED GRADIENT \n");
	//printMatrix(layer->traininfo->updatedWeightMat,row,col);

	//printf(" GRADIENT \n");
	//printMatrix(layer->traininfo->dwFeatMat,row,col);


	for (i =0 ; i<row;i++) {
		for (j =0; j < col;j++){
			w = layer->traininfo->dwFeatMat->elems[i*col +j] * layer->traininfo->updatedWeightMat->elems[i*col+j];
			if (w > 0){
				layer->rProp->stepWght->elems[i*col+j] = min (layer->rProp->stepWght->elems[i*col+j]*eta_plus,delta_max);
				layer->rProp->delWght->elems[i*col+j] = layer->traininfo->dwFeatMat->elems[i*col+j] >0 ? -1 * layer->rProp->stepWght->elems[i*col +j] :layer->rProp->stepWght->elems[i*col +j]; 
				layer->weights->elems[i*col +j] += layer->rProp->delWght->elems[i*col+j];
			}
			else if (w < 0){
				layer->rProp->stepWght->elems[i*col+j] = max (layer->rProp->stepWght->elems[i*col+j]*eta_minus,delta_min);
				//if(cost > oldcost )
				layer->weights->elems[i*col +j] = layer->weights->elems[i*col +j]- layer->rProp->delWght->elems[i*col+j];
				layer->traininfo->dwFeatMat->elems[i*col +j] = 0;	
			}else{
				layer->rProp->delWght->elems[i*col+j] = layer->traininfo->dwFeatMat->elems[i*col+j] >0 ? -1 * layer->rProp->stepWght->elems[i*col +j] :layer->rProp->stepWght->elems[i*col +j]; 
				layer->rProp->delWght->elems[i*col+j] = layer->traininfo->dwFeatMat->elems[i*col+j] == 0 ? 0 :layer->rProp->delWght->elems[i*col+j];
				layer->weights->elems[i*col +j] += layer->rProp->delWght->elems[i*col+j];
	
			}
		}

		//updating bias
		b = layer->traininfo->dbFeaMat->elems[i] * layer->traininfo->updatedBiasMat->elems[i];
		if ( b > 0){
			layer->rProp->stepBias->elems[i] = min (layer->rProp->stepBias->elems[i]*eta_plus,delta_max);
			layer->rProp->delbias->elems[i] = layer->traininfo->dbFeaMat->elems[i] >0 ? -1 * layer->rProp->stepBias->elems[i] :layer->rProp->stepBias->elems[i]; 
			layer->bias->elems[i] += layer->rProp->delbias->elems[i];		
		}else if (b < 0){
			layer->rProp->stepBias->elems[i] = max (layer->rProp->stepBias->elems[i]*eta_minus,delta_min);
			if (cost >oldcost) layer->bias->elems[i] =  layer->bias->elems[i]- layer->rProp->delbias->elems[i];
			layer->traininfo->dbFeaMat->elems[i] = 0;
		}else {
			layer->rProp->delbias->elems[i] = layer->traininfo->dbFeaMat->elems[i] >0 ? -1 * layer->rProp->stepBias->elems[i] :  layer->rProp->stepBias->elems[i];
			layer->rProp->delbias->elems[i] = layer->traininfo->dbFeaMat->elems[i] ==0 ? 0 : layer->rProp->delbias->elems[i] ;
			layer->bias->elems[i] += layer->rProp->delbias->elems[i];		
		}
	}
	#endif
	//printf(" Updated Weights \n");
	//printMatrix(layer->weights,row,col);
	
}

void updateWithRprop(ADLink anndef,float cost, float oldcost){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		IterOfRpropLayer(layer,cost,oldcost);
	}
}
void TrainDNNRprop(){
	printf("TRAINING with Rprop Optmiser \n");
	
	clock_t start = clock();
	int currentEpochIdx;
	float min_validation_error ;
	float cost ;
	float oldcost;
	
	currentEpochIdx = 0;
	min_validation_error = 1;

	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;

	//array to store the  training errors and validation error
	float * log_likelihood  = malloc(sizeof(float)*(maxEpochNum+1));
	float * zero_one_errorTraining =  malloc(sizeof(float)*(maxEpochNum+1));
	float * zero_one_error =  malloc(sizeof(float)*(maxEpochNum+1));
	
	//compute negative-loglikelihood of training data
	printf("computing initial the mean negative log-likelihood of training data \n");
	loadDataintoANN(inputData,labelMat);
	fwdPassOfANN(anndef);
	log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);
	modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
	printf("initial the mean negative log-likelihood of training data %f \n",modelSetInfo->crtVal);
	
	if (modelSetInfo->crtVal < modelSetInfo->bestValue){
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	zero_one_errorTraining[currentEpochIdx] = updatateAcc(labels, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	

	//with the initialisation of weights,checking how well DNN performs on validation data
	printf("computing initial error on validation data\n");
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,NULL);
	fwdPassOfANN(anndef);
	zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	min_validation_error = zero_one_error[currentEpochIdx];
	printf("validation error %lf \n ",zero_one_error[currentEpochIdx] );
		
	initialiseParameterCaches(anndef);
	while(currentEpochIdx < maxEpochNum){
		printf("epoc number %d \n", currentEpochIdx);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labelMat);
		fwdPassOfANN(anndef);
		if (currentEpochIdx==0) cost = log_likelihood[currentEpochIdx];
		else {
			oldcost =  cost;
			cost = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);
			log_likelihood[currentEpochIdx] = cost;
			zero_one_errorTraining[currentEpochIdx] = updatateAcc(labels, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	
		}
		printf("computing gradient at epoch %d\n", currentEpochIdx);
		backPropBatch(anndef,FALSE);
		printf("computed gradient at epoch %d\n", currentEpochIdx);
		
		updateWithRprop(anndef,cost,oldcost);
		//cache current gradient
		accumulateGradientsofANN(anndef);
		printf("caching gradient at epoch %d\n", currentEpochIdx);
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
		
		printf("checking the performance of the neural net on the validation data\n");
		setBatchSize(validationDataSetSize);
		reinitLayerFeaMatrices(anndef);
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,NULL);
		fwdPassOfANN(anndef);
		zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			modelSetInfo->bestValue = modelSetInfo->crtVal;
		}	
		if (zero_one_error[currentEpochIdx]<min_validation_error){
				min_validation_error = zero_one_error[currentEpochIdx];
				cacheParameters(anndef);
		}
		currentEpochIdx+=1;
		
		
	}
	clock_t end = clock();
	printf("TRAINING ERROR >>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printArray(log_likelihood,maxEpochNum+1);
	printf("THE  VALIDATION RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printArray(zero_one_error,maxEpochNum+1);
	printf("The zero -one loss on training set >>>>>>>>>>>>>>>>>\n");
	printArray(zero_one_errorTraining,maxEpochNum+1);
	
	printf("The minimum error on the validation data set is %lf percent  and min log_likelihood is %lf \n",min_validation_error*100, modelSetInfo->bestValue);
	free(zero_one_error);
	free(log_likelihood);
	free(zero_one_errorTraining);
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
 	printf("total time taken %lf \n ", seconds);
}




//------------------------------------------------------------------------------------------------------
/**this segment of the code is reponsible for accumulating the gradients **/
//------------------------------------------------------------------------------------------------------
void setHook(Ptr m, Ptr ptr,int incr){
	Ptr *p;
	p = (Ptr *) m; 
	p -= incr; 
  *p = ptr;
 }

Ptr getHook(Ptr m,int incr){
	Ptr *p;
  p = (Ptr *) m; p -=incr; return *p;
}
/** accumulate direction of steepest descent */
void accumulateLayerGradient(LELink layer,float weight){
	assert(layer->traininfo->updatedBiasMat != NULL);
	assert(layer->traininfo->updatedWeightMat != NULL);
	CopyMatrix (layer->traininfo->dwFeatMat,0, layer->traininfo->updatedWeightMat,0,layer->srcDim*layer->dim);
	CopyVec (layer->traininfo->dbFeaMat,0, layer->traininfo->updatedBiasMat,0,layer->dim);
	if (doHF){
		scaleMatrix(layer->traininfo->updatedWeightMat, -1 ,layer->dim*layer->srcDim);
		scaleVec(layer->traininfo->updatedBiasMat, -1 ,layer->dim);
	}
}
void accumulateGradientsofANN(ADLink anndef){
	int i;
	LELink layer;
	for (i = 0; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		accumulateLayerGradient(layer,1);
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
//additional functions to check CG sub-routines just in case 
//----------------------------------------------------------------------------------------------------------------------------------
void normOfWeights(ADLink anndef){
    int i;
    float weightsum = 0;
    float biasSum = 0 ;
    LELink layer;
    for (i = 0; i<anndef->layerNum;i++){
        layer = anndef->layerList[i];
		weightsum+= computeDotProductMatrix(layer->weights,layer->weights,layer->dim*layer->srcDim);
        biasSum +=  computeDotProductVector(layer->bias,layer->bias,layer->dim);
     }
    printf( " The norm  of weights is %lf %lf  %lf \n",weightsum, biasSum,weightsum+biasSum);
    
    
}
void normaliseSearchDirections(ADLink anndef){
	int i; 
	LELink layer;
	float dotProduct[] ={ 0,0};
	//computeSearchDirDotProduct(anndef,dotProduct);
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		scaleMatrix(layer->cgInfo->searchDirectionUpdateWeights, 1/sqrt(dotProduct[0]) ,layer->dim*layer->srcDim);
		scaleVec(layer->cgInfo->searchDirectionUpdateBias, 1/sqrt(dotProduct[1]) ,layer->dim);
	}	
}
void normaliseResidueDirections(ADLink anndef, float* magnitudeOfGradient){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		scaleMatrix(layer->cgInfo->residueUpdateWeights, 1/sqrt(magnitudeOfGradient[0]) ,layer->dim*layer->srcDim);
		scaleVec(layer->cgInfo->residueUpdateBias, 1/sqrt(magnitudeOfGradient[1]) ,layer->dim);
	}	

}
void computeNormOfGradient(ADLink anndef){
	int i; 
	float weightsum = 0;
	float biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum+= computeDotProductMatrix(layer->traininfo->dwFeatMat,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->traininfo->dbFeaMat,layer->traininfo->dbFeaMat,layer->dim);
		
	}
	printf( " The norm  of gradient is %lf %lf  %lf \n",weightsum, biasSum,weightsum+biasSum);
}
void computeNormOfAccuGradient(ADLink anndef){
	int i; 
	float weightsum = 0;
	float biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum+= computeDotProductMatrix(layer->traininfo->updatedWeightMat,layer->traininfo->updatedWeightMat,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->traininfo->updatedBiasMat,layer->traininfo->updatedBiasMat,layer->dim);
	}
	printf( " The norm  of  accumulated gradient is %lf %lf  %lf \n",weightsum, biasSum,weightsum+biasSum);
}

void normOfVweights(ADLink anndef){
	int i; 
	float weightsum = 0;
	float biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum+= computeDotProductMatrix(layer->gnInfo->vweights,layer->gnInfo->vweights,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->gnInfo->vbiases,layer->gnInfo->vbiases,layer->dim);
	}
	printf( " The norm  of veights direction is %lf %lf  %lf \n",weightsum, biasSum,weightsum+biasSum);

}

void printGVoutput(ADLink anndef){
	printf("PRINTING JV >>>>>>>>>>>>>>>>>>>>>>\n");
	printf("JV of  layer\n");
	printMatrix(anndef->layerList[1]->gnInfo->Ractivations,BATCHSAMPLES,anndef->layerList[1]->dim);


	printf("PRINTING HJV >>>>>>>>>>>>>>>>>>>>>>\n");	
	printMatrix(anndef->layerList[1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[1]->dim);


	printf("de/dw of hidden layer\n");
	printMatrix(anndef->layerList[0]->traininfo->dwFeatMat,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	printf("de/db of hidden layer\n");
	printVector(anndef->layerList[0]->traininfo->dbFeaMat,10);

	printf("de/dw of output layer\n");
	printMatrix(anndef->layerList[1]->traininfo->dwFeatMat,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	printf("de/db of output layer\n");
	printVector(anndef->layerList[1]->traininfo->dbFeaMat,10);

}
void computeSearchDirDotProduct(ADLink anndef){
	int i; 
	float weightsum = 0;
	float biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum+=computeDotProductMatrix(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->cgInfo->searchDirectionUpdateBias,layer->cgInfo->searchDirectionUpdateBias,layer->dim);
	}
	printf( " The norm  of search direction is %lf %lf  %lf \n",weightsum, biasSum,weightsum+biasSum);
}

void displaySearchDirection(ADLink anndef){
	//Assuming we are using Blas
	printf("search direction layer 0\n");
	printMatrix(anndef->layerList[0]->cgInfo->searchDirectionUpdateWeights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	printf("search direction layer 0\n");
	printVector(anndef->layerList[0]->cgInfo->searchDirectionUpdateBias,10);

	printf("search direction layer 1\n");
	printMatrix(anndef->layerList[1]->cgInfo->searchDirectionUpdateWeights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	printf("search direction layer 1\n");
	printVector(anndef->layerList[1]->cgInfo->searchDirectionUpdateBias,10);

}
void displayResidueDirection(ADLink anndef){
	//Assuming we are using Blas
	
	printf(" residue direction layer 0\n");
	printMatrix(anndef->layerList[0]->cgInfo->residueUpdateWeights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	printf("reisdue direction layer 0\n");
	printVector(anndef->layerList[0]->cgInfo->residueUpdateBias,10);

	printf("residue direction layer 1\n");
	printMatrix(anndef->layerList[1]->cgInfo->residueUpdateWeights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	printf("residue direction layer 1\n");
	printVector(anndef->layerList[1]->cgInfo->residueUpdateBias,10);

}
void displayVweights(ADLink anndef){
	//Assuming we are using Blas
	printf(" vweights layer 0\n");
	printMatrix(anndef->layerList[0]->gnInfo->vweights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	printf(" vbaias layer 0\n");
	printVector(anndef->layerList[0]->gnInfo->vbiases,10);

	printf("vweights layer 1\n");
	printMatrix(anndef->layerList[1]->gnInfo->vweights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	printf("vweights  layer 1\n");
	printVector(anndef->layerList[1]->gnInfo->vbiases,10);

}
void displaydelWs(ADLink anndef){
	//Assuming we are using Blas
	
	printf(" delWs 0\n");
	printMatrix(anndef->layerList[0]->cgInfo->delweightsUpdate,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	printf(" delB layer 0\n");
	printVector(anndef->layerList[0]->cgInfo->delbiasUpdate,10);

	printf("delW layer 1\n");
	printMatrix(anndef->layerList[1]->cgInfo->delweightsUpdate,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	printf(" delB  layer 1\n");
	printVector(anndef->layerList[1]->cgInfo->delbiasUpdate,10);

}

void normofGV(ADLink anndef){
	int i;
	float weightsum,biasSum;
	weightsum =0;
	biasSum =0; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum +=computeDotProductMatrix(layer->traininfo->dwFeatMat,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->traininfo->dbFeaMat,layer->traininfo->dbFeaMat,layer->dim);
	}
	printf("the norm of GV is %lf %lf  %lf \n", weightsum, biasSum,weightsum+biasSum);
}
void normofDELW(ADLink anndef){
	int i;
	float weightsum,biasSum;
	weightsum =0;
	biasSum =0;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum +=computeDotProductMatrix(layer->cgInfo->delweightsUpdate,layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->cgInfo->delbiasUpdate,layer->cgInfo->delbiasUpdate,layer->dim);
	}
	printf("the norm of del W is %lf  \n", weightsum+biasSum);
}
//------------------------------------------------------------------------------------------------------
/* This section of the code implements HF training*/
//------------------------------------------------------------------------------------------------------
void resetdelWeights(ADLink anndef){
	int i;
	LELink layer;
	for (i =0 ; i <(anndef->layerNum); i++){
		layer = anndef->layerList[i];
		initialiseWithZeroMatrix(layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
		initialiseWithZeroFVector(layer->cgInfo->delbiasUpdate,layer->dim,sizeof(float)*layer->dim);		
	}
}

void getBestParamsCG(ADLink anndef){
	int i; 
	LELink layer;
	for (i =0 ; i < anndef->layerNum; i++){
		layer = anndef->layerList[i];
		CopyMatrix (layer->traininfo->bestWeightParamsHF,0,layer->weights,0,layer->dim*layer->srcDim);
		CopyVec (layer->traininfo->bestBiasParamsHF,0,layer->bias,0,layer->dim);
		
	}

}
void cacheParamsCG(ADLink anndef){
	int i; 
	LELink layer;
	for (i =0 ; i < anndef->layerNum; i++){
		layer = anndef->layerList[i];
		//printf("best weight so for layer %d \n" ,i);
		//printMatrix(layer->weights,layer->dim,layer->srcDim);
		CopyMatrix (layer->weights,0,layer->traininfo->bestWeightParamsHF,0,layer->dim*layer->srcDim);
		CopyVec (layer->bias,0,layer->traininfo->bestBiasParamsHF,0,layer->dim);
	}

}
void updateNeuralNetParamsHF( ADLink anndef){
	int i;
	LELink layer;
	for (i =0 ; i <(anndef->layerNum); i++){
		layer = anndef->layerList[i];
		addMatrix(layer->cgInfo->delweightsUpdate,layer->weights,layer->dim*layer->srcDim,1);
		addVec(layer->cgInfo->delbiasUpdate,layer->bias,layer->dim,1);
	}
}
void backtrackNeuralNetParamsCG(ADLink anndef){
	int i;
	LELink layer;
	for (i =0 ; i < anndef->layerNum; i++){
		layer = anndef->layerList[i];
		addMatrix(layer->cgInfo->delweightsUpdate,layer->weights,layer->dim*layer->srcDim,-1);
		addVec(layer->cgInfo->delbiasUpdate,layer->bias,layer->dim,-1);
	}
}


//-----------------------------------------------------------------------------------
/**This section of the code implements the small sub routinesof the  conjugate Gradient algorithm **/
void updateParameterDirection(ADLink anndef,float beta){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		//first we set p_K+1 = beta  p_k then we set p_k+1+=r_k+1
		scaleMatrix(layer->cgInfo->searchDirectionUpdateWeights,beta,layer->dim*layer->srcDim);
		addMatrix(layer->cgInfo->residueUpdateWeights,layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim,1);
		
		scaleVec(layer->cgInfo->searchDirectionUpdateBias,beta,layer->dim);
		addVec(layer->cgInfo->residueUpdateBias,layer->cgInfo->searchDirectionUpdateBias,layer->dim,1);
	}	
}

void updateResidue(ADLink anndef){
	int i; 
	LELink layer;
	//set del w
	setSearchDirectionCG(anndef,FALSE);
	//displayVweights(anndef);
	// compute Jv i.e 
	computeDirectionalErrDerivativeofANN(anndef);
	//compute J^T del L^2 J v i.e A del_w_k+1
	backPropBatch(anndef,TRUE);
	addTikhonovDamping(anndef);
	printf("norm of G del wk \n");
	normofGV(anndef);
	printf("CHECKING RESIDUE UPDATE _________________________\n");
	//residue r_k+1 = b - A del w_k+1
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		CopyMatrix (layer->traininfo->dwFeatMat,0,layer->cgInfo->residueUpdateWeights,0,layer->dim*layer->srcDim);
		CopyVec (layer->traininfo->dbFeaMat,0,layer->cgInfo->residueUpdateBias,0,layer->dim);
		scaleMatrix(layer->cgInfo->residueUpdateWeights, -1 ,layer->dim*layer->srcDim);
		scaleVec(layer->cgInfo->residueUpdateBias, -1 ,layer->dim);
		addMatrix(layer->traininfo->updatedWeightMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim,1);
		addVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim,1);
	}	
}
float  computeQuadfun( ADLink anndef){
	float obj_fun = 0;
	float weightsum =0;
	float biasSum = 0;
	int i; 
	LELink layer;
	//compute (b+r_K)+x *-0.5
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		addMatrix(layer->traininfo->updatedWeightMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim,1);
		addVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim,1);
		
		biasSum += computeDotProductVector(layer->cgInfo->residueUpdateBias,layer->cgInfo->residueUpdateBias,layer->dim);
		weightsum+= computeDotProductMatrix(layer->cgInfo->residueUpdateWeights,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		
		addMatrix(layer->traininfo->updatedWeightMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim,-1);
		addVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim,-1);
	}
	obj_fun = (weightsum+biasSum)*-0.5;
	printf("the objective function is %lf >>>>\n", obj_fun);
	return obj_fun;

}	
void updatedelParameters(float alpha){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		addMatrix(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim,alpha);
		addVec(layer->cgInfo->searchDirectionUpdateBias,layer->cgInfo->delbiasUpdate,layer->dim,alpha);
	}		
}
void computeSearchDirMatrixProduct( ADLink anndef,float * searchVecMatrixVecProductResult){
	int i; 
	float weightsum = 0;
	float biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum+= computeDotProductMatrix(layer->cgInfo->searchDirectionUpdateWeights,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim);
		biasSum +=  computeDotProductVector(layer->cgInfo->searchDirectionUpdateBias,layer->traininfo->dbFeaMat,layer->dim);
	}
	printf("values of p_k * A  p_k %lf n",weightsum + biasSum);	
 	*searchVecMatrixVecProductResult = weightsum + biasSum;
}
void computeResidueDotProduct(ADLink anndef, float * residueDotProductResult){
	int i; 
	float weightsum = 0;
	float biasSum = 0;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		weightsum+= computeDotProductMatrix(layer->cgInfo->residueUpdateWeights,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		biasSum += computeDotProductVector(layer->cgInfo->residueUpdateBias,layer->cgInfo->residueUpdateBias,layer->dim);
	}
	*residueDotProductResult = weightsum + biasSum;
	
}
void addTikhonovDamping(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i < anndef->layerNum; i++){
		layer = anndef->layerList[i]; 
		addMatrix(layer->gnInfo->vweights,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim,weightdecay);
		addVec(layer->gnInfo->vbiases,layer->traininfo->dbFeaMat,layer->dim,weightdecay);
	}		
}
//-----------------------------------------------------------------------------------
/* the following routines compute the directional derivative using forward differentiation*/

/**this function computes R(z) = h'(a)R(a)  : we assume h'a is computed during computation of gradient**/
void updateRactivations(LELink layer){
	//CPU Version
	int i;
	switch (layer->actfuncKind){
		case SIGMOID:
			#ifdef CUDA
			computeSigmoidDrvCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->gnInfo->Ractivations->deviceElems);
			#else
			for (i = 0; i < layer->dim*BATCHSAMPLES; i++){
				layer->gnInfo->Ractivations->elems[i] = layer->gnInfo->Ractivations->elems[i]* (layer->feaElem->yfeatMat->elems[i])*(1-layer->feaElem->yfeatMat->elems[i]);
				}
			#endif	
			break;
		case TANH:
			#ifdef CUDA
				computeTanHDrvCUDA(layer->feaElem->yfeatMat->deviceElems,layer->dim*BATCHSAMPLES, layer->gnInfo->Ractivations->deviceElems);
			#else
			for (i = 0; i < layer->dim*BATCHSAMPLES; i++){
				layer->gnInfo->Ractivations->elems[i] = layer->gnInfo->Ractivations->elems[i]* (1 -layer->feaElem->yfeatMat->elems[i]*layer->feaElem->yfeatMat->elems[i]) ; 
			}
			#endif	
			break;

		default :
			break;
		}
		
	
}
/** this function compute \sum wji R(zi)-previous layer and adds it to R(zj)**/
void computeRactivations(LELink layer){
	int i,off;
	NMatrix * buffer;
	buffer = CreateMatrix( BATCHSAMPLES,layer->dim);
	initialiseWithZeroMatrix(buffer, BATCHSAMPLES*layer->dim,sizeof(float)*BATCHSAMPLES*layer->dim);
	#ifdef CBLAS
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights->elems, layer->srcDim, layer->src->gnInfo->Ractivations->elems, layer->srcDim, 1.0, buffer->elems, layer->dim);
	#else
		#ifdef CUDA
		HNBlasTNgemmCUDA(layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights->deviceElems,layer->src->gnInfo->Ractivations->deviceElems, 1, buffer->deviceElems);
		#endif
	
	#endif

	addMatrix(buffer, layer->gnInfo->Ractivations,BATCHSAMPLES*layer->dim, 1);
	DisposeMatrix(buffer);
}

/**this function computes sum vji xi */
void computeVweightsProjection(LELink layer){
	int i,off;
	for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
		CopyVecToMat (layer->gnInfo->vbiases,0,layer->gnInfo->Ractivations ,off ,layer->dim);
	}
	#ifdef CBLAS
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->gnInfo->vweights->elems, layer->srcDim, layer->feaElem->xfeatMat->elems, layer->srcDim, 1, layer->gnInfo->Ractivations->elems, layer->dim);
	#else
		#ifdef CUDA
		HNBlasTNgemmCUDA(layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->gnInfo->vweights->deviceElems,layer->feaElem->xfeatMat->deviceElems, 1,  layer->gnInfo->Ractivations->deviceElems);
		#endif
	#endif	
}

void computeDirectionalErrDrvOfLayer(LELink layer, int layerid){
	if (layerid == 0){
		computeVweightsProjection(layer);
		/** compute R(z) = h'(a)* R(a)**/
		//note h'(a) is already computed during backprop of gradients;
		updateRactivations(layer);
		}else{
		/**R(zk) = sum vkz zj**/;
		computeVweightsProjection(layer);
		//printf("printing R activations after linear map layer id %d \n",layer->id);
		/** R(zk) += sum wkj Rj **/
		computeRactivations(layer);
		if (layer->role != OUTPUT){
			updateRactivations(layer);
		}
	}
}
void computeDirectionalErrDerivativeofANN(ADLink anndef){
	int i;
	LELink  layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		computeDirectionalErrDrvOfLayer(layer,i);
	}
}
/**given a vector in parameteric space, this function copies the the segment of the vector that aligns with the parameters of the given layer*/
void setParameterDirections(NMatrix* weights, NFVector * bias, LELink layer){
	assert(layer->gnInfo !=NULL);
	initialiseWithZeroFVector(layer->gnInfo->vbiases,layer->dim,sizeof(float)*layer->dim);
	initialiseWithZeroMatrix(layer->gnInfo->vweights,layer->dim*layer->srcDim,sizeof(float)*layer->dim*layer->srcDim);
	CopyMatrix (weights,0,layer->gnInfo->vweights,0,layer->dim*layer->srcDim);
	CopyVec (bias,0,layer->gnInfo->vbiases,0,layer->dim);
}
void setSearchDirectionCG(ADLink anndef, Boolean Parameter){
	int i; 
	LELink layer;

	for (i = 0; i < anndef->layerNum; i++){
		layer = anndef->layerList[i]; 
		if(Parameter){
			setParameterDirections(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->searchDirectionUpdateBias,layer);
		}else{
			setParameterDirections(layer->cgInfo->delweightsUpdate,layer->cgInfo->delbiasUpdate,layer);
		}
	}
}
//-----------------------------------------------------------------------------------

void initialiseResidueaAndSearchDirection(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];

		CopyMatrix (layer->traininfo->updatedWeightMat, 0,layer->cgInfo->residueUpdateWeights,0,layer->dim*layer->srcDim);
		CopyMatrix (layer->traininfo->updatedWeightMat, 0,layer->cgInfo->searchDirectionUpdateWeights,0,layer->dim*layer->srcDim);
		
		CopyVec (layer->traininfo->updatedBiasMat,0,layer->cgInfo->residueUpdateBias,0,layer->dim);
		CopyVec (layer->traininfo->updatedBiasMat,0,layer->cgInfo->searchDirectionUpdateBias,0,layer->dim);
	}
}
void runConjugateGradient(){
	int numberofRuns = 0;
	float residueDotProductResult = 0;
	float prevresidueDotProductResult = 0;
	float searchVecMatrixVecProductResult = 0;
	float alpha = 0;
	float beta = 0;
	float cost = 0;
	float oldcost =0;
	float minCost = DBL_MAX;
	float obj_fun_value = DBL_MAX;
	float old_fun_value = DBL_MAX;
	int num  = maxNumOfCGruns/5;
	float *costlist = malloc(sizeof(float)*num);
	memset(costlist,0,sizeof(float)*num);
	int counter = 0;
	int listcounter = 0;

	initialiseResidueaAndSearchDirection(anndef);
	
	computeActivationOfOutputLayer(anndef);
	cost = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[anndef->layerNum-1]->dim,minBatchLabels);	
	
	printf("Initial cost on minibatch %lf\n ",cost);
	while(numberofRuns < maxNumOfCGruns){
		printf("RUN  %d >>>>>>>>>>>>>>>>>>>>>\n",numberofRuns);
		//----Compute Gauss Newton Matrix product	
		//set v
		setSearchDirectionCG(anndef,TRUE);
		// compute Jv i.e 
		computeDirectionalErrDerivativeofANN(anndef);
		//compute J^T del L^2 J v i.e A p_k
		printf("Successfully computing Jv\n ");
	
		backPropBatch(anndef,TRUE);
		addTikhonovDamping(anndef);
		normofGV(anndef);
		//normOfVweights(anndef);
		computeSearchDirDotProduct(anndef);
		//compute r_k^T r_k
		computeResidueDotProduct(anndef, &residueDotProductResult);
		//compute p_k^T A p_k
		computeSearchDirMatrixProduct(anndef,&searchVecMatrixVecProductResult);
		alpha = residueDotProductResult/searchVecMatrixVecProductResult;
		
		if (numberofRuns >0){
			old_fun_value = obj_fun_value;
		}
		obj_fun_value = computeQuadfun(anndef);
		if (numberofRuns > 0) printf("the difference old and new func value %lf \n ", old_fun_value - obj_fun_value);
		updatedelParameters(alpha);
		
		if (counter == 5){
			oldcost = cost;
			printf("old cost %lf \n", oldcost);
			updateNeuralNetParamsHF(anndef);
			fwdPassOfANN(anndef);
			printf("fwdpass successful \n");
			cost = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[anndef->layerNum-1]->dim,minBatchLabels);	
			costlist[listcounter] = cost;
			listcounter++;
			if (cost < minCost){
				printf("starting parameter cache\n");
				cacheParamsCG(anndef);
				printf(" parameter cache successful \n");
				minCost = cost;
			}
			printf("cost is %lf \n ", cost);
			backtrackNeuralNetParamsCG(anndef);
			if(cost > oldcost || (old_fun_value - obj_fun_value)<0.000001){
				break;
			}
			fwdPassOfANN(anndef);
			counter = 0;
		}
		
		//displaydelWs(anndef);
		normofDELW(anndef);
		printf("residue norm and pkApk  %lf %lf \n",residueDotProductResult,searchVecMatrixVecProductResult);
		updateResidue(anndef);
		prevresidueDotProductResult = residueDotProductResult;
		//compute r_(k+1)^T r_(k+1)
		computeResidueDotProduct(anndef, &residueDotProductResult);
		printf("the new residue norm %lf \n ", residueDotProductResult);
		beta = residueDotProductResult/prevresidueDotProductResult;
		//compute p_(k+1) = r_k+1 + beta p_k
		updateParameterDirection(anndef,beta);
		computeSearchDirDotProduct(anndef);
		numberofRuns+=1;
		int i;
		printf("alpha  and beta  is %lf %lf \n", alpha,beta);
		counter+=1;
	}
	resetdelWeights(anndef);
	normofDELW(anndef);
	getBestParamsCG(anndef);
	//fwdPassOfANN(anndef);
	//cost = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,minBatchLabels);	
			
	printf("the actual minCOst on the mini-batch is %lf\n ",minCost);
	printf("List of costs \n");
	printArray(costlist,listcounter);
	free(costlist);
}
void TrainDNNHF(){
	printf("TRAINING with Hessian Free Optmiser \n");
	clock_t start = clock();
	int currentEpochIdx;
	float min_validation_error ;
	
	currentEpochIdx = 0;
	min_validation_error = 1;

	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;

	//array to store the  training errors and validation error
	float * log_likelihood  = malloc(sizeof(float)*(maxEpochNum+1));
	float * zero_one_errorTraining =  malloc(sizeof(float)*(maxEpochNum+1));
	float * zero_one_error =  malloc(sizeof(float)*(maxEpochNum+1));
	

	//compute negative-loglikelihood of training data
	printf("computing initial the mean negative log-likelihood of training data \n");
	loadDataintoANN(inputData,labelMat);
	fwdPassOfANN(anndef);
	log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[anndef->layerNum-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);
	modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
    printf("  mean negative log-likelihood of training data without training %f \n",modelSetInfo->crtVal);
   
    if (modelSetInfo->crtVal < modelSetInfo->bestValue){
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	printf("computing initial error on training  data\n");
	zero_one_errorTraining[currentEpochIdx] = updatateAcc(labels, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	printf("training 0-1error %lf \n ",zero_one_errorTraining[0] );
    
    //with the initialisation of weights,checking how well DNN performs on validation data
	printf("computing initial error on validation data\n");
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,NULL);
	fwdPassOfANN(anndef);
	zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	min_validation_error = zero_one_error[currentEpochIdx];
	printf("validation error %lf \n ",zero_one_error[currentEpochIdx] );
	
	initialiseParameterCaches(anndef);
	
	while(currentEpochIdx <  maxEpochNum){
		printf("epoc number %d \n", currentEpochIdx);
		currentEpochIdx+=1;
		//load training data into the ANN and perform forward pass and accumulate gradient
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labelMat);
		fwdPassOfANN(anndef);
        printf(" mean negative log-likelihood of training data %lf \n",computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels));
		backPropBatch(anndef,FALSE);
		printf("back prop successful\n");
    	accumulateGradientsofANN(anndef);
	    printf("successfully accumulated Gradients \n");
       
        //perform CG on smaller minibatch
		setUpMinibatchforHF(anndef);
		setBatchSizetoHFminiBatch();
		reinitLayerFeaMatrices(anndef);
		reinitLayerErrFeaMatrices(anndef);
		loadMiniBatchintoANN();
		fwdPassOfANN(anndef);
		printf("forward pass on minibatch successful \n");
		//minBatchLabels =labels;
		runConjugateGradient();
		printf("successfully completed a run of CG \n");

		//checking the performance of updated parameters on the entire training set
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		reinitLayerErrFeaMatrices(anndef);
		loadDataintoANN(inputData,labelMat);
		fwdPassOfANN(anndef);
		log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);	
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
		printf(" mean negative log-likelihood of entire training data after HF %lf \n ",modelSetInfo->crtVal );
		zero_one_errorTraining[currentEpochIdx] = updatateAcc(labels, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
	

		printf("checking the performance of the neural net on the validation data\n");
		setBatchSize(validationDataSetSize);
		printf("BATCHSAMPLES set to %d\n", BATCHSAMPLES);
		reinitLayerFeaMatrices(anndef);
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,NULL);
		fwdPassOfANN(anndef);
		zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[anndef->layerNum-1],BATCHSAMPLES);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			modelSetInfo->bestValue = modelSetInfo->crtVal;
		}	
		if (zero_one_error[currentEpochIdx]<min_validation_error){
				min_validation_error = zero_one_error[currentEpochIdx];
				cacheParameters(anndef);
		}
		printf("\n\n");

	}
	clock_t end = clock();
	printf("TRAINING ERROR >>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printArray(log_likelihood,maxEpochNum+1);
	printf("THE  VALIDATION RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printArray(zero_one_error,maxEpochNum+1);
	printf("The zero -one loss on training set >>>>>>>>>>>>>>>>>\n");
	printArray(zero_one_errorTraining,maxEpochNum+1);
	
	printf("The minimum error on the validation data set is %lf percent  and min log_likelihood is %lf \n",min_validation_error*100, modelSetInfo->bestValue);
	free(zero_one_error);
	free(log_likelihood);
	free(zero_one_errorTraining);
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
 	printf("total time taken %lf \n ", seconds);
	

}
//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------

void freeMemoryfromANN(){
	int i;
	printf("Start freeing memory\n");
	if (anndef != NULL){
		for (i = 0;i<numLayers;i++){
			if (anndef->layerList[i] !=NULL){
				if (anndef->layerList[i]->feaElem != NULL){
					if (anndef->layerList[i]->feaElem->yfeatMat !=NULL){
						DisposeMatrix(anndef->layerList[i]->feaElem->yfeatMat);
					}
					free(anndef->layerList[i]->feaElem);
				}
				if (anndef->layerList[i]->errElem !=NULL){
					if (anndef->layerList[i]->errElem->dxFeatMat != NULL){
						DisposeMatrix(anndef->layerList[i]->errElem->dxFeatMat);
					}
					free(anndef->layerList[i]->errElem);
				}
				if (anndef->layerList[i]->traininfo!=NULL){
					DisposeMatrix (anndef->layerList[i]->traininfo->dwFeatMat);
					DisposeFloatVec(anndef->layerList[i]->traininfo->dbFeaMat);
					if(anndef->layerList[i]->traininfo->updatedBiasMat !=NULL){
						DisposeFloatVec(anndef->layerList[i]->traininfo->updatedBiasMat);
					}
					if (anndef->layerList[i]->traininfo->updatedWeightMat!= NULL){
						DisposeMatrix(anndef->layerList[i]->traininfo->updatedWeightMat);
					}
					free (anndef->layerList[i]->traininfo);
				}
				if (anndef->layerList[i]->weights !=NULL){
					DisposeMatrix (anndef->layerList[i]->weights);
				}
				if (anndef->layerList[i]->bias !=NULL){
					DisposeFloatVec (anndef->layerList[i]->bias);
				}
				if(anndef->layerList[i]->gnInfo != NULL){
					if (anndef->layerList[i]->gnInfo->vweights !=NULL){
						DisposeMatrix(anndef->layerList[i]->gnInfo->vweights);
					}
					if (anndef->layerList[i]->gnInfo->vbiases !=NULL){
						DisposeFloatVec (anndef->layerList[i]->gnInfo->vbiases);
					}
					if (anndef->layerList[i]->gnInfo->Ractivations !=NULL){
						DisposeMatrix(anndef->layerList[i]->gnInfo->Ractivations);
					}
					free (anndef->layerList[i]->gnInfo);
				}
				if(anndef->layerList[i]->cgInfo !=NULL){
					if(anndef->layerList[i]->cgInfo->delweightsUpdate != NULL){
						DisposeMatrix(anndef->layerList[i]->cgInfo->delweightsUpdate);
					}
					if (anndef->layerList[i]->cgInfo->delbiasUpdate != NULL){
						DisposeFloatVec (anndef->layerList[i]->cgInfo->delbiasUpdate);
					}
					if (anndef->layerList[i]->cgInfo->residueUpdateWeights != NULL){
						DisposeMatrix(anndef->layerList[i]->cgInfo->residueUpdateWeights );
					}
					if (anndef->layerList[i]->cgInfo->residueUpdateBias != NULL){
						DisposeFloatVec(anndef->layerList[i]->cgInfo->residueUpdateBias);
					}
					if (anndef->layerList[i]->cgInfo->searchDirectionUpdateBias != NULL){
						DisposeFloatVec(anndef->layerList[i]->cgInfo->searchDirectionUpdateBias);
					}
					if (anndef->layerList[i]->cgInfo->searchDirectionUpdateWeights != NULL){
						DisposeMatrix(anndef->layerList[i]->cgInfo->searchDirectionUpdateWeights);
					}
					free(anndef->layerList[i]->cgInfo);
				}
				free (anndef->layerList[i]);
			}
		}
		free(anndef->layerList);	
		free(anndef);
		free(modelSetInfo);
	}
	if(inputData !=NULL) DisposeMatrix(inputData);
	if (labels != NULL) DisposeIntVec(labels);
	if (labelMat !=NULL) DisposeMatrix(labelMat);
	if (validationData != NULL ) DisposeMatrix(validationData);
	if (validationLabelIdx != NULL) DisposeIntVec(validationLabelIdx);
	if (miniBatchforHF != NULL) DisposeMatrix (miniBatchforHF);
	if (minBatchLabels !=NULL) DisposeIntVec (minBatchLabels);
	
	#ifdef CUDA
	StopCUDA();
	#endif
	printf("Finished freeing memory\n");
}

void printLayerDWMatrices(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		printf("LAYER ID %d >>>>>>\n",i);
		layer = anndef->layerList[i];
		printf("printing  DEL W >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->delweightsUpdate,layer->dim,layer->srcDim);
		printf("printing  DEL B >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printVector(layer->cgInfo->delbiasUpdate,layer->dim);
		

	}	
}
void printLayerMatrices(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		printf("LAYER ID %d >>>>>>\n",i);
		layer = anndef->layerList[i];
		printf("printing   W >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->weights,layer->dim,layer->srcDim);
		printf("printing  B >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->weights,1,layer->dim);
		

	}	
}
void printIArray(int *vector , int dim){
	int i ;
	printf("[ ");
	for (i = 0; i < dim; i++){
		printf( " %d ",vector[i]);
	}	
	printf("]\n ");
}


void printArray(float *vector , int dim){
	int i ;
	printf("[ ");
	for (i = 0; i < dim; i++){
		printf( " %lf ",vector[i]);
	}	
	printf("]\n ");
}


void printVector(NFVector *vector , int dim){
	int i ;
	#ifdef CUDA
	SyncDev2Host(vector->deviceElems,vector->elems,sizeof(float)*dim);
	#endif
	printf("[ ");
	for (i = 0; i < dim; i++){
		printf( " %lf ",vector->elems[i]);
	}	
	printf("]\n ");
}

void printMatrix(NMatrix *matrix,int row,int col){
int k,j;
int r = row;
int c = col;
#ifdef CUDA
printf("copying from device to host \n");
SyncDev2Host(matrix->deviceElems,matrix->elems,sizeof(float)*row*col);
#endif
for (j = 0 ; j< r;j++){
	for (k = 0 ; k <c;k++){
		printf(" %lf ", matrix->elems[j*col +k]);
	
		}
	printf("\n");	
}
printf("printed matrix \n");

}

void printYfeat(ADLink anndef,int id){
	int i,k,j; 
	LELink layer;
	
		layer = anndef->layerList[id];
		int dim;
	if (layer->dim >10 ) {
		dim= 20;
	} else{
		dim =10;
	}
		printf("layer id %d dim is  %d\n ",id,layer->dim);
		for (i = 0; i<20;i++){
			for (j =0 ; j<dim;j++){
				printf("%lf ",layer->feaElem->yfeatMat->elems[i*layer->dim+j]);
		}
		printf("\n");
	}
	
	printf("printing WEIGHTS \n");
		for (j = 0 ; j< dim;j++){
			for (k =0 ; k <20;k++){
				printf(" %lf ", layer->weights->elems[j*layer->dim +k]);
			}
			printf("\n");
		}
		printf("printing BIAS \n");
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->bias->elems[j]);
		}
		printf("\n");
	
	
}

void UnitTest_computeGradientDotProd(){
	int i,k,j; 
	LELink layer;
	float sum =0;
	for ( i  =0 ; i < anndef->layerNum; i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		sum += cblas_sdot(layer->dim*layer->srcDim,layer->traininfo->updatedWeightMat->elems, 1, layer->traininfo->updatedWeightMat->elems,1);
		#endif
	}
	printf("dotproduct of cache gradient vector is %lf \n",sum);
	

}

void unitTestofIterRprop(int iter){
	float cost = DBL_MAX;
	float oldcost = DBL_MAX;
	int i;
	for (i = 0; i <iter;i++){
		printf("\n\n ITER 5%d \n\n",i);
		fwdPassOfANN(anndef);
		//float loglik =computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,lab);	
		//printf("log like %lf\n>>>>",loglik);
		printf("Computing derivative  >>>>>>>>>>>>>>>\n");
		cost = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);	
		printf(" cost %f oldcost %f \n",cost,oldcost);

		backPropBatch(anndef,FALSE);
		computeNormOfGradient(anndef);
		//updateNeuralNetParams(anndef,0.1,0,0);
		
		printf("de/dw of output layer berore Rprop\n");
		printMatrix(anndef->layerList[1]->traininfo->dwFeatMat,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);
		printf("de/dw of hidden layer berore Rprop\n");
		printMatrix(anndef->layerList[0]->traininfo->dwFeatMat,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);



		updateWithRprop(anndef,cost, oldcost);	
		//printf("PRINTING Updated WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");
		/*printf("Layer 0 weights \n");
		printMatrix(anndef->layerList[0]->weights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);
		printf("Layer 0 bias \n");
		printVector(anndef->layerList[0]->bias,anndef->layerList[0]->dim);

		printf("Layer 1 weights \n");
		printMatrix(anndef->layerList[1]->weights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);
		printf("Layer 1 bias \n");
		printVector(anndef->layerList[1]->bias,anndef->layerList[1]->dim);
		printf("done printing update  WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");
		*/
		if(i==1){doHF=TRUE;}
		accumulateGradientsofANN(anndef);
		computeNormOfGradient(anndef);
		computeNormOfAccuGradient(anndef);
		oldcost = cost;
	}	

}

/*This function is used to check the correctness of implementing the forward pass of DNN and the back-propagtion algorithm*/
void unitTests(){
	//checking the implementation of doing products with the gauss newton matrix
	float cost,oldcost;
	size_t memoryM = sizeof(float)*(100);
	size_t memoryV = sizeof(float)*(10);

	//only required for HF //////////////////////////////////////////////////////////////////////
	NMatrix * vweightsH = malloc(sizeof(NMatrix));
	float  vweightsHM[] ={0.775812455441121,0.598968296474700,0.557221023059154,0.299707632069582,0.547657254432423,0.991829829468103,0.483412970471810,0.684773696180478,0.480016831195868,0.746520774837809,0.211422430258003,0.248486726468514,0.0978390795725171,0.708670434511610,0.855745919278928,0.789034249923306,0.742842549759275,0.104522115223072,0.520874909265143,0.846632465315532,0.843613150651208,0.377747297699522,0.272095361461968,0.125303924302938,0.691352539306520,0.555131164040551,0.00847423194155961,0.416031807168918,0.439118205411470,0.784360645016731,0.829997491112413,0.589606438454759,0.142074038271687,0.593313383711502,0.726192799270104,0.428380358133696,0.210792055986133,0.265384268404268,0.993183755212665,0.480756369506705,0.827750131864470,0.603238265822881,0.817841681068066,0.955547170535556,0.650378193390669,0.627647346270778,0.646563331304311,0.0621331286917197,0.938196645878781,0.898716437594415,0.694859162934643,0.310406171229422,0.343545605630366,0.0762294431350499,0.519836940596257,0.608777321469077,0.727159960434458,7.39335770538752e-05,0.169766268796397,0.463291023631989,0.361852151271785,0.952065614552366,0.932193359196872,0.958068660798924,0.206538420408243,0.159700734052653,0.571760567030289,0.592919191548301,0.698610749870688,0.305813464081036,0.393851341078336,0.336885826085107,0.128993041260436,0.0869363561072062,0.548943348806040,0.317733245140830,0.992747576055697,0.723565253441235,0.572101390105790,0.717227171632155,0.976559227688336,0.426990558230810,0.913013771059024,0.897311254281181,0.835228567475913,0.0479581714127454,0.359280569413193,0.295753896108793,0.629125618231216,0.136183801688937,0.374045801279154,0.864781698209823,0.250273591941488,0.0295055785950556,0.999448971390378,0.605142675082282,0.244195121925386,0.438195271553610,0.735093567145817,0.557989092238218};	
	vweightsH->elems = vweightsHM;
	#ifdef CUDA
	cudaMalloc(&vweightsH->deviceElems,memoryM);
	SyncHost2Dev(vweightsH->elems,vweightsH->deviceElems,memoryM);
	#endif
	NFVector * vbaisH = malloc(sizeof(NFVector));
	float vbaisHM[] ={0,0,0,0,0,0,0,0,0,0};
	vbaisH->elems = vbaisHM ;
	#ifdef CUDA
	cudaMalloc(&vbaisH->deviceElems,memoryV);
	SyncHost2Dev(vbaisH->elems,vbaisH->deviceElems,memoryV);
	#endif

	NMatrix * vweights0 = malloc(sizeof(NMatrix));
	float  vweights0M[] ={0.775812455441121,0.598968296474700,0.557221023059154,0.299707632069582,0.547657254432423,0.991829829468103,0.483412970471810,0.684773696180478,0.480016831195868,0.746520774837809,0.211422430258003,0.248486726468514,0.0978390795725171,0.708670434511610,0.855745919278928,0.789034249923306,0.742842549759275,0.104522115223072,0.520874909265143,0.846632465315532,0.843613150651208,0.377747297699522,0.272095361461968,0.125303924302938,0.691352539306520,0.555131164040551,0.00847423194155961,0.416031807168918,0.439118205411470,0.784360645016731,0.829997491112413,0.589606438454759,0.142074038271687,0.593313383711502,0.726192799270104,0.428380358133696,0.210792055986133,0.265384268404268,0.993183755212665,0.480756369506705,0.827750131864470,0.603238265822881,0.817841681068066,0.955547170535556,0.650378193390669,0.627647346270778,0.646563331304311,0.0621331286917197,0.938196645878781,0.898716437594415,0.694859162934643,0.310406171229422,0.343545605630366,0.0762294431350499,0.519836940596257,0.608777321469077,0.727159960434458,7.39335770538752e-05,0.169766268796397,0.463291023631989,0.361852151271785,0.952065614552366,0.932193359196872,0.958068660798924,0.206538420408243,0.159700734052653,0.571760567030289,0.592919191548301,0.698610749870688,0.305813464081036,0.393851341078336,0.336885826085107,0.128993041260436,0.0869363561072062,0.548943348806040,0.317733245140830,0.992747576055697,0.723565253441235,0.572101390105790,0.717227171632155,0.976559227688336,0.426990558230810,0.913013771059024,0.897311254281181,0.835228567475913,0.0479581714127454,0.359280569413193,0.295753896108793,0.629125618231216,0.136183801688937,0.374045801279154,0.864781698209823,0.250273591941488,0.0295055785950556,0.999448971390378,0.605142675082282,0.244195121925386,0.438195271553610,0.735093567145817,0.557989092238218};	
	vweights0->elems = vweights0M;
	#ifdef CUDA
	cudaMalloc(&vweights0->deviceElems,memoryM);
	SyncHost2Dev(vweights0->elems,vweights0->deviceElems,memoryM);
	#endif
	
	NFVector * vbias0 = malloc(sizeof(NFVector));
	float vbaisOM[] ={0,0,0,0,0,0,0,0,0,0};
	vbias0->elems = vbaisOM;
	#ifdef CUDA
	cudaMalloc(&vbias0->deviceElems,memoryV);
	SyncHost2Dev(vbias0->elems,vbias0->deviceElems,memoryV);
	#endif

	////////////////////////////////////////////////////////////////////////////////




	NMatrix * data = malloc(sizeof(NMatrix));
	float dataM[] ={0.254790156597005,0.224040030824219,0.667832727013717,0.844392156527205,0.344462411301042,0.780519652731358,0.675332065747000,0.00671531431847749,0.602170487581795,0.386771194520985,0.915991244131425,0.00115105712910724,0.462449159242329,0.424349039815375,0.460916366028964,0.770159728608609,0.322471807186779,0.784739294760742,0.471357153710612,0.0357627332691179};
	data->elems = dataM;
	#ifdef CUDA
	cudaMalloc(&data->deviceElems, sizeof(float)*10*2);
	SyncHost2Dev(data->elems,data->deviceElems,memoryV*2);
	#endif

	labels = malloc(sizeof(NIntVector));
	int * labelsV = malloc(sizeof(int)*2);
	labelsV[0] =4;
	labelsV[1] =1;
	labels->elems = labelsV;
	#ifdef CUDA
	cudaMalloc(&labels->deviceElems,sizeof(int)*2);
	SyncHost2Dev(labels->elems,labels->deviceElems,sizeof(int)*2);
	#endif


	NMatrix * labmat = malloc(sizeof(NMatrix));
	float labmatM[] ={0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
	labmat->elems = labmatM;
	#ifdef CUDA
	cudaMalloc(&labmat->deviceElems,sizeof(float)*20);
	SyncHost2Dev(labmat->elems,labmat->deviceElems,sizeof(float)*20);
	#endif

	numLayers = 2;
	hidUnitsPerLayer = malloc(sizeof(int)*numLayers);
	actfunLists = malloc( sizeof(ActFunKind)*numLayers);
	hidUnitsPerLayer[0]=10;
	actfunLists[0]= TANH;
	actfunLists[1]= SOFTMAX;		
	target=CLASSIFICATION;
	BATCHSAMPLES= 2;
	errfunc = XENT;
	inputDim = 10;
	targetDim =10;
	doHF = FALSE;
	doRprop = TRUE;
	useGNMatrix =TRUE;
	maxNumOfCGruns =10;
	weightdecay =1;

	
	

	initialiseDNN();
	anndef->layerList[0]->feaElem->xfeatMat = data;
	anndef->labelMat = labmat;

	NMatrix * Weight = malloc(sizeof(NMatrix));
	float weight [] ={-0.33792351, 0.13376346, -0.06821584,0.31259467,0.30669813, -0.24911232,-0.24487114,0.3306844, 0.50186652,0.41181357,-0.15575338,0.00109011,0.20097358,0.2330034,-0.14213318,0.06703706, 0.00337744, -0.53263998,0.29886659,0.41916242,-0.14800999,0.12641018, -0.46514654, -0.1436961, 0.47448121,0.16582645,-0.11260893,0.31628802, -0.20064598,0.07459834, 0.4043588,-0.06991851,0.33098616, -0.39023389,0.22375668,0.22410759,-0.30804781,0.46541917 ,-0.06338163,0.44838317,-0.48220484, -0.34584617, -0.49584745,0.19157248,0.10365625,0.03648946,-0.50026342,0.06729657, -0.18658887,0.00325,-0.42514847,0.11742482,0.07223874, -0.5403129, 0.12865095,0.451458, 0.31825324,0.53904824,0.50259215,0.31983069,-0.23524579,0.13683939, -0.02399704, -0.33337114, -0.12891477, -0.48870689,-0.05296651,0.52800974, -0.41195013, -0.41694734, 0.26128892,0.09563634, -0.031075, -0.43037101, -0.2966262, 0.4381399,-0.09119193,0.03927353, -0.54092147, -0.21838607,-0.06913007,0.12285307,0.45811304,0.13773762,0.22565903,-0.38358795,0.26954896,0.36259999,0.14648924, -0.06757814,-0.38058746,0.07493898,0.03091815,0.49451543, -0.02151544,0.00280386, 0.04039804,0.34966835, -0.48515551,0.18559222};
	Weight->elems = weight;
	#ifdef CUDA
	cudaMalloc(&Weight->deviceElems,memoryM);
	SyncHost2Dev(Weight->elems,Weight->deviceElems,memoryM);
	#endif
	

	printf("Printing input data >>>>>\n");
	printMatrix(anndef->layerList[0]->feaElem->xfeatMat,BATCHSAMPLES,anndef->layerList[0]->dim);
	
	printf("printing initialised weights >>>>>\n");
	printf("layer 0\n");
	printMatrix(anndef->layerList[0]->weights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);
    printf("Layer 0 bias \n");
	printVector(anndef->layerList[0]->bias,anndef->layerList[0]->dim);

    printf("layer 1\n");
	printMatrix(anndef->layerList[1]->weights,anndef->layerList[1]->dim,anndef->layerList[0]->srcDim);
    printf("Layer 1 bias \n");
	printVector(anndef->layerList[1]->bias,anndef->layerList[1]->dim);
	

    CopyMatrix (Weight,0,anndef->layerList[0]->weights,0,anndef->layerList[0]->dim*anndef->layerList[0]->srcDim);
	//CopyMatrix (Weight,0,anndef->layerList[1]->weights,0,anndef->layerList[1]->dim*anndef->layerList[1]->srcDim);
	
	//CopyMatrixOrVec (Weight,anndef->layerList[1]->weights,anndef->layerList[1]->dim*anndef->layerList[1]->srcDim);
	
	printf("PRINTING WEIGHTS OF LAYERS after copying>>>>>>>>>>>>>>>\n");
	printf("Layer 0 weights \n");
	printMatrix(anndef->layerList[0]->weights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);
	printf("Layer 0 bias \n");
	printVector(anndef->layerList[0]->bias,anndef->layerList[0]->dim);

	printf("Layer 1 weights \n");
	printMatrix(anndef->layerList[1]->weights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);
	printf("Layer 1 bias \n");
	printVector(anndef->layerList[1]->bias,anndef->layerList[1]->dim);
	printf("done  WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");
	
	printf("computing norm of weights\n");
	normOfWeights(anndef);	

	//fwdPassOfANN(anndef);
	printf("RUNNING rProp UNIT TESTS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");
	unitTestofIterRprop(4);
	printf("FINISH RUNNING rProp UNIT TESTS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");
	

	/*
	CopyMatrix (vweightsH,0,anndef->layerList[0]->gnInfo->vweights,0,inputDim*anndef->layerList[0]->dim);
	printf("vweights of hidden layer\n");
	printMatrix(anndef->layerList[0]->gnInfo->vweights,10,10);
	CopyVec (vbaisH,0,anndef->layerList[0]->gnInfo->vbiases,0,anndef->layerList[0]->dim);
	printf("vbais of hidden layer\n");
	printVector(anndef->layerList[0]->gnInfo->vbiases,10);
	CopyMatrix (vweights0,0,anndef->layerList[1]->gnInfo->vweights,0,anndef->layerList[1]->dim*anndef->layerList[1]->srcDim);
	printf("vweights of output layer\n");
	printMatrix(anndef->layerList[1]->gnInfo->vweights,10,10);
	CopyVec (vbias0,0,anndef->layerList[1]->gnInfo->vbiases,0,anndef->layerList[1]->dim);
	printf("vbias of hidden layer\n");
	printVector(anndef->layerList[1]->gnInfo->vbiases,10);
	*/
	fwdPassOfANN(anndef);
	//float loglik =computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,lab);	
	//printf("log like %lf\n>>>>",loglik);
	oldcost = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);	
	printf("old cost %f \n",oldcost);

	printf("Computing derivative  >>>>>>>>>>>>>>>\n");
	
	backPropBatch(anndef,FALSE);
	computeNormOfGradient(anndef);
	//updateNeuralNetParams(anndef,0.1,0,0);
	
	printf("de/dw of output layer berore Rprop\n");
	printMatrix(anndef->layerList[1]->traininfo->dwFeatMat,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);



	printf("PRINTING Updated WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");
	printf("Layer 0 weights \n");
	printMatrix(anndef->layerList[0]->weights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);
	printf("Layer 0 bias \n");
	printVector(anndef->layerList[0]->bias,anndef->layerList[0]->dim);

	printf("Layer 1 weights \n");
	printMatrix(anndef->layerList[1]->weights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);
	printf("Layer 1 bias \n");
	printVector(anndef->layerList[1]->bias,anndef->layerList[1]->dim);
	printf("done printing update  WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");

	
	
	accumulateGradientsofANN(anndef);
	computeNormOfGradient(anndef);
	computeNormOfAccuGradient(anndef);
	//printf("de/dw of hidden layer\n");
	//printMatrix(anndef->layerList[0]->traininfo->dwFeatMat,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	//printf("de/db of hidden layer\n");
	//printVector(anndef->layerList[0]->traininfo->dbFeaMat,10);

	
	//printf("de/dw of hidden layer\n");
//	printMatrix(anndef->layerList[0]->traininfo->dwFeatMat,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	//printf("de/db of hidden layer\n");
	//printVector(anndef->layerList[0]->traininfo->dbFeaMat,10);

	printf("de/dw of output layer\n");
	printMatrix(anndef->layerList[1]->traininfo->dwFeatMat,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	//printf("de/db of output layer\n");
	//printVector(anndef->layerList[1]->traininfo->dbFeaMat,10);

	printf("Running ConjuageGradient\n");
	


	//minBatchLabels =labels;
	//runConjugateGradient();
	#ifdef CUDA
	DevDispose(Weight->deviceElems);
	#endif
	free(Weight);
	#ifdef CUDA
	DevDispose(vweightsH->deviceElems);
	#endif
	free(vweightsH);
	#ifdef CUDA
	DevDispose(vweights0->deviceElems);
	#endif
	free(vweights0);
	#ifdef CUDA
	DevDispose(vbaisH->deviceElems);
	#endif
	free(vbaisH);
	#ifdef CUDA
	DevDispose(vbias0->deviceElems);
	#endif
	free(vbias0);
	#ifdef CUDA
	DevDispose(data->deviceElems);
	#endif
	free(data);
	#ifdef CUDA
	DevDispose(labmat->deviceElems);
	#endif
	free(labmat);

	printf("Successfully freed local matrices\n");
	
	freeMemoryfromANN();

}


//==============================================================================

int main(int argc, char *argv[]){
	#ifdef CUDA 
	printf("Using Cuda\n");
	StartCUDA();
	#endif
	
	//unitTests();
	//exit(0);
	
	//
	
	/**testing gauss newton product**/
	if (argc != 11 && argc != 13 ){
		printf("The program expects a minimum of  5 args and a maximum of 6 args : Eg : -C config \n -S traindatafile \n -L traininglabels \n -v validationdata \n -vl validationdataLabels \n optional argument : -T testData \n ");
		exit(0);
	}
	parseCMDargs(argc, argv);
	//exit(0);
	//initialise();
	/*NMatrix * Wtest = malloc (sizeof(NMatrix));
	float  * W = malloc(sizeof(float )*784*500);
	for (int i =0 ; i< 784*500;i++){
		W[i] = 0.03; 
	
	} 
	Wtest->elems = W;*/
	//free(W);
	/*NMatrix * Wtest;
	Wtest = CreateMatrix(500,784); 
	//float  * W = malloc(sizeof(float )*100);
	int i;
	for ( i =0 ; i< 500*784;i++){
		Wtest->elems[i] = 0.03; 
	
	} */
	//Wtest->elems = W;
	//printMatrix(Wtest,10,10);
	//free(W);
	
	//CopyMatrix (Wtest,0,anndef->layerList[0]->weights,0,anndef->layerList[0]->dim*anndef->layerList[0]->srcDim);
    //free(Wtest);
    /*
    loadDataintoANN(inputData,labelMat);
	normOfWeights(anndef);
	fwdPassOfANN(anndef);
    printf("initial cost %lf \n",computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels));
	backPropBatch(anndef,FALSE);
	printf("back prop successful\n");
        printf("de/dw of hidden layer\n");
		printMatrix(anndef->layerList[0]->traininfo->dwFeatMat,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

		printf("de/db of hidden layer\n");
		printVector(anndef->layerList[0]->traininfo->dbFeaMat,7840);

		printf("de/dw of output layer\n");
		printMatrix(anndef->layerList[1]->traininfo->dwFeatMat,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

		printf("de/db of output layer\n");
		printVector(anndef->layerList[1]->traininfo->dbFeaMat,10);

        computeNormOfGradient(anndef);
	*/					
	
	
	
	
	initialise();
	//setValueInMatrix(anndef->layerList[0]->weights,0.001);

	if (doHF) TrainDNNHF();
	else if (doRprop) TrainDNNRprop();
	else TrainDNNGD();
	
	freeMemoryfromANN();

} 

#ifdef __cplusplus
}
#endif 
