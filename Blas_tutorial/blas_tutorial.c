#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "/Users/adnan/NeuralNet/CBLAS/include/cblas.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void * Ptr;

typedef enum {SIGMOID,IDENTITY,TANH} ActFunKind;

double computeTanh(double x);
double computeSigmoid(double x);
void computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc);
void testfunct(double *v);

void initialiseWeights(double *weightMat,int length,int srcDim);
void initialiseBias(double *biasVec,int dim, int srcDim);
double random_normal() ;
double drand()  ;
void setHook(Ptr m, Ptr ptr,int incr);
Ptr getHook(Ptr m,int incr);

static double randSeed;
typedef struct _hookStruct{
	double *matrix;
	int id;
	double t;
}Hook;

void settingHookTest(Hook * hook){
int i;
	double * hookmat2 = malloc(sizeof(double)*1000);
   
   double * hookmat3 = malloc(sizeof(double)*1000);
   for (i =0 ; i< 1000;i++){
		hookmat3[i] =1;
	}

   setHook(hook,hookmat2,2);
   setHook(hook,hookmat3,1);
}




static double *arr;

int main(){

	char *ptr;
	int *stateIdx;
   int i,off;
	int dim;
	int srcDim;
	int size;
	double weights[] ={ 2, 0,0,2,1,0,0,2};
	double *weightmat;
	double *biasVec;
	Hook *hook;
	Hook *hook2;

	/**testing string formating */
	char str[40] = "hmm1/HER$.acc";
	sprintf(str,"%d",10);
	printf("str is %s \n",str);	




	//---------------------------------------------------------
	double A3[] = { 0.11, 0.12, 0.13,
                0.21, 0.22, 0.23 };
   double B2[] = { 1, 2, 0.13,
                0.21, 0.22, 0.23 };             
   //double *hookmat;             
	/** testing pointer hooks-**/
   hook = malloc(sizeof(Hook)+ sizeof(double)*2000); 
	hook->matrix = A3;
   hook->id =3;
   /*cblas_dcopy(8,weights,1,hookmat,1);
   for (i =0 ; i< 8;i++){
   	printf("testing hooks values %d %f \n",i,*(hookmat+i));
   }
   printf(" Ptr  size hookmat  is %lu \n",sizeof(hookmat));
   free(hookmat);
	
	hook->id = 2;
	setHook(A3, A3);
   hookmat =(double *) getHook(A3);
   A3[1] =1;
   for (i =0 ; i< 6;i++){
   	printf("testing hooks values %d %f \n",i,*(hookmat+i));
   }
   printf(" Ptr  size hook  is %lu \n",sizeof(hook));
	free(hook);
   */
   settingHookTest(hook);

   



  // free(hookmat2);
   double* retrievedhookmat2 = (double *)getHook(hook,2);
   double* retrievedhookmat1 = (double *)getHook(hook,1);
   
   memcpy(retrievedhookmat2,retrievedhookmat1,sizeof(double)*1000);
	for (i =0 ; i< 6;i++){
		printf("retrieved hookmat value 2 %lf \n",retrievedhookmat2[i]);
	}
	for (i =0 ; i< 6;i++){
		printf("hook matrix value %lf \n",hook->matrix[i]);
	}
	for (i =0 ; i< 6;i++){
		printf("retrievedhookmat1 value %lf \n",retrievedhookmat1[i]);
	}
	hook->matrix = B2;
	
	double* retrievedhookmat2_ = (double *)getHook(hook,2);
	memcpy(retrievedhookmat2_,B2,sizeof(B2));

	for (i =0 ; i< 6;i++){
		printf("hook matrix value %lf \n",hook->matrix[i]);
	}
	

	for (i =0 ; i< 10;i++){
		printf("retrieved hookmat value 2_2 %lf \n",retrievedhookmat2_[i]);
	}

	double* retrievedhookmat3_ = (double *)getHook(hook,2);
	for (i =0 ; i< 10;i++){
		printf("retrieved hookmat value 3_2 %lf \n",retrievedhookmat3_[i]);
	}

	//free(retrievedhookmat);
	free(hook);
	

	//for (i =0 ; i< 6;i++){
	//	printf("hookmat value %lf \n",hookmat2[i]);
	//}

	//free(hookmat3);   


   //free(hook2->matrix);

//----------------------------------------------------------







	//--------------------------------------------------------





	dim = 4;
	srcDim = 2;

	/**testing the dot product of two vectors */
	double p[] =  {0.47021, 0.761908,0.0000};
	double s[] ={0.458996,-0.329061,0.234};


	printf("the dot product is A %f \n",cblas_ddot(3,p,1,s,1) );

	//====================================================
	 int lda = 3;

  float A1[] = { 0.11, 0.12, 0.13,
                0.21, 0.22, 0.23 };



  float A2[] ={ 0.11, 0.12, 
  				0.13,0.21, 
  				0.22, 0.23 };        
  float ones[] ={1,1,1};
  float sumCOls[] ={ 0,0};
  float test[] = { 1,0,1};

  int ldb = 2;
  
  float B[] = { 1, 2,
                3, 4,
                5, 6 };
  
  float B_2[] = { 1, 2,3, 
  				4,5, 6 }; 
  float D3[] = { 1,2,
  				3,4,
  				5,6,
  				7,8 };				             
  int ldc = 2;

  float C[] = {0.00, 0.00,0.00,
                0.00, 0.00 ,000,
             	0.00 ,0.00, 0.00,
             	0.00 ,0.00,0.00 };

  /* Compute C = A B */

/** when we say col major and cblasTrans and cblas noTrans then we multiply B'*A' where A is col-major but B is row-major
	change only lda of A  and the result is stored in row major
*/
/*when we say col major and cblasnoTrans and cblasnoTrans both matrices are presented in col major and we A*B 
and lda of C will row of A
and the result is stored in col major*/
 /* when we say col major cblasno Trans and cblasTrans then we mutliply B'*A' where B' is col major and A is row major
	change only lda b and result is stored in row major
 */  
  printf("printing C before \n");
  printf ("[ %g, %g\n", C[0], C[1]);
  printf ("  %g, %g ]\n", C[2], C[3]);
  printf ("  %g, %g \n", C[4], C[5]);
  printf (" %g, %g\n", C[6], C[7]);
  printf ("  %g ]\n", C[8]);
  
  cblas_sgemm (CblasColMajor, 
               CblasNoTrans, CblasTrans, 3,4,2,
               1.0, A2,3, D3, 4, 0.0, C, 3);

  printf("printing C \n");
  printf ("[ %g, %g\n", C[0], C[1]);
  printf ("  %g, %g ]\n", C[2], C[3]);
  printf ("  %g, %g \n", C[4], C[5]);
  printf (" %g, %g\n", C[6], C[7]);
  printf ("[ %g, %g\n", C[8], C[9]);
  printf ("  %g, %g ]\n", C[10], C[11]);
  printf(" done printing C \n");
  
  

//testing copying segments**/
  float *copy = malloc(sizeof(float)*3);
  memcpy(copy,A1,sizeof(float)*3);

  float *outpro = malloc(sizeof(float)*3);

  cblas_sgemm (CblasColMajor, 
               CblasTrans, CblasTrans, 3, 3, 1,
               1.0, copy,1, copy, 3, 0, outpro, 3);

  printf ("[ %g, %g\n", outpro[0], outpro[1]);
  printf ("  %g, %g ]\n", outpro[2], outpro[3]);
  printf ("  %g, %g \n", outpro[4], outpro[5]);
  printf (" %g, %g\n", outpro[6], outpro[7]);
  printf ("  %g ]\n", outpro[8]);
  
  free (outpro);


  printf("element %d of copy is %f\n",0,copy[0]);
  printf("element %d of copy is %f\n",0,copy[1]);
  printf("element %d of copy is %f\n",0,copy[2]);

  free(copy);



/**

  cblas_saxpy(3,-1,ones,1,test,1);
  printf (" T %g, %g  %g ]\n", test[0], test[1],test[2]);

  cblas_sscal(3,2.5,ones,1);
  printf (" SC %g, %g  %g ]\n", ones[0], ones[1],ones[2]);

  
  cblas_sgemv(CblasRowMajor,CblasNoTrans, 2,3,1,A1,3,ones,1,0,sumCOls,1);
  printf ("[ %g, %g\n", sumCOls[0], sumCOls[1]);
  //printf ("[ %g, %g\n", sumCOls[2], sumCOls[3]);





**/









	//=======================================================
   /*  W is 4 by 2 and X is 3 by 2 so X *W^T be 3 by 4
	float W [] ={ 1, 1, 0,
						0,0,0};
	float X[] ={ 0, 1,
						1 ,0,
						1,1};
	
	float r[]  = {0,0,0,0};  
	float b[] ={11,11};

	cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,2,2,3,1,W,2,X,2,0,r,2);
	
	for (i=0;i<4;i++){
		printf("the CHECK output is %f\n ",r[i]);
	}
	for (i =0, off = 0; i < 2;++i, off += 2){
		cblas_scopy(2, b, 1, r + off, 1);
	}
	for (i=0;i<4;i++){
		printf("the new CHECK output is %f\n ",r[i]);
	}



	//----------------------------------------------------
	
	ptr =malloc(sizeof(char));
	
	if(ptr==NULL){
		printf("Null pointer 1\n");
	}else{
		printf("value of ptr %c\n",*ptr );
	}

	*ptr = '\0';
	printf("value ptr %c\n",*ptr);
	
	printf("the value is %d\n",atoi(++ptr));
	
	free((ptr-1));
	
**/
//------------------------------------------------------------------
	/**testing the initialisation of weights and bias
	printf("This is ok here 1\n");
	weightmat =malloc(sizeof(double)*(3*5));
	printf("This is ok here 2\n");

	initialiseWeights(weightmat,15,5);
	printf("This is ok here 3\n");
	for(i = 0; i<15;i++){
		printf(" the weights are  %d  %lf\n",i,weightmat[i] );
	}
	free(weightmat);
	printf("\n");
	biasVec = malloc(sizeof(double)*3);
	initialiseBias(biasVec,3,5);
	for(i = 0; i<15;i++){
		printf(" the bias are  %d  %lf\n",i,biasVec[i] );
	}
	free(biasVec);
	

	//---------------------------------------------------------------

	double *linearActivation = malloc(sizeof(double)*dim);

	double vector[] = { 1,1,1};
	double test3[] ={1,1};
	double bias[] = { 0,0,1,1};

	double vec[] ={2,2,2};
	double test_vec[] ={ 3 ,3,3};

	
	
	double A[] ={ 3, 0, 0,0,3,0,0,0,3};
	

	float dotproduct;
	double result;
	double norm;

	printf("reaches here 1\n");

	**/


	//-----------------------------------------------------------

	/*testing for memory leaks
	arr = (double *) malloc(sizeof(double)*3);
	testfunct(vector);
	for(i = 0; i<srcDim;i++){
		printf(" the output is %lf and the vector   %lf\n",arr[i],vector[i] );
	}
	printf("reaches here 2 \n");

	free(arr);

	//-------------------------------------------------------
	printf("reaches here 3\n");

	cblas_dgemv(CblasRowMajor,CblasNoTrans,dim,srcDim,1,weights,srcDim,test3,1,0,linearActivation,1);
	cblas_daxpy(dim,1,bias,1,linearActivation,1);
	size = sizeof(linearActivation)/sizeof(*linearActivation);
	printf("The size of the activation function is %d \n",size );
	for(i = 0; i<dim;i++){
		printf(" the output is %d  %lf\n",i,linearActivation[i] );
	}
	computeActOfLayer(linearActivation,dim,SIGMOID);
	for(i = 0; i<dim;i++){
		printf(" the output is %d  %lf\n",i,linearActivation[i] );
	}

	computeActOfLayer(linearActivation,dim,TANH);
	for(i = 0; i<dim;i++){
		printf(" the output is %d  %lf\n",i,linearActivation[i] );
	}


	free(linearActivation);


	
	dotproduct = cblas_ddot(3,vec,0,vec,0);
	result = cblas_ddot(3,vec,0,vec,0);
	norm = cblas_dnrm2(3,vec,1);
	//cblas_daxpy(1,1,vec,0,test_vec,0);


	/* y = a A*x + by*/
	/**
	cblas_dgemv(CblasRowMajor,CblasNoTrans,3,3,1,A,3,vector,1,1,vector,1);
	//cblas_dscal(3, 4.323, vector, 1);

	for (i =0; i < 3; i++){
		printf( "x[%d] = %lf ",i,vector[i]);

	}
	printf("The dot product is %f\n", dotproduct);
	printf("The second dot product value is %lf\n", result);
	printf("The norm of the first vector is %lf\n",norm);
	printf("the value should 4  so %lf\n",vector[0]);
	**/

 }

 double computeTanh(double x){
	return 2*(computeSigmoid(2*x))-1;
}

double computeSigmoid(double x){
	double result;
	result = 1/(1+ exp(-1*x));
	return result;
}

void computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc){
	int i = 0;
	switch(actfunc){
		case SIGMOID:
			for (i = 0;i < dim;i++){
				yfeatMat[i] = computeSigmoid(yfeatMat[i]);
			}
			break;
		case TANH:
			for(i = 0; i< dim; i++){
				yfeatMat[i] = computeTanh(yfeatMat[i]);
			}
			break;	
		default:
			break;	
	}
	
}

double drand()   /* uniform distribution, (0..1] */
{return (rand()+1.0)/(RAND_MAX+1.0);
}
/* normal distribution, centered on 0, std dev 1 */
double random_normal() {
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

void initialiseBias(double *biasVec,int dim, int srcDim){
	int i;
	double randm;
	
	for ( i = 0; i<dim;i++){
		randm = random_normal();
		biasVec[i] = randm*(1/sqrt(srcDim));
	}
	

}
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(double *weightMat,int length,int srcDim){
	int i;
	double randm;
	srand((unsigned int)time(NULL));
	for ( i = 0; i<(length);i++){
		randm = random_normal();
		weightMat[i] = randm*1/(sqrt(srcDim));
	}
	
}

void testfunct(double *v){
	int i;
	
	computeActOfLayer(v,3,SIGMOID);
	for (i =0; i<3 ;i++){
		arr[i] = v[i];
	}

}

void setHook(Ptr m, Ptr ptr,int incr){
	Ptr *p;
	printf(" Ptr is %p \n",p);

   p = (Ptr *) m; 
   printf(" Ptr  is %p \n",p);
   printf(" Ptr  size is %lu \n",sizeof(p));
   printf("hello casting success\n");
   p -= incr; 
   printf(" Ptr  is %p \n",p);
   printf("The difference is %lu", (p+2)-p);
   printf(" Ptr  size is %lu \n",sizeof(p));
   *p = ptr;
}

Ptr getHook(Ptr m,int incr){
	Ptr *p;
   p = (Ptr *) m; p -=incr; return *p;
}


