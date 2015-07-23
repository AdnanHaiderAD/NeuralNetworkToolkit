# NeuralNetworkToolkit
A flexible Neural Network ToolKit in C to handle BIG Datasets
This toolkit allows:

• users to construct complex deep neutral networks to do classification on as well as perform unsupervised feature extraction on a variety of large datasets ranging from computer vision, natural language processing to speech data

• Networks to be trained using the standard first order gradient descent learning algorithms

• Networks to be trained using the complex yet powerful second order Hessian Free learning algorithm. THIS Toolkit at present is the only neural network toolkit in C that contains an implementation of HF !

The main source file in ANN.c which is placed under the NeuralNet folder.

This toolkit currently utilises BLAS library to perform efficient low-level computations. Future work involves integrating Cuda to make use of GPU computing

To compile : gcc -DCBLAS -o ANN ANN.c -L /Users/adnan/NeuralNet/BLAS -lcblas -this assumes that the user and downloaded CBLAs and have run the make file

./ANN -C cfg -S /Users/adnan/NeuralNet/sampleDataSets/MNIST/sampleTrainBig.txt -L /Users/adnan/NeuralNet/sampleDataSets/MNIST/sampleTrainLabelsBig.txt -v /Users/adnan/NeuralNet/sampleDataSets/MNIST/samplevalid.txt -vl /Users/adnan/NeuralNet/sampleDataSets/MNIST/samplevalidLabels.tx
