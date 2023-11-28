#include "../network/NeuralNetwork.h"
#include "../network/LossFunction.h"
#include "../data/DataSet.h"


NeuralNetwork createTinyNeuralNetwork(DataSet dataSet, LossFunction lossFunction);
NeuralNetwork createSmallNeuralNetwork(DataSet dataSet, LossFunction lossFunction); 
NeuralNetwork createLargeNeuralNetwork(DataSet dataSet, LossFunction lossFunction); 
void testXORNeuralNetworkDynamic(DataSet xorData); 
void testXORNeuralNetworkDynamicBonus(DataSet xorData); 
void textXORNeuralNetworkForwardPass(DataSet xorData); 
void textXORNeuralNetworkForwardPassBonus(DataSet xorData); 