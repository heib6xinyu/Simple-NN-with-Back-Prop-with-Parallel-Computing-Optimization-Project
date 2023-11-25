#include "../data/DataSet.h"
#include "../network/LossFunction.h"
#include "../network/NeuralNetwork.h"
#include "../data/Instance.h"

void testTinyGradientNumeric(DataSet xorData);
void testSmallGradientNumeric(DataSet xorData);
void testLargeGradientNumeric(DataSet xorData);
void testTinyGradients(DataSet dataSet, LossFunction lossFunction);
void testSmallGradients(DataSet dataSet, LossFunction lossFunction);
void testLargeGradients(DataSet dataSet, LossFunction lossFunction);
void testNetworkOnInstances(NeuralNetwork nn, std::vector<Instance> instances, std::string description);
void testTinyGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction);
void testSmallGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction);
void testLargeGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction);
double random_double();