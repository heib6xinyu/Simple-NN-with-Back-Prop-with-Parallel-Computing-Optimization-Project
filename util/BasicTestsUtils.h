#ifndef BASICTESTSUTILS_H
#define BASICTESTSUTILS_H

#include <vector>
#include <string>
#include "../network/NeuralNetwork.h"

bool closeEnough(double n1, double n2);
bool vectorsCloseEnough(std::vector<double> v1, std::vector<double> v2);
bool gradientsCloseEnough(std::vector<double> g1, std::vector<double> g2);
void checkGetSetWeights(NeuralNetwork network, std::string networkName);
void testLoadingXOR();
void testXORNeuralNetwork();

#endif