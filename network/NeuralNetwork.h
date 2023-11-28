#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include "Node.h"  // Make sure this path is correct
#include "Edge.h"  // Make sure this path is correct
#include "LossFunction.h"  // Enum or class needs to be defined
#include "../data/Instance.h" // Forward declare Instance if it's a class

class NeuralNetwork {
private:
    LossFunction lossFunction;
    int numberWeights;
    std::vector<std::vector<Node>> layers;

public:
    NeuralNetwork(int inputLayerSize, const std::vector<int>& hiddenLayerSizes, int outputLayerSize, LossFunction lossFunc);
   

    int getNumberWeights() const;
    void reset();
    std::vector<double> getWeights() const;
    void setWeights(std::vector<double>& newWeights);
    std::vector<double> getDeltas() const;
    void connectFully();
    void connectNodes(int inputLayer, int inputNumber, int outputLayer, int outputNumber);
    void initializeRandomly(double bias);
    double forwardPass(const Instance& instance);
    double forwardPass(const std::vector<Instance>& instances);
    double calculateAccuracy(const std::vector<Instance>& instances);
    std::vector<double> getOutputValues() const;
    std::vector<double> getNumericGradient(const Instance& instance);
    std::vector<double> getNumericGradient(const std::vector<Instance>& instances);
    void backwardPass();
    std::vector<double> getGradient(const Instance& instance);
    std::vector<double> getGradient(const std::vector<Instance>& instances);
};

#endif // NEURAL_NETWORK_H
