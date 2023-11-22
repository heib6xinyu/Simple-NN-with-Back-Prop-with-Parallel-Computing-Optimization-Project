#include <vector>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>
#include "./data/DataSet.h"
#include "./data/Instance.h"
#include "./network/NeuralNetwork.h"
#include "./network/LossFunction.h"
#include "Vector.h"


void testLoadingXOR() {
    bool passed = true;

    LogInfo("Loading the xor.txt file as a DataSet.");
    try {
        DataSet xorData = DataSet("xor data", "./datasets/xor.txt");

        //Test getting each instance individually
        //make sure the 4 different instances from XOR were read correctly
        Instance i0 = xorData.getInstance(0);
        if (!i0.equals(std::vector<double>{0}, std::vector<double>{0, 0})) {
            LogError("testLoadingXOR was not correct on getInstance 0.");
            LogError("\tinstance was:     " + i0.toString());
            LogError("\tshould have been: [0 : 0, 0]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstance 0.");
        }

        Instance i1 = xorData.getInstance(1);
        if (!i1.equals(std::vector<double>{1}, std::vector<double>{1, 0})) {
            LogError("testLoadingXOR was not correct on getInstance 1.");
            LogError("\tinstance was:     " + i1.toString());
            LogError("\tshould have been: [1 : 1, 0]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstance 1.");
        }

        Instance i2 = xorData.getInstance(2);
        if (!i2.equals(std::vector<double>{1}, std::vector<double>{0, 1})) {
            LogError("testLoadingXOR was not correct on getInstance 2.");
            LogError("\tinstance was:     " + i2.toString());
            LogError("\tshould have been: [1 : 0, 1]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstance 2.");
        }

        Instance i3 = xorData.getInstance(3);
        if (!i3.equals(std::vector<double>{0}, std::vector<double>{1, 1})) {
            LogError("testLoadingXOR was not correct on getInstance 3.");
            LogError("\tinstance was:     " + i3.toString());
            LogError("\tshould have been: [0 : 1, 1]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstance 3.");
        }

        //Gets all instances as once to test the getInstances Method
        std::vector<Instance> is = xorData.getInstances(0,4);
        if (!is[0].equals(std::vector<double>{0}, std::vector<double>{0, 0})) {
            LogError("testLoadingXOR was not correct on getInstances 0.");
            LogError("\tinstance was:     " + i0.toString());
            LogError("\tshould have been: [0 : 0, 0]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstances 0.");
        }

        if (!is[1].equals(std::vector<double>{1}, std::vector<double>{1, 0})) {
            LogError("testLoadingXOR was not correct on getInstances 1.");
            LogError("\tinstance was:     " + i1.toString());
            LogError("\tshould have been: [0 : 1 , 0]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstances 1.");
        }

        if (!is[2].equals(std::vector<double>{1}, std::vector<double>{0, 1})) {
            LogError("testLoadingXOR was not correct on getInstances 2.");
            LogError("\tinstance was:     " + i2.toString());
            LogError("\tshould have been: [1 : 0, 1]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstances 2.");
           }

        if (!is[3].equals(std::vector<double>{0}, std::vector<double>{1, 1})) {
            LogError("testLoadingXOR was not correct on getInstances 3.");
            LogError("\tinstance was:     " + i3.toString());
            LogError("\tshould have been: [0 : 1, 1]");
            passed = false;
        } else {
            LogTrace("testLoadingXOR passed getInstances 3.");
        }
    } catch (std::exception e) {
        LogFatal("Exception occurred in testLoadingXOR: " + (std::string) e.what());
        passed = false;
    }

    if (passed) {
        LogInfo("Passed testLoadingXOR.");
    } else {
        LogFatal("FAILED testLoadingXOR!");
    }
}

void testXORNeuralNetwork() {
    bool passed = true;

    //creating this data set should work correctly if the previous test passed.
    DataSet xorData = DataSet("xor data", "./datasets/xor.txt");

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and one hidden layer with 3 nodes.
        NeuralNetwork xorNeuralNetwork1 = NeuralNetwork(xorData.getNumberInputs(), std::vector<int>{3}, xorData.getNumberOutputs(), LossFunction::NONE);
        //make this a fully connected network
        xorNeuralNetwork1.connectFully();

        int numberWeights = xorNeuralNetwork1.getNumberWeights();
        if (numberWeights != 12) {
            //this network should have 12 weights, 9 for the edges and 3 for the 3 hidden nodes
            passed = false;
            throw std::runtime_error("Failed getNumberWeights on XOR Neural Network 1, returned " + std::to_string(numberWeights) + " which should have been 29.");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
        checkGetSetWeights(xorNeuralNetwork1, "xorNeuralNetwork1");

        LogInfo("Passed testXORNeuralNetwork 1");
    } catch (std::exception e) {
        LogFatal("Failed testXORNeuralNetwork on Neural Network 1");
        LogFatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and two hidden layers with 3 and 4 nodes.
        NeuralNetwork xorNeuralNetwork2 = NeuralNetwork(xorData.getNumberInputs(), std::vector<int>{3, 4}, xorData.getNumberOutputs(), LossFunction::NONE);
        //make this a fully connected network
        xorNeuralNetwork2.connectFully();

        int numberWeights = xorNeuralNetwork2.getNumberWeights();
        if (numberWeights != 29) {
            //this network should have 29 weights:
            //6 from the 2 input to the 3 in the first hidden layer
            //12 from the 3 on hidden layer 1 to the 4 on hidden layer 2
            //4 from the 4 on hidden layer 2 to the 1 on the output layer
            //7 for the 7 hidden node biases
            passed = false;
            throw std::runtime_error("Failed getNumberWeights on XOR Neural Network 2, returned " + std::to_string(numberWeights) + " which should have been 29.");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
        checkGetSetWeights(xorNeuralNetwork2, "xorNeuralNetwork2");

        LogInfo("Passed testXORNeuralNetwork 2");
    } catch (std::exception e) {
        LogFatal("Failed testXORNeuralNetwork on Neural Network 2");
        LogFatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }


    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and three hidden layers with 3, 2 and 4 nodes.
        NeuralNetwork xorNeuralNetwork3 = NeuralNetwork(xorData.getNumberInputs(), std::vector<int>{3,2,4}, xorData.getNumberOutputs(), LossFunction::NONE);
        //make this a fully connected network
        xorNeuralNetwork3.connectFully();

        int numberWeights = xorNeuralNetwork3.getNumberWeights();
        if (numberWeights != 33) {
            //this network should have 26 weights:
            //6 from the 2 input to the 3 in the first hidden layer
            //6 from the 3 on hidden layer 1 to the 2 on hidden layer 2
            //8 from the 2 on hidden layer 2 to the 4 on hidden layer 3
            //4 from the 4 on hidden layer 3 to the 1 on the output layer
            //9 for the 9 hidden node biases
            passed = false;
            throw std::runtime_error("Failed getNumberWeights on XOR Neural Network 3, returned " + std::to_string(numberWeights) + " which should have been 33.");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
        checkGetSetWeights(xorNeuralNetwork3, "xorNeuralNetwork3");

        LogInfo("Passed testXORNeuralNetwork 3");
    } catch (std::exception e) {
        LogFatal("Failed testXORNeuralNetwork on Neural Network 3");
        LogFatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    if (passed) {
        LogInfo("Passed testXORNeuralNetwork.");
    } else {
        LogFatal("FAILED testXORNeuralNetwork!");
    }
}

bool closeEnough(double n1, double n2) {
    return abs(n1-n2) < 2e-6;
}

bool vectorsCloseEnough(std::vector<double> v1, std::vector<double> v2) {
    for (size_t i = 0; i < v1.size(); ++i) {
        if (!closeEnough(v1[i], v2[i])) { 
            return false;
        }
    }
    return true;
}

bool gradientsCloseEnough(std::vector<double> g1, std::vector<double> g2) {
    double relativeError = Vector::norm(Vector::subtractVector(g1, g2)) / std::max(Vector::norm(g1), Vector::norm(g2));
    
    if (relativeError >= 1e-4) {
        LogError("relativeError bad: " + std::to_string(relativeError));
        for (int i = 0; i < g1.size(); ++i) {
            LogError("\tg1[" + std::to_string(i) + "]: " + std::to_string(g1[i]) + ", g2[" + std::to_string(i) + "]: " + std::to_string(g2[i]) + ", difference: " + std::to_string(abs(g1[i] - g2[i])));
        }
    } else if (relativeError >= 1e-5) {
        LogWarning("relativeError probably bad: " + std::to_string(relativeError));
        for (int i = 0; i < g1.size(); ++i) {
            LogTrace("\tg1[" + std::to_string(i) + "]: " + std::to_string(g1[i]) + ", g2[" + std::to_string(i) + "]: " + std::to_string(g2[i]) + ", difference: " + std::to_string(abs(g1[i] - g2[i])));
        }
    } else if (relativeError >= 1e-7) {
        LogDebug("relativeError might be bad: " + std::to_string(relativeError));
        for (int i = 0; i < g1.size(); ++i) {
            LogTrace("\tg1[" + std::to_string(i) + "]: " + std::to_string(g1[i]) + ", g2[" + std::to_string(i) + "]: " + std::to_string(g2[i]) + ", difference: " + std::to_string(abs(g1[i] - g2[i])));
        }
    }
    return relativeError <= 1e-5;
}

void checkGetSetWeights(NeuralNetwork network, std::string networkName) {
    LogDebug("Testing get/set weights on neural network '" + networkName + "'");
    int numberWeights = network.getNumberWeights();
    std::vector<double> testWeights(numberWeights);
    for (int i = 0; i < numberWeights; ++i) {
        testWeights[i] = (double) i;
    }

    network.setWeights(testWeights);

    std::vector<double> testWeights2 = network.getWeights();

    bool passed = true;
    for (int i = 0; i < numberWeights; ++i) {
        LogTrace("testWeights[" + std::to_string(i) + "]: " + std::to_string(testWeights[i]) 
            + ", testWeights2[" + std::to_string(i) + "]: " + std::to_string(testWeights2[i]));
        if (testWeights[i] != testWeights2[i]) {
            throw std::runtime_error("Failed getSetWeights test on " + networkName + ", testWeights[" + std::to_string(i) + "] was " 
                + std::to_string(testWeights[i]) + " and testWeights2[" + std::to_string(i) + "] was " + std::to_string(testWeights2[i]) + ".");
        }
    }
}

void LogError(std::string msg) {
    printf("Error: %s", msg.c_str());
}

void LogWarning(std::string msg) {
    printf("Warning: %s", msg.c_str());
}

void LogDebug(std::string msg) {
    printf("Debug: %s", msg.c_str());
}

void LogTrace(std::string msg) {
    printf("Trace: %s", msg.c_str());
}

void LogInfo(std::string msg) {
    printf("Info: %s", msg.c_str());
}

void LogFatal(std::string msg) {
    printf("Fatal: %s", msg.c_str());
}

int main(int argc, char* argv[]) {
    testLoadingXOR();
    testXORNeuralNetwork();
}