#include "./data/DataSet.h"
#include "./network/LossFunction.h"
#include "./util/PA11TestsUtils.h"

int main(int argc, char* argv[]) {
    DataSet xorData = DataSet("xor data", "./datasets/xor.txt");

    //test creating a tiny NN
    //It will not work until you correctly implement the 
    //NeuralNetwork.connectNodes(int, int, int, int) method.
    createTinyNeuralNetwork(xorData, LossFunction::NONE);

    //test creating a small NN
    //It will not work until you correctly implement the 
    //NeuralNetwork.connectNodes(int, int, int, int) method.
    createSmallNeuralNetwork(xorData, LossFunction::NONE);

    //test creating a larger NN
    //It will not work until you correctly implement the 
    //NeuralNetwork.connectNodes(int, int, int, int) method.
    createLargeNeuralNetwork(xorData, LossFunction::NONE);

    //This tests creating a non-fully connected neural network.
    //It will not work until you correctly implement the 
    //NeuralNetwork.connectNodes(int, int, int, int) method.
    testXORNeuralNetworkDynamic(xorData);

    //This tests creating a non-fully connected neural network.
    //It will not work until you correctly implement the 
    //bonus for the NeuralNetwork.connectNodes(int, int, int, int) 
    //method.
    testXORNeuralNetworkDynamicBonus(xorData);

    //This tests the forward pass. It will not work until you
    //correctly implement the NeuralNetworks.forwardPass(Instance) 
    //method
    textXORNeuralNetworkForwardPass(xorData);

    //This tests the forward pass. It will not work until you
    //correctly implement the bonus for the 
    //NeuralNetworks.forwardPass(Instance) method
    textXORNeuralNetworkForwardPassBonus(xorData);
}