#include <vector>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>
#include "./data/DataSet.h"
#include "./network/LossFunction.h"
#include "./util/Vector.h"
#include "NeuralNetwork.h"
#include "./data/Instance.h"
#include "BasicTests.cpp"
#include <optional>


/**
 * This creates a NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output 
 * (the number of outputs in the XOR dataset) and one hidden layer with 2 nodes.
 *
 * @param DataSet should be the xorDataset
 *
 * @return A NeuralNetwork of the above size
 */
NeuralNetwork createTinyNeuralNetwork(DataSet dataSet, LossFunction lossFunction) {
    std::optional<NeuralNetwork> tinyNN;
    try {
        int expectedWeights = 0;
        if (dataSet.getName() == "iris data") {
            tinyNN = NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{2}, dataSet.getNumberClasses(), lossFunction);
            expectedWeights = 16;
        } else if (dataSet.getName() == "mushroom data") {
            tinyNN = NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{2}, dataSet.getNumberClasses(), lossFunction);
            expectedWeights = 8;
        } else {
            tinyNN = NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{2}, dataSet.getNumberOutputs(), lossFunction);
            expectedWeights = 8;
        }
        tinyNN.value().connectFully();

        int numberWeights = tinyNN.value().getNumberWeights();
        if (numberWeights != expectedWeights) {
            //this network should have 8 weights:
            //6 for the edges
            //2 for the hidden nodes
            throw std::runtime_error("Failed getNumberWeights on tinyNN, returned " + std::to_string(numberWeights) + " which should have been " + std::to_string(expectedWeights) + ".");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
        checkGetSetWeights(tinyNN.value(), "tinyNN");
        
        Log::debug("successfully created tinyNN");
    } catch (std::runtime_error e) {
        Log::fatal("Failed creating tinyNN");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }

    return tinyNN.value();
}


/**
 * This creates a NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output 
 * (the number of outputs in the XOR dataset) and two hidden layers each with 3 nodes.
 *
 * @param DataSet should be the xorDataset
 *
 * @return A NeuralNetwork of the above size
 */
NeuralNetwork createSmallNeuralNetwork(DataSet dataSet, LossFunction lossFunction) {
    std::optional<NeuralNetwork> smallNN;
    try {
        int expectedWeights = 0;
        if (dataSet.getName() == "iris data") {
            smallNN = NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{3, 3}, dataSet.getNumberClasses(), lossFunction);
            smallNN.value().connectFully();
            expectedWeights = 36;
        } else if (dataSet.getName() == "mushroom data") {
            smallNN = NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{3, 3}, dataSet.getNumberClasses(), lossFunction);
            smallNN.value().connectFully();
            expectedWeights = 8;
        } else {
            smallNN = NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{3, 3}, dataSet.getNumberOutputs(), lossFunction);
            expectedWeights = 19;

            //connect the input to the first hidden layer
            smallNN.value().connectNodes(0,0, 1,0);
            smallNN.value().connectNodes(0,0, 1,1);
            smallNN.value().connectNodes(0,1, 1,1);
            smallNN.value().connectNodes(0,1, 1,2);

            //connect the first hidden to the second hidden layer
            smallNN.value().connectNodes(1,0, 2,0);
            smallNN.value().connectNodes(1,0, 2,1);
            smallNN.value().connectNodes(1,1, 2,1);
            smallNN.value().connectNodes(1,1, 2,2);
            smallNN.value().connectNodes(1,2, 2,1);
            smallNN.value().connectNodes(1,2, 2,2);

            //connect the second hidden layer to the output
            smallNN.value().connectNodes(2,0, 3,0);
            smallNN.value().connectNodes(2,1, 3,0);
            smallNN.value().connectNodes(2,2, 3,0);

        }

        int numberWeights = smallNN.value().getNumberWeights();
        if (numberWeights != expectedWeights) {
            //this network should have 19 weights:
            //13 for the edges
            //6 for the hidden nodes
            throw   std::runtime_error("Failed getNumberWeights on smallNN, returned " + std::to_string(numberWeights) + " which should have been " + std::to_string(expectedWeights) + ".");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
        checkGetSetWeights(smallNN.value(), "smallNN");

        Log::debug("successfully created smallNN");
    } catch (std::runtime_error e) {
        Log::fatal("Failed creating smallNN");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }

    return smallNN.value();
}

/**
 * This creates a NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output 
 * (the number of outputs in the XOR dataset) and three hidden layers each with 3, 5 and 4 nodes.
 *
 * @param DataSet should be the xorDataset
 *
 * @return A NeuralNetwork of the above size
 */
NeuralNetwork createLargeNeuralNetwork(DataSet dataSet, LossFunction lossFunction) {
    std::optional<NeuralNetwork> largeNN;
    try {
        int expectedWeights = 0;
        if (dataSet.getName()  == "iris data") {
            largeNN =   NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{3, 5, 4}, dataSet.getNumberClasses(), lossFunction);
            largeNN.value().connectFully();
            expectedWeights = 71;
        } else if (dataSet.getName()  == "mushroom data") {
            largeNN.value() =   NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{3, 5, 4}, dataSet.getNumberClasses(), lossFunction);
            largeNN.value().connectFully();
            expectedWeights = 8;
        } else {
            largeNN.value() =   NeuralNetwork(dataSet.getNumberInputs(), std::vector<int>{3, 5, 4}, dataSet.getNumberOutputs(), lossFunction);
            expectedWeights = 41;

            //connect the input to the first hidden layer (full connections)
            largeNN.value().connectNodes(0,0, 1,0);
            largeNN.value().connectNodes(0,0, 1,1);
            largeNN.value().connectNodes(0,0, 1,2);
            largeNN.value().connectNodes(0,1, 1,0);
            largeNN.value().connectNodes(0,1, 1,1);
            largeNN.value().connectNodes(0,1, 1,2);

            //connect the first hidden to the second hidden layer
            largeNN.value().connectNodes(1,0, 2,0);
            largeNN.value().connectNodes(1,0, 2,1);
            largeNN.value().connectNodes(1,0, 2,2);
            largeNN.value().connectNodes(1,0, 2,4);

            largeNN.value().connectNodes(1,1, 2,1);
            largeNN.value().connectNodes(1,1, 2,2);
            largeNN.value().connectNodes(1,1, 2,3);

            largeNN.value().connectNodes(1,2, 2,0);
            largeNN.value().connectNodes(1,2, 2,2);
            largeNN.value().connectNodes(1,2, 2,3);
            largeNN.value().connectNodes(1,2, 2,4);

            //connect the second hidden layer to the third hidden layer
            largeNN.value().connectNodes(2,0, 3,0);

            largeNN.value().connectNodes(2,1, 3,0);
            largeNN.value().connectNodes(2,1, 3,1);

            largeNN.value().connectNodes(2,2, 3,1);
            largeNN.value().connectNodes(2,2, 3,2);

            largeNN.value().connectNodes(2,3, 3,2);
            largeNN.value().connectNodes(2,3, 3,3);

            largeNN.value().connectNodes(2,4, 3,3);

            //connect the third hidden layer to the output layer
            largeNN.value().connectNodes(3,0, 4,0);
            largeNN.value().connectNodes(3,1, 4,0);
            largeNN.value().connectNodes(3,2, 4,0);
            largeNN.value().connectNodes(3,3, 4,0);
        }


        int numberWeights = largeNN.value().getNumberWeights();
        if (numberWeights != expectedWeights) {
            //this network should have 41 weights:
            //29 for the edges
            //12 for the hidden nodes
            throw   std::runtime_error("Failed getNumberWeights on largeNN, returned " + std::to_string(numberWeights) + " which should have been " + std::to_string(expectedWeights) + ".");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
        checkGetSetWeights(largeNN.value(), "largeNN");

        Log::debug("successfully created largeNN");
    } catch (std::runtime_error e) {
        Log::fatal("Failed creating largeNN");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }

    return largeNN.value();
}


/**
 * Tests neural networks that are dynamically connected (ie., not fully connected)
 * but do not contain layer skipping edges.
 */
void testXORNeuralNetworkDynamic(DataSet xorData) {
    bool passed = true;

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and two hidden layers each with 3 nodes.
        NeuralNetwork xorNeuralNetwork1 = createSmallNeuralNetwork(xorData, LossFunction::NONE);

        Log::info("Passed testXORNeuralNetworDynamic 1");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworDynamic on xorNeuralNetwork1");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and three hidden layers each with 3 5 and 4 nodes.
        NeuralNetwork xorNeuralNetwork2 =  createLargeNeuralNetwork(xorData, LossFunction::NONE);

        Log::info("Passed testXORNeuralNetworkDynamic 2");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworkDynamic on xorNeuralNetwork2");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    if (passed) Log::info("Passed all testXORNeuralNetworkDynamic.");
    else Log::fatal("FAILED testXORNeuralNetworkDynamic!");
}

/**
 * Tests neural networks that are dynamically connected (ie., not fully connected)
 * and do contain layer skipping edges.
 */
void testXORNeuralNetworkDynamicBonus(DataSet xorData) {
    bool passed = true;

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and two hidden layers each with 3 nodes.
        NeuralNetwork xorNeuralNetwork1 =  createSmallNeuralNetwork(xorData, LossFunction::NONE);

        //add some layer skipping connections
        xorNeuralNetwork1.connectNodes(0,0, 2,0);
        xorNeuralNetwork1.connectNodes(0,0, 2,1);
        xorNeuralNetwork1.connectNodes(0,0, 3,0);

        xorNeuralNetwork1.connectNodes(0,1, 2,1);
        xorNeuralNetwork1.connectNodes(0,1, 2,2);
        xorNeuralNetwork1.connectNodes(0,1, 3,0);

        xorNeuralNetwork1.connectNodes(1,0, 3,0);
        xorNeuralNetwork1.connectNodes(1,2, 3,0);

        int expectedWeights = 27;
        int numberWeights = xorNeuralNetwork1.getNumberWeights();
        if (numberWeights != expectedWeights) {
            //this network should have 27 weights:
            //21 for the edges
            //6 for the hidden nodes
            passed = false;
            throw   std::runtime_error("Failed getNumberWeights on xorNeuralNetwork1, returned " + std::to_string(numberWeights) + " which should have been " + std::to_string(expectedWeights) + ".");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
         checkGetSetWeights(xorNeuralNetwork1, "xorNeuralNetwork1");

        Log::info("Passed testXORNeuralNetworDynamicBonus 1");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworDynamicBonus on xorNeuralNetwork1");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and three hidden layers each with 3 5 and 4 nodes.
        NeuralNetwork xorNeuralNetwork2 =  createLargeNeuralNetwork(xorData, LossFunction::NONE);

        //add some layer skipping connections
        xorNeuralNetwork2.connectNodes(0,0, 2,0);
        xorNeuralNetwork2.connectNodes(0,0, 2,1);
        xorNeuralNetwork2.connectNodes(0,0, 2,2);

        xorNeuralNetwork2.connectNodes(0,1, 2,2);
        xorNeuralNetwork2.connectNodes(0,1, 2,3);
        xorNeuralNetwork2.connectNodes(0,1, 2,4);

        xorNeuralNetwork2.connectNodes(0,0, 3,0);
        xorNeuralNetwork2.connectNodes(0,0, 3,2);
        xorNeuralNetwork2.connectNodes(0,1, 3,1);
        xorNeuralNetwork2.connectNodes(0,1, 3,3);

        xorNeuralNetwork2.connectNodes(1,0, 3,0);
        xorNeuralNetwork2.connectNodes(1,0, 3,1);
        xorNeuralNetwork2.connectNodes(1,0, 3,2);
        xorNeuralNetwork2.connectNodes(1,0, 3,3);

        xorNeuralNetwork2.connectNodes(1,1, 3,0);
        xorNeuralNetwork2.connectNodes(1,1, 3,1);
        xorNeuralNetwork2.connectNodes(1,1, 3,2);
        xorNeuralNetwork2.connectNodes(1,1, 3,3);

        xorNeuralNetwork2.connectNodes(1,0, 4,0);
        xorNeuralNetwork2.connectNodes(1,1, 4,0);

        int expectedWeights = 61;
        int numberWeights = xorNeuralNetwork2.getNumberWeights();
        if (numberWeights != expectedWeights) {
            //this network should have 61 weights:
            //49 for the edges
            //12 for the hidden nodes
            passed = false;
            throw   std::runtime_error("Failed getNumberWeights on xorNeuralNetwork2, returned " + std::to_string(numberWeights) + " which should have been " + std::to_string(expectedWeights) + ".");
        }

        //set the weights and then get the weights to make
        //sure the weights we get are the same and in the
        //same order as the weights we set
         checkGetSetWeights(xorNeuralNetwork2, "xorNeuralNetwork2");

        Log::info("Passed testXORNeuralNetworkDynamicBonus 2");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworkDynamicBonus on xorNeuralNetwork2");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    if (passed) Log::info("Passed all testXORNeuralNetworkDynamicBonus.");
    else Log::fatal("FAILED testXORNeuralNetworkDynamicBonus!");
}


/**
 * Tests the forward pass onneural networks that are fully and 
 * dynamically connected and do not contain layer skipping edges.
 */
void textXORNeuralNetworkForwardPass(DataSet xorData) {
    bool passed = true;

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and one hidden layer with 3 nodes.
        NeuralNetwork tinyNN =  createTinyNeuralNetwork(xorData, LossFunction::NONE);

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(std::vector<double>{1.0}, std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(std::vector<double>{0.0}, std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(std::vector<double>{0.0}, std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(std::vector<double>{1.0}, std::vector<double>{1.0, 1.0});

        //this network has 8 weights, generate some weights that we have a solution for
        //to test that the forward pass is workign right
        tinyNN.setWeights(std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
        tinyNN.reset();
        tinyNN.forwardPass(instance00);

        std::vector<double> outputValues = tinyNN.getOutputValues();

        double expectedOutput = 0.6815196954796512;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw std::runtime_error("Failed forward pass test on tinyNN and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        tinyNN.reset();
        tinyNN.forwardPass(instance10);

        outputValues = tinyNN.getOutputValues();

        expectedOutput = 0.70997611281561; 
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on tinyNN and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        tinyNN.reset();
        tinyNN.forwardPass(instance01);

        outputValues = tinyNN.getOutputValues();

        expectedOutput = 0.7386225071588598;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on tinyNN and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        tinyNN.reset();
        tinyNN.forwardPass(instance11);

        outputValues = tinyNN.getOutputValues();

        expectedOutput = 0.7538323614392307;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on tinyNN and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        Log::info("Passed testXORNeuralNetworkForwardPass 1");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworkForwardPass on tinyNN");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }


    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and one hidden layer with 3 nodes.
        NeuralNetwork xorNeuralNetwork1 =   NeuralNetwork(xorData.getNumberInputs(), std::vector<int>{3}, xorData.getNumberOutputs(), LossFunction::NONE);
        //make this a fully connected network
        xorNeuralNetwork1.connectFully();

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(std::vector<double>{1.0}, std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(std::vector<double>{0.0}, std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(std::vector<double>{0.0}, std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(std::vector<double>{1.0}, std::vector<double>{1.0, 1.0});

        //this network has 12 weights, generate some weights that we have a solution for
        //to test that the forward pass is workign right
        xorNeuralNetwork1.setWeights(std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2});
        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance00);

        std::vector<double> outputValues = xorNeuralNetwork1.getOutputValues();

        double expectedOutput = 0.8966357844617932;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance10);

        outputValues = xorNeuralNetwork1.getOutputValues();

        expectedOutput = 0.9163801544763723;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance01);

        outputValues = xorNeuralNetwork1.getOutputValues();

        expectedOutput = 0.9339025390085295;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance11);

        outputValues = xorNeuralNetwork1.getOutputValues();

        expectedOutput = 0.9396544694550081;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }



        Log::info("Passed testXORNeuralNetworkForwardPass 1");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworkForwardPass on xorNeuralNetwork1");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and two hidden layers with 3 and 4 nodes.
        NeuralNetwork xorNeuralNetwork2 =   NeuralNetwork(xorData.getNumberInputs(), std::vector<int>{3, 4}, xorData.getNumberOutputs(), LossFunction::NONE);
        //make this a fully connected network
        xorNeuralNetwork2.connectFully();

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(std::vector<double>{1.0}, std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(std::vector<double>{0.0}, std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(std::vector<double>{0.0}, std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(std::vector<double>{1.0}, std::vector<double>{1.0, 1.0});

        //this network has 29 weights, generate some weights that we have a solution for
        //to test that the forward pass is workign right
        std::vector<double> test_weights(29);
        for (int i = 0; i < test_weights.size(); i++) {
            test_weights[i] = i * 0.05;
        }

        xorNeuralNetwork2.setWeights(test_weights);
        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance00);

        std::vector<double> outputValues = xorNeuralNetwork2.getOutputValues();

        double expectedOutput = 0.9925468726777413;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance10);

        outputValues = xorNeuralNetwork2.getOutputValues();

        expectedOutput = 0.9926518886068347;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance10, output was: " + std::to_string(outputValues[0] )+ " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance01);

        outputValues = xorNeuralNetwork2.getOutputValues();

        expectedOutput = 0.9928523218399518;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance11);

        outputValues = xorNeuralNetwork2.getOutputValues();

        expectedOutput = 0.9928978880221059;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        Log::info("Passed testXORNeuralNetworkForwardPass 2");

    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworkForwardPass on xorNeuralNetwork2");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }


}


/**
 * Tests the forward pass onneural networks that are fully and 
 * dynamically connected and do contain layer skipping edges.
 */
void textXORNeuralNetworkForwardPassBonus(DataSet xorData) {
    bool passed = true;

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and two hidden layers each with 3 nodes.
        NeuralNetwork xorNeuralNetwork1 =  createSmallNeuralNetwork(xorData, LossFunction::NONE);

        //add some layer skipping connections
        xorNeuralNetwork1.connectNodes(0,0, 2,0);
        xorNeuralNetwork1.connectNodes(0,0, 2,1);
        xorNeuralNetwork1.connectNodes(0,0, 3,0);

        xorNeuralNetwork1.connectNodes(0,1, 2,1);
        xorNeuralNetwork1.connectNodes(0,1, 2,2);
        xorNeuralNetwork1.connectNodes(0,1, 3,0);

        xorNeuralNetwork1.connectNodes(1,0, 3,0);
        xorNeuralNetwork1.connectNodes(1,2, 3,0);

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(std::vector<double>{1.0}, std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(std::vector<double>{0.0}, std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(std::vector<double>{0.0}, std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(std::vector<double>{1.0}, std::vector<double>{1.0, 1.0});

        //this network has 27 weights, generate some weights that we have a solution for
        //to test that the forward pass is working right
        std::vector<double> test_weights = std::vector<double>(27);
        for (int i = 0; i < test_weights.size(); i++) {
            test_weights[i] = i * 0.05;
        }

        xorNeuralNetwork1.setWeights(test_weights);
        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance00);

        std::vector<double> outputValues = xorNeuralNetwork1.getOutputValues();

        double expectedOutput = 0.9879060915673169;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance10);

        outputValues = xorNeuralNetwork1.getOutputValues();

        expectedOutput = 0.9903812468156195;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance01);

        outputValues = xorNeuralNetwork1.getOutputValues();

        expectedOutput =  0.993345775069195;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork1.reset();
        xorNeuralNetwork1.forwardPass(instance11);

        outputValues = xorNeuralNetwork1.getOutputValues();

        expectedOutput = 0.9946924618665479;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork1 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 1 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        Log::info("Passed testXORNeuralNetworForwardPassBonus 1");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworForwardPassBonus on xorNeuralNetwork1");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    try {
        //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
        //in the XOR dataset) and three hidden layers each with 3 5 and 4 nodes.
        NeuralNetwork xorNeuralNetwork2 =  createLargeNeuralNetwork(xorData, LossFunction::NONE);

        //add some layer skipping connections
        xorNeuralNetwork2.connectNodes(0,0, 2,0);
        xorNeuralNetwork2.connectNodes(0,0, 2,1);
        xorNeuralNetwork2.connectNodes(0,0, 2,2);

        xorNeuralNetwork2.connectNodes(0,1, 2,2);
        xorNeuralNetwork2.connectNodes(0,1, 2,3);
        xorNeuralNetwork2.connectNodes(0,1, 2,4);

        xorNeuralNetwork2.connectNodes(0,0, 3,0);
        xorNeuralNetwork2.connectNodes(0,0, 3,2);
        xorNeuralNetwork2.connectNodes(0,1, 3,1);
        xorNeuralNetwork2.connectNodes(0,1, 3,3);

        xorNeuralNetwork2.connectNodes(1,0, 3,0);
        xorNeuralNetwork2.connectNodes(1,0, 3,1);
        xorNeuralNetwork2.connectNodes(1,0, 3,2);
        xorNeuralNetwork2.connectNodes(1,0, 3,3);

        xorNeuralNetwork2.connectNodes(1,1, 3,0);
        xorNeuralNetwork2.connectNodes(1,1, 3,1);
        xorNeuralNetwork2.connectNodes(1,1, 3,2);
        xorNeuralNetwork2.connectNodes(1,1, 3,3);

        xorNeuralNetwork2.connectNodes(1,0, 4,0);
        xorNeuralNetwork2.connectNodes(1,1, 4,0);

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(std::vector<double>{1.0}, std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(std::vector<double>{0.0}, std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(std::vector<double>{0.0}, std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(std::vector<double>{1.0}, std::vector<double>{1.0, 1.0});

        //this network has 61 weights, generate some weights that we have a solution for
        //to test that the forward pass is workign right
        std::vector<double> test_weights = std::vector<double>(61);
        for (int i = 0; i < test_weights.size(); i++) {
            test_weights[i] = (i * 0.05) - 1.5;
        }

        xorNeuralNetwork2.setWeights(test_weights);
        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance00);

        std::vector<double> outputValues = xorNeuralNetwork2.getOutputValues();

        double expectedOutput = 0.9957885338260906;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance00, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance10);

        outputValues = xorNeuralNetwork2.getOutputValues();

        expectedOutput = 0.9798649068685976;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance10, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance01);

        outputValues = xorNeuralNetwork2.getOutputValues();

        expectedOutput = 0.9925468048166154;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance01, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        xorNeuralNetwork2.reset();
        xorNeuralNetwork2.forwardPass(instance11);

        outputValues = xorNeuralNetwork2.getOutputValues();

        expectedOutput = 0.6873877161921984;
        if (! closeEnough(outputValues[0], expectedOutput)) {
            throw   std::runtime_error("Failed forward pass test on xorNeuralNetwork2 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        } else {
            Log::debug("Passed forward pass on xorNeuralNetwork 2 and instance11, output was: " + std::to_string(outputValues[0]) + " and expected was " + std::to_string(expectedOutput));
        }

        Log::info("Passed testXORNeuralNetworkForwardPassBonus 2");
    } catch (std::runtime_error e) {
        Log::fatal("Failed testXORNeuralNetworkForwardPassBonus on xorNeuralNetwork2");
        Log::fatal("Threw exception: " + (std::string) e.what());
        passed = false;
    }

    if (passed) Log::info("Passed all testXORNeuralNetworkForwardPassBonus.");
    else Log::fatal("FAILED testXORNeuralNetworkForwardPassBonus!");
}

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