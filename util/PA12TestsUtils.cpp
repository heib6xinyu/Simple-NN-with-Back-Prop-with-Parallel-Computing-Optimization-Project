#include "PA12TestsUtils.h"
#include <vector>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>
#include "../data/DataSet.h"
#include "../network/LossFunction.h"
#include "Vector.h"
#include "Log.h"
#include "../network/NeuralNetwork.h"
#include "../data/Instance.h"
#include "BasicTestsUtils.h"
#include "PA11TestsUtils.h"
#include <stdlib.h>


/**
 * NUMBER_REPEATS is used to repeatedly test the gradient comparison
 * between the gradient calculated by the finite difference method
 * and the gradient calculated using backprop. We can repeatedly
 * do this with random waits to be quite sure we've implemented both
 * methods correctly.
 */
int NUMBER_REPEATS = 100;

/**
 * This tests calculation of the numeric gradient for
 * the tiny fully connected neural network generated
 * by  createTinyNeuralNetwork()
 */
void testTinyGradientNumeric(DataSet xorData) {
    try {
        NeuralNetwork tinyNN =  createTinyNeuralNetwork(xorData,  LossFunction::NONE);

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(   std::vector<double>{1.0},    std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(   std::vector<double>{0.0},    std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(   std::vector<double>{0.0},    std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(   std::vector<double>{1.0},    std::vector<double>{1.0, 1.0});

         std::vector<double> weights = std::vector<double>(tinyNN.getNumberWeights());

        for (int i = 0; i < weights.size(); ++i) {
            //give the test weights a spread of positive and negative values
            weights[i] = (-1 * (i%2)) * 0.05 * i;
        }

        tinyNN.setWeights(weights);
        
        std::vector<double> calculatedGradient =    std::vector<double>{0.0, 0.0, 0.0, 0.0, -0.06250000073038109, 0.0, -0.0875000000233328, 0.0};
        std::vector<double> numericGradient = tinyNN.getNumericGradient(instance00);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw std::runtime_error("Gradients not close enough on testTinyGradientNumeric, instance 00!");
        }
        Log::info("passed testTinyGradientNumeric on instance00");

        calculatedGradient =    std::vector<double>{-0.06249522344070613, -0.0872749678082485, 0.0, 0.0, -0.06249522344070613, 0.0, -0.0872749678082485, -0.012488639566932136};
        numericGradient = tinyNN.getNumericGradient(instance10);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testTinyGradientNumeric, instance 10!");
        }
        Log::info("passed testTinyGradientNumeric on instance10");

        calculatedGradient =    std::vector<double>{0.0, 0.0, -0.06245759021084041, -0.08550240071514281, -0.06245759021084041, 0.0, -0.08550240071514281, -0.03719600183416105};
        numericGradient = tinyNN.getNumericGradient(instance01);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testTinyGradientNumeric, instance 01!");
        }
        Log::info("passed testTinyGradientNumeric on instance01");

        calculatedGradient =    std::vector<double>{-0.06242549310808698, -0.08399108686329981, -0.06242549310808698, -0.08399108686329981, -0.06242549310808698, 0.0, -0.08399108686329981, -0.04928500607626063};
        numericGradient = tinyNN.getNumericGradient(instance11);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testTinyGradientNumeric, instance 11!");
        }
        Log::info("passed testTinyGradientNumeric on instance11");

    } catch (const std::exception& e) {
        Log::fatal("Failed testTinyGradientNumeric");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}



/**
 * This tests calculation of the numeric gradient for
 * the small fully connected neural network generated
 * by  createSmallNeuralNetwork()
 */
void testSmallGradientNumeric(DataSet xorData) {
    try {
        NeuralNetwork smallNN =  createSmallNeuralNetwork(xorData,  LossFunction::NONE);

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(   std::vector<double>{1.0},    std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(   std::vector<double>{0.0},    std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(   std::vector<double>{0.0},    std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(   std::vector<double>{1.0},    std::vector<double>{1.0, 1.0});

         std::vector<double> weights = std::vector<double>(smallNN.getNumberWeights());

        for (int i = 0; i < weights.size(); i++) {
            //give the test weights a spread of positive and negative values
            weights[i] = (-1 * (i%2)) * 0.05 * i;
        }

        smallNN.setWeights(weights);

         std::vector<double> calculatedGradient =    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723891304958, 0.0, -0.1508745262057687};
         std::vector<double> numericGradient = smallNN.getNumericGradient(instance00);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testSmallGradientNumeric, instance 00!");
        }
        Log::info("passed testSmallGradientNumeric on instance00");

        calculatedGradient =    std::vector<double>{0.0, 6.106226635438361E-9, 0.0, 0.0, 0.0, 0.0, 0.0, 6.106226635438361E-9, 0.0, 6.106226635438361E-9, 6.106226635438361E-9, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723835793807, -1.609823385706477E-8, -0.147720963794562};
        numericGradient = smallNN.getNumericGradient(instance10);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testSmallGradientNumeric, instance 10!");
        }
        Log::info("passed testSmallGradientNumeric on instance10");

        //calculatedGradient =    std::vector<double>{0.0, 0.0, 1.27675647831893E-8, 0.0, 0.0, 0.0, 0.0, 1.27675647831893E-8, 0.0, 1.0547118733938987E-8, 1.0547118733938987E-8, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723780282655, -3.164135620181696E-8, -0.15087452565065718};
        calculatedGradient =    std::vector<double>{
            0.0,
            0.0,
            1.27675647831893E-8,
            0.0,
            0.0,
            0.0,
            0.0,
            1.27675647831893E-8,
            0.0,
            1.0547118733938987E-8,
            0.0,
            0.0,
            4.440892098500626E-9,
            0.0,
            -0.14291749173001023,
            0.0,
            -0.14593435015974876,
            -3.164135620181696E-8,
            -0.15087452565065718
        };
        numericGradient = smallNN.getNumericGradient(instance01);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw std::runtime_error("Gradients not close enough on testSmallGradientNumeric, instance 01!");
        }
        Log::info("passed testSmallGradientNumeric on instance01");

        calculatedGradient =    std::vector<double>{
            0.0,
            1.887379141862766E-8,
            1.887379141862766E-8,
            0.0,
            0.0,
            0.0,
            0.0,
            1.887379141862766E-8,
            0.0,
            1.8318679906315083E-8,
            0.0,
            0.0,
            7.2164496600635175E-9,
            0.0,
            -0.14291749117489871,
            0.0,
            -0.14593435015974876,
            -4.884981308350689E-8,
            -0.1477209632394505
        };
        numericGradient = smallNN.getNumericGradient(instance11);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testSmallGradientNumeric, instance 11!");
        }
        Log::info("passed testSmallGradientNumeric on instance11");

    } catch (const std::exception& e) {
        Log::fatal("Failed testSmallGradientNumeric");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}


/**
 * This tests calculation of the numeric gradient for
 * the large fully connected neural network generated
 * by  createLargeNeuralNetwork()
 */
void testLargeGradientNumeric(DataSet xorData) {
    try {
        NeuralNetwork largeNN =  createLargeNeuralNetwork(xorData,  LossFunction::NONE);

        //test the 4 possible XOR instances
        Instance instance00 =   Instance(   std::vector<double>{1.0},    std::vector<double>{0.0, 0.0});
        Instance instance10 =   Instance(   std::vector<double>{0.0},    std::vector<double>{1.0, 0.0});
        Instance instance01 =   Instance(   std::vector<double>{0.0},    std::vector<double>{0.0, 1.0});
        Instance instance11 =   Instance(   std::vector<double>{1.0},    std::vector<double>{1.0, 1.0});

         std::vector<double> weights = std::vector<double>(largeNN.getNumberWeights());

        for (int i = 0; i < weights.size(); i++) {
            //give the test weights a spread of positive and negative values
            weights[i] = (-1 * (i%2)) * 0.05 * i;
        }

        largeNN.setWeights(weights);

         std::vector<double> calculatedGradient =    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.21879212330766507, 0.0, -0.2400798448931596};
         std::vector<double> numericGradient = largeNN.getNumericGradient(instance00);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testLargeGradientNumeric, instance 00!");
        }
        Log::info("passed testLargeGradientNumeric on instance00");

        calculatedGradient =    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.22038990632466948, 0.0, -0.2400798448931596};
        numericGradient = largeNN.getNumericGradient(instance10);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testLargeGradientNumeric, instance 10!");
        }
        Log::info("passed testLargeGradientNumeric on instance10");

        calculatedGradient =    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.22926425757852087, 0.0, -0.2400798504442747};
        numericGradient = largeNN.getNumericGradient(instance01);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testLargeGradientNumeric, instance 01!");
        }
        Log::info("passed testLargeGradientNumeric on instance01");

        calculatedGradient =    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.2304554880261378, -5.551115123125783E-9, -0.2400798504442747};
        numericGradient = largeNN.getNumericGradient(instance11);
        if (! gradientsCloseEnough(calculatedGradient, numericGradient)) {
            throw   std::runtime_error("Gradients not close enough on testLargeGradientNumeric, instance 11!");
        }
        Log::info("passed testLargeGradientNumeric on instance11");


    } catch (const std::exception& e) {
        Log::fatal("Failed testLargeGradientNumeric");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}

void testTinyGradients(DataSet dataSet, LossFunction lossFunction) {
    try {
        NeuralNetwork tinyNN =  createTinyNeuralNetwork(dataSet, lossFunction);

        //test all the XOR instances

        for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
            for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                Instance instance = dataSet.getInstance(i);

                 std::vector<double> weights = std::vector<double>(tinyNN.getNumberWeights());

                for (int j = 0; j < weights.size(); j++) {
                    //give the test weights some random positive and negative values
                    weights[j] = (random_double() * 2.0) - 1.0;
                }

                tinyNN.setWeights(weights);

                 std::vector<double> numericGradient = tinyNN.getNumericGradient(instance);
                 std::vector<double> backpropGradient = tinyNN.getGradient(instance);
                if (! gradientsCloseEnough(numericGradient, backpropGradient)) {
                    throw std::runtime_error("testTinyGradients failed on repeat " + std::to_string(repeat) + " and instance" + std::to_string(i) + "!");
                }
            }
            Log::trace("testTinyGradients passed repeat " + std::to_string(repeat) + "!");

            if ((repeat % 10) == 0) {
                Log::info("testTinyGradients repeat " + std::to_string(repeat) + " completed.");
            }
        }

    } catch (const std::exception& e) {
        Log::fatal("Failed testTinyGradients");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}

void testSmallGradients(DataSet dataSet, LossFunction lossFunction) {
    try {
        NeuralNetwork smallNN =  createSmallNeuralNetwork(dataSet, lossFunction);

        //test all the XOR instances

        for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
            for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                Instance instance = dataSet.getInstance(i);

                 std::vector<double> weights = std::vector<double>(smallNN.getNumberWeights());

                for (int j = 0; j < weights.size(); j++) {
                    //give the test weights some random positive and negative values
                    weights[j] = (random_double() * 2.0) - 1.0;
                }

                smallNN.setWeights(weights);

                 std::vector<double> numericGradient = smallNN.getNumericGradient(instance);
                 std::vector<double> backpropGradient = smallNN.getGradient(instance);
                if (! gradientsCloseEnough(numericGradient, backpropGradient)) {
                    throw   std::runtime_error("testSmallGradients failed on repeat " + std::to_string(repeat) + " and instance" + std::to_string(i) + "!");
                }
            }
            Log::trace("testSmallGradients passed repeat " + std::to_string(repeat) + "!");

            if ((repeat % 10) == 0) {
                Log::info("testSmallGradients repeat " + std::to_string(repeat) + " completed.");
            }
        }

    } catch (const std::exception& e) {
        Log::fatal("Failed testSmallGradients");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}


void testLargeGradients(DataSet dataSet, LossFunction lossFunction) {
    try {
        NeuralNetwork largeNN =  createLargeNeuralNetwork(dataSet, lossFunction);

        //test all the XOR instances

        for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
            for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                Instance instance = dataSet.getInstance(i);

                 std::vector<double> weights = std::vector<double>(largeNN.getNumberWeights());

                for (int j = 0; j < weights.size(); j++) {
                    //give the test weights some random positive and negative values
                    weights[j] = (random_double() * 2.0) - 1.0;
                }

                largeNN.setWeights(weights);

                 std::vector<double> numericGradient = largeNN.getNumericGradient(instance);
                 std::vector<double> backpropGradient = largeNN.getGradient(instance);

                if (! gradientsCloseEnough(numericGradient, backpropGradient)) {
                    throw   std::runtime_error("testLargeGradients failed on repeat " + std::to_string(repeat) + " and instance" + std::to_string(i) + "!");
                }
            }
            Log::trace("testLargeGradients passed repeat " + std::to_string(repeat) + "!");

            if ((repeat % 10) == 0) {
                Log::info("testLargeGradients repeat " + std::to_string(repeat) + " completed.");
            }
        }

    } catch (const std::exception& e) {
        Log::fatal("Failed testLargeGradients");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }

}

void testNetworkOnInstances(NeuralNetwork nn, std::vector<Instance> instances, std::string description) {
     std::vector<double> weights = std::vector<double>(nn.getNumberWeights());

    for (int j = 0; j < weights.size(); j++) {
        //give the test weights some random positive and negative values
        weights[j] = (random_double() * 2.0) - 1.0;
    }

    nn.setWeights(weights);

     std::vector<double> numericGradient = nn.getNumericGradient(instances);
     std::vector<double> backpropGradient = nn.getGradient(instances);

    if (! gradientsCloseEnough(numericGradient, backpropGradient)) {
        throw   std::runtime_error(description + " failed!");
    }

    Log::trace(description + " passed!");
}

void testTinyGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction) {
    try {
        NeuralNetwork tinyNN =  createTinyNeuralNetwork(dataSet, lossFunction);

        //test all the XOR instances

        std::vector<Instance> instances;
        for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
            for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                instances = dataSet.getInstances(0, 2);
                testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 2");

                instances = dataSet.getInstances(1, 2);
                testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 1, 2");

                instances = dataSet.getInstances(2, 2);
                testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 2, 2");

                instances = dataSet.getInstances(0, 3);
                testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 3");

                instances = dataSet.getInstances(1, 3);
                testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 1, 3");

                instances = dataSet.getInstances(0, 4);
                testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 4");

            }

            if ((repeat % 10) == 0) {
                Log::info("testTinyGradientsMultiInstance repeat " + std::to_string(repeat) + " completed.");
            }
        }

    } catch (const std::exception& e) {
        Log::fatal("Failed testTinyGradientsMultiInstance");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}


void testSmallGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction) {
    try {
        NeuralNetwork smallNN =  createSmallNeuralNetwork(dataSet, lossFunction);

        //test all the XOR instances

        std::vector<Instance> instances;
        for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
            for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                instances = dataSet.getInstances(0, 2);
                testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 2");

                instances = dataSet.getInstances(1, 2);
                testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 1, 2");

                instances = dataSet.getInstances(2, 2);
                testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 2, 2");

                instances = dataSet.getInstances(0, 3);
                testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 3");

                instances = dataSet.getInstances(1, 3);
                testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 1, 3");

                instances = dataSet.getInstances(0, 4);
                testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 4");

            }

            if ((repeat % 10) == 0) {
                Log::info("testSmallGradientsMultiInstance repeat " + std::to_string(repeat) + " completed.");
            }
        }

    } catch (const std::exception& e) {
        Log::fatal("Failed testSmallGradientsMultiInstance");
        Log::fatal("Threw exception: " + (std::string) e.what());
    }
}


void testLargeGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction) {
    try {
        NeuralNetwork largeNN =  createLargeNeuralNetwork(dataSet, lossFunction);

        //test all the XOR instances
        std::vector<Instance> instances;
        for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
            for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                instances = dataSet.getInstances(0, 2);
                testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 2");

                instances = dataSet.getInstances(1, 2);
                testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 1, 2");

                instances = dataSet.getInstances(2, 2);
                testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 2, 2");

                instances = dataSet.getInstances(0, 3);
                testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 3");

                instances = dataSet.getInstances(1, 3);
                testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 1, 3");

                instances = dataSet.getInstances(0, 4);
                testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + std::to_string(repeat) + ", instances 0, 4");

            }

            if ((repeat % 10) == 0) {
                Log::info("testLargeGradientsMultiInstance repeat " + std::to_string(repeat) + " completed.");
            }
        }

    } catch (const std::exception& e) {
        Log::fatal("Failed testLargeGradientsMultiInstance");
        Log::fatal("Threw exception: " + (std::string) e.what());

    }

}

double random_double() {
    return rand() / (RAND_MAX + 1.);
}
