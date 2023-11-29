#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "./util/Log.h"
#include "./data/DataSet.h"
#include "./network/LossFunction.h"
#include "./network/NeuralNetwork.h"
#include "./data/Instance.h"
#include "./util/Vector.h"

// Function to display usage information
void helpMessage() {
    Log::info("Usage:");
    Log::info("\t./program <data set> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <adaptive learning rate> <decayRate> <eps> <beta1> <beta2> <layer_size_1 ... layer_size_n");
    Log::info("\t\tdata set can be: 'and', 'or' or 'xor', 'iris' or 'mushroom'");
    Log::info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
    Log::info("\t\tbatch size should be > 0. Will be ignored for stochastic or batch gradient descent");
    Log::info("\t\tloss function can be: 'svm' or 'softmax'");
    Log::info("\t\tepochs is an integer > 0");
    Log::info("\t\tbias is a double");
    Log::info("\t\tlearning rate is a double usually small and > 0");
    Log::info("\t\tmu is a double < 1 and typical values are 0.5, 0.9, 0.95, and 0.99");
    Log::info("\t\tadaptive learning rate can be: 'nesterov', 'rmsprop' or 'adam'");
    Log::info("\t\tdecayRate is a double");
    Log::info("\t\teps is a double");
    Log::info("\t\tbeta1 is a double");
    Log::info("\t\tbeta2 is a double");
    Log::info("\t\tlayer_size_1..n is a list of integers which are the number of nodes in each hidden layer");
}

DataSet getDataset(std::string dataSetName) {
    if (dataSetName == "and") {
        DataSet dataSet = DataSet("and data", "./datasets/and.txt");
        return dataSet;
    }
    else if (dataSetName == "or") {
        DataSet dataSet = DataSet("or data", "./datasets/or.txt");
        return dataSet;
    }
    else if (dataSetName == "xor") {
        DataSet dataSet = DataSet("xor data", "./datasets/xor.txt");
        return dataSet;
    }
    else if (dataSetName == "iris") {
        DataSet dataSet = DataSet("iris data", "./datasets/iris.txt");
        std::vector<double> means = dataSet.getInputMeans();
        std::vector<double> stdDevs = dataSet.getInputStandardDeviations();

        Log::info("data set means: ");
        for (double x : means) {
            printf("%g ", x);
        }
        printf("\n");

        Log::info("data set standard deviations: ");
        for (double x : stdDevs) {
            printf("%g ", x);
        }
        printf("\n");

        dataSet.normalize(means, stdDevs);
        return dataSet;
    }
    else if (dataSetName == "mushroom") {
        DataSet dataSet = DataSet("mushroom data", "./datasets/agaricus-lepiota.txt");
        return dataSet;
    }
    else {
        Log::fatal("unknown data set : " + dataSetName);
        exit(1);
    }
}

int getOutputLayerSize(std::string dataSetName, DataSet dataSet) {
    if (dataSetName == "and") {
        return dataSet.getNumberOutputs();
    }
    else if (dataSetName == "or") {
        return dataSet.getNumberOutputs();
    }
    else if (dataSetName == "xor") {
        return dataSet.getNumberOutputs();
    }
    else if (dataSetName == "iris") {
        return dataSet.getNumberClasses();
    }

    else if (dataSetName == "mushroom") {
        return dataSet.getNumberClasses();
    }
    else {
        Log::fatal("unknown data set : " + dataSetName);
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 15) {
        helpMessage();
        return 1;
    }

    std::string dataSetName = argv[1];
    std::string descentType = argv[2];
    int batchSize = std::stoi(argv[3]);
    std::string lossFunctionName = argv[4];
    int epochs = std::stoi(argv[5]);
    double bias = std::stod(argv[6]);
    double learningRate = std::stod(argv[7]);
    double mu = std::stod(argv[8]);
    std::string adaptive_l_r = argv[9];
    double decayRate = std::stod(argv[10]);
    double eps = std::stod(argv[11]);
    double beta1 = std::stod(argv[12]);
    double beta2 = std::stod(argv[13]);

    std::vector<int> layerSizes(argc - 14);
    for (int i = 14; i < argc; i++) {
        layerSizes[i - 14] = std::stoi(argv[i]);
    }

    DataSet dataSet = getDataset(dataSetName);
    int outputLayerSize = getOutputLayerSize(dataSetName, dataSet);

    LossFunction lossFunction = LossFunction::NONE;
    if (lossFunctionName == "svm") {
        Log::info("Using an SVM loss function.");
        lossFunction = LossFunction::SVM;
    }
    else if (lossFunctionName == "softmax") {
        Log::info("Using an SOFTMAX loss function.");
        lossFunction = LossFunction::SOFTMAX;
    }
    else {
        Log::fatal("unknown loss function : " + lossFunctionName);
        exit(1);
    }


    NeuralNetwork nn(dataSet.getNumberInputs(), layerSizes, outputLayerSize, lossFunction);

    try {
        nn.connectFully();
    }
    catch (const std::runtime_error& e) {
        Log::fatal("ERROR connecting the neural network -- this should not happen!.");
        exit(1);
    }

    // Start the gradient descent
    try {
        Log::info("Starting " + descentType + " gradient descent!");

        if (descentType == "minibatch") {
            Log::info(descentType + "(" + std::to_string(batchSize) + "), " + dataSetName + ", " + lossFunctionName + ", lr: " + std::to_string(learningRate) + ", mu:" + std::to_string(mu));
        }
        else {
            Log::info(descentType + ", " + dataSetName + ", " + lossFunctionName + ", lr: " + std::to_string(learningRate) + ", mu:" + std::to_string(mu));
        }

        nn.initializeRandomly(bias);

        std::vector<double> velocity(nn.getNumberWeights());
        std::vector<double> velocityPrev(nn.getNumberWeights());
        std::vector<double> cache(nn.getNumberWeights());
        std::vector<double> m(nn.getNumberWeights());

        // implement the RMSprop
        // per-parameter adaptive learning rate method.
        // and implement the Adam
        // per-parameter adaptive learning rate method.
        // For these, you will need to add a command line flag
        // to select which method you'll use (nesterov, rmsprop, or adam)

        double error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
        double bestError = error;
        double accuracy = nn.calculateAccuracy(dataSet.getInstances());

        Log::info("  " + std::to_string(bestError) + " " + std::to_string(error) + " " + std::to_string(accuracy * 100.0));

        for (int i = 0; i < epochs; i++) {
            if (descentType == "stochastic") {
                // implement one epoch (pass through the
                // training data) for stochastic gradient descent
                dataSet.shuffle();
                for (int ins = 0; ins < dataSet.getNumberInstances(); ins++) {
                    Instance instance = dataSet.getInstance(ins);
                    std::vector<double> gradient = nn.getGradient(instance);
                    std::vector<double> newWeights = nn.getWeights();
                    for (int j = 0; j < newWeights.size(); j++) {
                        if (adaptive_l_r == "nesterov") {
                            velocityPrev[j] = velocity[j];
                            velocity[j] = mu * velocity[j] - learningRate * gradient[j];
                            newWeights[j] += (-1 * mu * velocityPrev[j]) + ((1 + mu) * velocity[j]);
                        }
                        else if (adaptive_l_r == "rmsprop") {
                            cache[j] = decayRate * cache[j] + (1 - decayRate) * std::pow(gradient[j], 2);
                            newWeights[j] -= (learningRate / (std::sqrt(cache[j]) + eps)) * gradient[j];
                        }
                        else if (adaptive_l_r == "adam") {
                            m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                            velocity[j] = beta2 * velocity[j] + (1 - beta2) * std::pow(gradient[j], 2);
                            newWeights[j] -= learningRate * m[j] / std::sqrt(velocity[j] + eps);
                        }
                        else {
                            Log::fatal("unknown adaptive learning rate type: " + adaptive_l_r);
                            helpMessage();
                            exit(1);
                        }
                    }
                    nn.setWeights(newWeights);
                }
            }
            else if (descentType == "minibatch") {
                // implement one epoch (pass through the
                // training data) for minibatch gradient descent
                dataSet.shuffle();
                for (int ins = 0; ins < dataSet.getNumberInstances(); ins += batchSize) {
                    std::vector<Instance> instances = dataSet.getInstances(ins, batchSize);
                    std::vector<double> gradient = nn.getGradient(instances);
                    std::vector<double> newWeights = nn.getWeights();
                    for (int j = 0; j < newWeights.size(); j++) {
                        if (adaptive_l_r == "nesterov") {
                            velocityPrev[j] = velocity[j];
                            velocity[j] = mu * velocity[j] - learningRate * gradient[j];
                            newWeights[j] += (-1 * mu * velocityPrev[j]) + ((1 + mu) * velocity[j]);
                        }
                        else if (adaptive_l_r == "rmsprop") {
                            cache[j] = decayRate * cache[j] + (1 - decayRate) * std::pow(gradient[j], 2);
                            newWeights[j] -= (learningRate / (std::sqrt(cache[j]) + eps)) * gradient[j];
                        }
                        else if (adaptive_l_r == "adam") {
                            m[j] = beta1 * m[j] + (1 - beta1) * gradient[j];
                            velocity[j] = beta2 * velocity[j] + (1 - beta2) * std::pow(gradient[j], 2);
                            newWeights[j] -= learningRate * m[j] / std::sqrt(velocity[j] + eps);
                        }
                        else {
                            Log::fatal("unknown adaptive learning rate type: " + adaptive_l_r);
                            helpMessage();
                            exit(1);
                        }
                    }
                    nn.setWeights(newWeights);
                }
            }
            else if (descentType == "batch") {
                // implement one epoch (pass through the training
                // instances) for batch gradient descent
                std::vector<Instance> instances = dataSet.getInstances();
                std::vector<double> gradient = nn.getGradient(instances);
                std::vector<double> newWeights = nn.getWeights();
                for (int ins = 0; ins < newWeights.size(); ins++) {
                    if (adaptive_l_r == "nesterov") {
                        velocityPrev[ins] = velocity[ins];
                        velocity[ins] = mu * velocity[ins] - learningRate * gradient[ins];
                        newWeights[ins] += (-1 * mu * velocityPrev[ins]) + ((1 + mu) * velocity[ins]);
                    }
                    else if (adaptive_l_r == "rmsprop") {
                        cache[ins] = decayRate * cache[ins] + (1 - decayRate) * std::pow(gradient[ins], 2);
                        newWeights[ins] -= (learningRate / (std::sqrt(cache[ins]) + eps)) * gradient[ins];
                    }
                    else if (adaptive_l_r == "adam") {
                        m[ins] = beta1 * m[ins] + (1 - beta1) * gradient[ins];
                        velocity[ins] = beta2 * velocity[ins] + (1 - beta2) * std::pow(gradient[ins], 2);
                        newWeights[ins] -= learningRate * m[ins] / std::sqrt(velocity[ins] + eps);
                    }
                    else {
                        Log::fatal("unknown adaptive learning rate type: " + adaptive_l_r);
                        helpMessage();
                        exit(1);
                    }
                }
                nn.setWeights(newWeights);
            }
            else {
                Log::fatal("unknown descent type: " + descentType);
                helpMessage();
                exit(1);
            }

            // At the end of each epoch, calculate the error over the entire
            // set of instances and print it out so we can see if we're decreasing
            // the overall error
            double err = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
            double acc = nn.calculateAccuracy(dataSet.getInstances());
            if (err < bestError) bestError = err;
            Log::info("  " + std::to_string(bestError) + " " + std::to_string(err) + " " + std::to_string(acc * 100.0));
        }

    }
    catch (const std::runtime_error& e) {
        Log::fatal("gradient descent failed with exception: " + (std::string) e.what());
        exit(1);
    }

    return 0;
}