#include <string>
#include "./data/DataSet.h"
#include "./util/Log.h"
#include "./network/NeuralNetwork.h"
#include <stdexcept>
#include "./util/PA11TestsUtils.h"


void helpMessage() {
    Log::info("Usage:");
    Log::info("\tjava PA13GradientDescent <data set> <network type> <gradient descent type> <loss function> <epochs> <bias> <learning rate>");
    Log::info("\t\tdata set can be: 'and', 'or' or 'xor'");
    Log::info("\t\tnetwork type can be: 'tiny', 'small' or 'large'");
    Log::info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
    Log::info("\t\tloss function can be: 'l1_norm', or 'l2 norm'");
    Log::info("\t\tepochs is an integer > 0");
    Log::info("\t\tbias is a double");
    Log::info("\t\tlearning rate is a double usually small and > 0");
}

void stochasticEpoch(DataSet dataSet, NeuralNetwork nn, double learningRate) {
    //implement one epoch (pass through the
    //training data) for stochastic gradient descent
    dataSet.shuffle();
    for(int i = 0; i < dataSet.getNumberInstances(); i ++) {
        Instance instance = dataSet.getInstance(i);
        std::vector<double> gradient = nn.getGradient(instance);
        std::vector<double> newWeights = nn.getWeights();
        for (int j = 0; j < newWeights.size(); j ++) {
            newWeights[j] -= learningRate * gradient[j];
        }
        nn.setWeights(newWeights);
    }
}

void minibatchEpoch(DataSet dataSet, NeuralNetwork nn, int batchSize, double learningRate) {
    //implement one epoch (pass through the
    //training data) for minibatch gradient descent
    dataSet.shuffle();
    for (int i = 0; i < dataSet.getNumberInstances(); i += batchSize) {
        std::vector<Instance> instances = dataSet.getInstances(i, batchSize);
        std::vector<double> gradient = nn.getGradient(instances);
        std::vector<double> newWeights = nn.getWeights();
        for (int j = 0; j < newWeights.size(); j ++) {
            newWeights[j] -= learningRate * gradient[j];
        }
        nn.setWeights(newWeights);
    }
}

void batchEpoch(DataSet dataSet, NeuralNetwork nn, double learningRate) {
    //implement one epoch (pass through the training
    //instances) for batch gradient descent
    std::vector<Instance> instances = dataSet.getInstances();
    std::vector<double> gradient = nn.getGradient(instances);
    std::vector<double> newWeights = nn.getWeights();
    for (int i = 0; i < newWeights.size(); i ++) {
        newWeights[i] -= learningRate * gradient[i];
    }
    nn.setWeights(newWeights);
}

/**
 * This performs one of the three types of gradient descent on a given
 * neural network for a given data set. It will initialize the neural
 * network's weights randomly and set the node's bias to a specified
 * bias. It will run for a specified number of epochs.
 *
 * @param descentType is the type of gradient descent, it can be either "stochastic", "minibatch" or "batch".
 * @param dataSet is the dataSet to train on
 * @param nn is the neural network
 * @param epochs is how many epochs to train the neural network for
 * @param bias is the bias to initialize each node's bias with
 * @param learnignRate is the step size/learning rate for the weight updates
 */
void gradientDescent(std::string descentType, DataSet dataSet, NeuralNetwork nn, int epochs, double bias, double learningRate) {
    nn.initializeRandomly(bias);

    Log::info("Initial weights:");
    //Vector.print(nn.getWeights());

    double bestError = 10000;

    for (int i = 0; i < epochs; i++) {

        if (descentType == "stochastic") {
            stochasticEpoch(dataSet, nn, learningRate);

        } else if (descentType == "minibatch") {
            //for now we will just use a batch size of 2 because there are only 
            //4 training samples
            minibatchEpoch(dataSet, nn, 2, learningRate);

        } else if (descentType == "batch") {
            batchEpoch(dataSet, nn, learningRate);

        } else {
            Log::fatal("unknown descent type: " + descentType);
            helpMessage();
            exit(1);
        }

        //at the end of each epoch, calculate the error over the entire
        //set of instances and print it out so we can see if we're decreasing
        //the overall error
        double error = nn.forwardPass(dataSet.getInstances());

        if (error < bestError) bestError = error;

        Log::info(std::to_string(i) + " " + std::to_string(bestError) + " " + std::to_string(error));
    }
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        helpMessage();
        exit(1);
    }
    std::string dataSetName = argv[1];
    printf("%s\n", dataSetName);
    std::string networkType = argv[2];
    std::string descentType = argv[3];
    std::string lossFunctionName = argv[4];
    int epochs = std::stoi(argv[5]);
    double bias = std::stod(argv[6]);
    double learningRate = std::stod(argv[7]);

    std::string name = "";
    std::string filepath = "";
    if (dataSetName == "and") {
        name = "and data";
        filepath = "./datasets/and.txt";
    } else if (dataSetName == "or") {
        name = "or data";
        filepath = "./datasets/or.txt";
    } else if (dataSetName == "xor") {
        name = "xor data";
        filepath = "./datasets/xor.txt";
    } else {
        Log::fatal("unknown data set : " + dataSetName);
        exit(1);
    }
    DataSet dataSet = DataSet(name, filepath);

    LossFunction lossFunction = LossFunction::NONE;
    if (lossFunctionName == "l1_norm") {
        Log::info("Using an L1_NORM loss function.");
        lossFunction = LossFunction::L1_NORM;
    } else if (lossFunctionName == "l2_norm") {
        Log::info("Using an L2_NORM loss function.");
        lossFunction = LossFunction::L2_NORM;
    } else {
        Log::fatal("unknown loss function : " + lossFunctionName);
        exit(1);
    }


    Log::info("Using a tiny neural network.");
    if (networkType == "tiny") {
        Log::info("Using a tiny neural network.");
    } else if (networkType == "small") {
        Log::info("Using a small neural network.");
    } else if (networkType == "large") {
        Log::info("Using a large neural network.");
    } else {
        Log::fatal("unknown network type: " + networkType);
        exit(1);
    }

    NeuralNetwork nn = networkType == "tiny" ? createTinyNeuralNetwork(dataSet, lossFunction) : (networkType == "small" ? createSmallNeuralNetwork(dataSet, lossFunction) : createLargeNeuralNetwork(dataSet, lossFunction));

    //start the gradient descent
    try {
        Log::info("Starting " + descentType + " gradient descent!");

        Log::info(descentType + " " + dataSetName + " " + lossFunctionName + " " + std::to_string(learningRate));

        gradientDescent(descentType, dataSet, nn, epochs, bias, learningRate);
    } catch (const std::runtime_error& e) { 
        Log::fatal("gradient descent failed with exception: " + (std::string) e.what());
        exit(1);
    }
}