#include "NeuralNetwork.h"
#include "Node.h"  
#include "../util/Log.h"
#include "../data/Instance.h"
#include "LossFunction.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <exception>

NeuralNetwork::NeuralNetwork(int inputLayerSize, const std::vector<int>& hiddenLayerSizes, int outputLayerSize, LossFunction lossFunc)
    : lossFunction(lossFunc), numberWeights(0) {
    // The number of layers in the neural network is 2 plus the number of hidden layers
    int totalLayers = hiddenLayerSizes.size() + 2;
    //layers.reserve(totalLayers);

    Log::info("Creating a neural network with " + std::to_string(hiddenLayerSizes.size()) + " hidden layers.");

    for (int layer = 0; layer < totalLayers; ++layer) {
        int layerSize;
        NodeType nodeType;
        ActivationType activationType;

        if (layer == 0) {
            layerSize = inputLayerSize;
            nodeType = NodeType::INPUT;
            activationType = ActivationType::LINEAR;
            Log::info("Input layer " + std::to_string(layer) + " has " + std::to_string(layerSize) + " nodes.");
        }
        else if (layer < totalLayers - 1) {
            layerSize = hiddenLayerSizes[layer - 1];
            nodeType = NodeType::HIDDEN;
            activationType = ActivationType::TANH;
            Log::info("Hidden layer " + std::to_string(layer) + " has " + std::to_string(layerSize) + " nodes.");
            numberWeights += layerSize; // Each hidden node has a bias weight
        }
        else {
            layerSize = outputLayerSize;
            nodeType = NodeType::OUTPUT;
            activationType = ActivationType::SIGMOID;
            Log::info("Output layer " + std::to_string(layer) + " has " + std::to_string(layerSize) + " nodes.");
        }

        // Create and add nodes to the current layer
        std::vector<Node> currentLayer;
        //currentLayer.reserve(layerSize);
        for (int j = 0; j < layerSize; ++j) {
            currentLayer.push_back(Node(layer, j, nodeType, activationType));
        }

        layers.push_back(std::move(currentLayer));
    }
}

int NeuralNetwork::getNumberWeights() const {
    return numberWeights;
}

void NeuralNetwork::reset() {
    for (auto& layer : layers) {
        for (auto& node : layer) {
            node.reset();
        }
    }
}

std::vector<double> NeuralNetwork::getWeights() const {
    std::vector<double> weights(numberWeights);
    int position = 0;
    //for (std::vector<Node> layer : layers) {
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].size(); ++j) {
            const Node& n = layers[i][j];
            int nWeights = n.getWeights(position, weights);
            position += nWeights;
            if (position > numberWeights) {
                throw std::runtime_error("The numberWeights field of the NeuralNetwork was less than the actual number of weights and biases.");
            }
        }
    }
    return weights;
}

void NeuralNetwork::setWeights(std::vector<double>& newWeights) {
    if (numberWeights != newWeights.size()) {
        throw std::runtime_error("Could not setWeights because the number of new weights: " + std::to_string(newWeights.size()) + " was not equal to the number of weights in the NeuralNetwork: " + std::to_string(numberWeights));
    }
    int position = 0;
    for (auto& layer : layers) {
        for (auto& node : layer) {
            int nWeights = node.setWeights(position, newWeights);
            position += nWeights;
            if (position > numberWeights) {
                throw std::runtime_error("The numberWeights field of the NeuralNetwork was (" + std::to_string(numberWeights) + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
            }
        }
    }
}

std::vector<double> NeuralNetwork::getDeltas() const {
    std::vector<double> deltas(numberWeights, 0.0);  // Initialize all deltas to zero.

    int position = 0;
    for (const auto& layer : layers) {
        for (Node node : layer) {
            int nDeltas = node.getDeltas(position, deltas);
            position += nDeltas;

            if (position > numberWeights) {
                throw std::runtime_error("The numberWeights field of the NeuralNetwork was (" + std::to_string(numberWeights) + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
            }
        }
    }

    return deltas;
}

void NeuralNetwork::connectFully() { // TODO: TEST!!!
    for (size_t layer = 0; layer < layers.size() - 1; ++layer) {
        for (Node& inputNode : layers[layer]) {
            for (Node& outputNode : layers[layer + 1]) {
                Edge newEdge = Edge(&inputNode, &outputNode);
                inputNode.addOutgoingEdge(newEdge);
                outputNode.addIncomingEdge(newEdge);
                numberWeights++;
                Log::trace("Number of weights now: " + std::to_string(numberWeights));
            }
        }
    }
}

void NeuralNetwork::connectNodes(int inputLayer, int inputNumber, int outputLayer, int outputNumber) {
    if (inputLayer >= outputLayer) {
        throw std::runtime_error("Cannot create an Edge between input layer " +
            std::to_string(inputLayer) + " and output layer " +
            std::to_string(outputLayer) +
            " because the layer of the input node must be less than the layer of the output node.");
    }

    Node* inputNode = &layers[inputLayer][inputNumber];
    Node* outputNode = &layers[outputLayer][outputNumber];
    Edge newEdge = Edge(inputNode, outputNode);
    inputNode->addOutgoingEdge(newEdge);
    outputNode->addIncomingEdge(newEdge);
    ++numberWeights;
    Log::trace("Number of weights now: " + std::to_string(numberWeights));
}

void NeuralNetwork::initializeRandomly(double bias) {
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (auto& layer : layers) {
        for (Node node : layer) {
            double fanIn = node.getInputEdges().size();
            double variance = fanIn > 0 ? 1.0 / std::sqrt(fanIn) : 1.0;

            for (Edge edge : node.getInputEdges()) {
                edge.weight = distribution(generator) * variance;
                
            }

            node.setBias(bias);
        }
    }
}

double NeuralNetwork::forwardPass(const Instance& instance) {
    reset();  // Reset the network before the forward pass

    // 1. Set input values to the neural network
    if (layers[0].size() != instance.inputs.size()) {
        throw std::runtime_error("Mismatch between network input layer size and instance input size.");
    }
    for (size_t i = 0; i < layers[0].size(); ++i) {
        // Directly assign the input values to the preActivationValue of input nodes
        layers[0][i].preActivationValue = instance.inputs[i];
    }

    // 2. Call forward propagation on each node
    for (auto& layer : layers) {
        for (auto& node : layer) {
            node.propagateForward();
        }
    }


    // The output layer calculations
    int outputLayerIndex = layers.size() - 1;
    const std::vector<Node> outputLayer = layers[outputLayerIndex];
    const std::vector<double>& expectedOutputs = instance.expectedOutputs;

    double outputSum = 0;
    printf("OutputSum: ");
    if (lossFunction == LossFunction::NONE) {
        // Just sum up the outputs
        for (const auto& node : outputLayer) {
            outputSum += node.postActivationValue;
            printf("%lf ", outputSum);
        }
        printf("\n");
    }
    else if (lossFunction == LossFunction::L1_NORM) {
        // Iterate over the output nodes to calculate the L1 loss and the deltas
        for (size_t number = 0; number < outputLayer.size(); ++number) {
            Node outputNode = outputLayer[number];
            double error = expectedOutputs[number] - outputNode.postActivationValue;
            outputSum += std::abs(error);
            // Set the delta for the output node
            // The sign of the error determines the direction of the delta
            outputNode.delta = (error > 0) ? -1 : 1;
        }
    }
    else if (lossFunction == LossFunction::L2_NORM) {
        // Calculate the L2 norm loss and set deltas
        double lossSum = 0.0;
        for (size_t number = 0; number < layers.back().size(); ++number) {
            Node& outputNode = layers.back()[number];
            double error = expectedOutputs[number] - outputNode.postActivationValue;
            lossSum += error * error;
        }
        double loss = std::sqrt(lossSum);

        // Set the delta for each output node based on the derivative of L2 norm
        for (size_t number = 0; number < layers.back().size(); ++number) {
            Node& outputNode = layers.back()[number];
            if (loss != 0) {  // To avoid division by zero
                outputNode.delta = -1 * (expectedOutputs[number] - outputNode.postActivationValue) / loss;
            }
            else {
                outputNode.delta = 0;
            }
        }
        outputSum = loss;
    }
    else if (lossFunction == LossFunction::SVM) {
        // Implement SVM loss
        int expectedIndex = static_cast<int>(expectedOutputs[0]);
        double expectedOutput = layers.back()[expectedIndex].postActivationValue;
        double deltaSum = 0.0;
        double hingeLossSum = 0.0;

        for (size_t number = 0; number < layers.back().size(); ++number) {
            Node& outputNode = layers.back()[number];
            if (number != expectedIndex) {
                double hingeLoss = std::max(0.0, outputNode.postActivationValue - expectedOutput + 1);
                hingeLossSum += hingeLoss;

                if (hingeLoss > 0) {
                    outputNode.delta = 1;
                    deltaSum += 1;
                }
                else {
                    outputNode.delta = 0;
                }
            }
        }

        // Adjust delta for the expected output node
        Node& expectedNode = layers.back()[expectedIndex];
        expectedNode.delta = -deltaSum;
        outputSum = hingeLossSum;
    }
    else if (lossFunction == LossFunction::SOFTMAX) {
        // Implement Softmax loss
        int expectedIndex = static_cast<int>(instance.expectedOutputs[0]);
        double totalExpSum = 0.0;
        double expectedExp = 0.0;

        // Calculate sum of exponentials for all outputs
        for (Node& outputNode : layers.back()) {
            double expValue = std::exp(outputNode.postActivationValue);
            totalExpSum += expValue;
            if (&outputNode == &layers.back()[expectedIndex]) {
                expectedExp = expValue;
            }
        }

        // Calculate softmax loss and delta for each output node
        for (Node& outputNode : layers.back()) {
            double softmaxProb = std::exp(outputNode.postActivationValue) / totalExpSum;
            outputNode.delta = ((&outputNode) == (&layers.back()[expectedIndex])) ? (softmaxProb - 1) : softmaxProb;
        }

        // Calculate the overall loss (negative log likelihood)
        outputSum = -std::log(expectedExp / totalExpSum);
    }
    else {
        throw std::runtime_error("Unsupported loss function in forward pass.");
    }


    return outputSum;
}

double NeuralNetwork::forwardPass(const std::vector<Instance>& instances) {
    double totalSum = 0.0;

    for (const Instance& instance : instances) {
        totalSum += forwardPass(instance);
    }

    return totalSum;
}

double NeuralNetwork::calculateAccuracy(const std::vector<Instance>& instances) {
    int correctCount = 0;
    int totalCount = instances.size();

    for (const Instance& instance : instances) {
        forwardPass(instance);
        std::vector<double> output = getOutputValues();

        double maxOutput = -std::numeric_limits<double>::infinity();
        int predictedIndex = -1;
        for (size_t i = 0; i < output.size(); ++i) {
            if (output[i] > maxOutput) {
                maxOutput = output[i];
                predictedIndex = static_cast<int>(i);
            }
        }

        if (instance.expectedOutputs.size() > 0 && static_cast<int>(instance.expectedOutputs[0]) == predictedIndex) {
            ++correctCount;
        }
    }

    return static_cast<double>(correctCount) / static_cast<double>(totalCount);
}

std::vector<double> NeuralNetwork::getOutputValues() const {
    if (layers.empty()) {
        throw std::runtime_error("Neural network has no layers.");
    }

    const auto& outputLayer = layers.back();
    std::vector<double> outputValues;
    outputValues.reserve(outputLayer.size());

    for (const Node& node : outputLayer) {
        outputValues.push_back(node.postActivationValue);
    }

    return outputValues;
}

const double H = 0.0000001;

std::vector<double> NeuralNetwork::getNumericGradient(const Instance& instance) {
    std::vector<double> currentWeights = getWeights();
    std::vector<double> testWeights = std::vector<double>(currentWeights.size(), 0.0);
    for (size_t i = 0; i < currentWeights.size(); ++i) {
        testWeights[i] = currentWeights[i];
    }
    std::vector<double> numericGradient(currentWeights.size(), 0.0);

    for (size_t i = 0; i < currentWeights.size(); ++i) {
        // Increment weight by H
        testWeights[i] = currentWeights[i] + H;
        setWeights(testWeights);
        double outputPlusH = forwardPass(instance);
        printf("OutputPlus: %g\n", outputPlusH);

        // Decrement weight by H
        testWeights[i] = currentWeights[i] - H;
        setWeights(testWeights);
        double outputMinusH = forwardPass(instance);
        printf("OutputMinus: %g\n", outputMinusH);
        // Calculate the gradient
        numericGradient[i] = (outputPlusH - outputMinusH) / (2 * H);

        // Reset the weight for the next iteration
        //testWeights[i] = currentWeights[i];
        printf("Numeric Gradient: ");
        for (double g : numericGradient) {
            printf("%lf ", g);
        }
        printf("\n");
    }

    // Reset the weights to their original values
    setWeights(currentWeights);
    
    return numericGradient;
}

std::vector<double> NeuralNetwork::getNumericGradient(const std::vector<Instance>& instances) {
    std::vector<double> currentWeights = getWeights();
    std::vector<double> testWeights = currentWeights;
    std::vector<double> numericGradient(currentWeights.size(), 0.0);

    for (size_t i = 0; i < currentWeights.size(); ++i) {
        double outputPlusH = 0.0;
        double outputMinusH = 0.0;

        // Increment weight by H
        testWeights[i] = currentWeights[i] + H;
        setWeights(testWeights);
        for (const auto& instance : instances) {
            outputPlusH += forwardPass(instance);
        }

        // Decrement weight by H
        testWeights[i] = currentWeights[i] - H;
        setWeights(testWeights);
        for (const auto& instance : instances) {
            outputMinusH += forwardPass(instance);
        }

        // Calculate the gradient
        numericGradient[i] = (outputPlusH - outputMinusH) / (2 * H);

        // Reset the weight for the next iteration
        testWeights[i] = currentWeights[i];
    }

    // Reset the weights to their original values
    setWeights(currentWeights);

    return numericGradient;
}

void NeuralNetwork::backwardPass() {
    // Propagate backward starting from the output layer to the input layer
    for (int i = layers.size() - 1; i >= 0; --i) {
        for (Node& node : layers[i]) {
            node.propagateBackward();
        }
    }
}

// Gets the gradient of the neural network at its current weights for a given instance.
std::vector<double> NeuralNetwork::getGradient(const Instance& instance) {
    forwardPass(instance);
    backwardPass();
    return getDeltas();
}

// Gets the gradient of the neural network for a list of instances.
std::vector<double> NeuralNetwork::getGradient(const std::vector<Instance>& instances) {
    std::vector<double> gradientSum(numberWeights, 0.0);

    // Accumulate the gradients for each instance
    for (const auto& instance : instances) {
        std::vector<double> singleGradient = getGradient(instance);
        for (size_t i = 0; i < numberWeights; ++i) {
            gradientSum[i] += singleGradient[i];
        }
    }

    return gradientSum;
}