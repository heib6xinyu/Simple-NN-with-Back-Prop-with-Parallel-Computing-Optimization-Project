#include "NeuralNetwork.h"
#include "Node.h"  
#include "util/Log.h"
#include "data/Instance.h"
#include "LossFunction.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <random>

NeuralNetwork::NeuralNetwork(int inputLayerSize, const std::vector<int>& hiddenLayerSizes, int outputLayerSize, LossFunction lossFunc)
    : lossFunction(lossFunc), numberWeights(0) {
    // The number of layers in the neural network is 2 plus the number of hidden layers
    int totalLayers = hiddenLayerSizes.size() + 2;
    layers.reserve(numLayers);  

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
        currentLayer.reserve(layerSize);
        for (int j = 0; j < layerSize; ++j) {
            currentLayer.emplace_back(layer, j, nodeType, activationType);
        }

        layers.push_back(std::move(currentLayer));
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
    std::vector<double> weights;
    weights.reserve(numberWeights);

    int position = 0;
    for (const auto& layer : layers) {
        for (const auto& node : layer) {
            int nWeights = node.getWeights(position, weights);
            position += nWeights;

            if (position > numberWeights) {
                throw NeuralNetworkException("The numberWeights field of the NeuralNetwork was less than the actual number of weights and biases.");
            }
        }
    }

    return weights;
}

void NeuralNetwork::setWeights(const std::vector<double>& newWeights) {
    if (numberWeights != newWeights.size()) {
        throw NeuralNetworkException("Could not setWeights because the number of new weights: " + std::to_string(newWeights.size()) + " was not equal to the number of weights in the NeuralNetwork: " + std::to_string(numberWeights));
    }

    int position = 0;
    for (auto& layer : layers) {
        for (auto& node : layer) {
            int nWeights = node.setWeights(position, newWeights);
            position += nWeights;

            if (position > numberWeights) {
                throw NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + std::to_string(numberWeights) + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
            }
        }
    }
}

std::vector<double> NeuralNetwork::getDeltas() const {
    std::vector<double> deltas(numberWeights, 0.0);  // Initialize all deltas to zero.

    int position = 0;
    for (const auto& layer : layers) {
        for (const auto& node : layer) {
            int nDeltas = node.getDeltas(position, deltas);
            position += nDeltas;

            if (position > numberWeights) {
                throw NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + std::to_string(numberWeights) + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
            }
        }
    }

    return deltas;
}

void NeuralNetwork::connectFully() {
    for (size_t layer = 0; layer < layers.size() - 1; ++layer) {
        for (Node& inputNode : layers[layer]) {
            for (Node& outputNode : layers[layer + 1]) {
                Edge* newEdge = new Edge(&inputNode, &outputNode);
                inputNode.addOutgoingEdge(newEdge);
                outputNode.addIncomingEdge(newEdge);
                ++numberWeights;
            }
        }
    }
    Log::trace("Number of weights now: " + std::to_string(numberWeights));
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
    Edge* newEdge = new Edge(inputNode, outputNode);
    inputNode->addOutgoingEdge(newEdge);
    outputNode->addIncomingEdge(newEdge);
    ++numberWeights;
    Log::trace("Number of weights now: " + std::to_string(numberWeights));
}

void NeuralNetwork::initializeRandomly(double bias) {
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (auto& layer : layers) {
        for (Node& node : layer) {
            double fanIn = node.getInputEdges().size();
            double variance = fanIn > 0 ? 1.0 / std::sqrt(fanIn) : 1.0;

            for (Edge* edge : node.getInputEdges()) {
                edge->setWeight(distribution(generator) * variance);
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
    const std::vector<Node>& outputLayer = layers[outputLayerIndex];
    const std::vector<double>& expectedOutputs = instance.expectedOutputs;

    double outputSum = 0;
    if (lossFunction == LossFunction::NONE) {
        // Just sum up the outputs
        for (const auto& node : outputLayer) {
            outputSum += node.postActivationValue;
        }
    }
    else if (lossFunction == LossFunction::L1_NORM) {
        // Iterate over the output nodes to calculate the L1 loss and the deltas
        for (size_t number = 0; number < outputLayer.size(); ++number) {
            Node& outputNode = outputLayer[number];
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

    }

    return outputSum;
}