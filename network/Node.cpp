#include "Node.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>

// Node constructor implementation
Node::Node(int layerValue, int numberValue, NodeType type, ActivationType actType)
    : layer(layerValue), number(numberValue), nodeType(type), activationType(actType),
    preActivationValue(0), postActivationValue(0), delta(0), activationDerivative(0),
    bias(0), biasDelta(0) {
    // Initialization logic here if needed
}

// Destructor to clean up the edges
Node::~Node() {
    for (auto* edge : inputEdges) {
        delete edge;
    }
    inputEdges.clear();

    for (auto* edge : outputEdges) {
        delete edge;
    }
    outputEdges.clear();
}

void Node::reset() {
    preActivationValue = 0;
    postActivationValue = 0;
    delta = 0;
    biasDelta = 0;
    for (Edge* edge : inputEdges) {
        edge->weightDelta = 0;
    }
}

void Node::addOutgoingEdge(Edge* outgoingEdge) {
    if (outgoingEdge != nullptr) {
        outputEdges.push_back(outgoingEdge);
    }
    else {
        throw std::invalid_argument("Cannot add a null outgoing edge.");
    }
}

void Node::addIncomingEdge(Edge* incomingEdge) {
    if (incomingEdge != nullptr) {
        inputEdges.push_back(incomingEdge);
    }
    else {
        throw std::invalid_argument("Cannot add a null incoming edge.");
    }
}

void Node::propagateForward() {
    preActivationValue += bias;
    for (Edge* edge : inputEdges) {
        preActivationValue += edge->weight * edge->startNode->postActivationValue;
    }

    switch (activationType) {
    case ActivationType::LINEAR:
        applyLinear();
        break;
    case ActivationType::SIGMOID:
        applySigmoid();
        break;
    case ActivationType::TANH:
        applyTanh();
        break;
    default:
        throw std::runtime_error("Unsupported activation type.");
    }
}

void Node::applyLinear() {
    postActivationValue = preActivationValue;
    activationDerivative = 1;
}

void Node::applySigmoid() {
    postActivationValue = 1.0 / (1.0 + exp(-preActivationValue));
    activationDerivative = postActivationValue * (1 - postActivationValue);
}

void Node::applyTanh() {
    postActivationValue = tanh(preActivationValue);
    activationDerivative = 1 - postActivationValue * postActivationValue;
}

void Node::propagateBackward() {
    double sum = 0;
    for (Edge* edge : outputEdges) {
        sum += edge->weight * edge->endNode->delta;
    }

    delta = sum * activationDerivative;
}

int Node::getWeights(int position, std::vector<double>& weights) {
    int weightCount = 0;

    // The first weight set will be the bias if it is a hidden node
    if (nodeType == NodeType::HIDDEN) {
        weights[position] = bias;
        weightCount = 1;
    }

    for (auto* edge : outputEdges) {
        weights[position + weightCount] = edge->getWeight();
        weightCount++;
    }

    return weightCount;
}

int Node::getDeltas(int position, std::vector<double>& deltas) {
    int deltaCount = 0;

    // The first delta set will be the bias if it is a hidden node
    if (nodeType == NodeType::HIDDEN) {
        deltas[position] = biasDelta;
        deltaCount = 1;
    }

    for (auto* edge : outputEdges) {
        deltas[position + deltaCount] = edge->getWeightDelta();
        deltaCount++;
    }

    return deltaCount;
}

int Node::setWeights(int position, const std::vector<double>& weights) {
    int weightCount = 0;

    // The first weight set will be the bias if it is a hidden node
    if (nodeType == NodeType::HIDDEN) {
        bias = weights[position];
        weightCount = 1;
    }

    for (auto* edge : outputEdges) {
        edge->setWeight(weights[position + weightCount]);
        weightCount++;
    }

    return weightCount;
}

void Node::applyDerivativeLinear() {
    activationDerivative = 1;
}

void Node::applyDerivativeSigmoid() {
    activationDerivative = postActivationValue * (1 - postActivationValue);
}

void Node::applyDerivativeTanh() {
    activationDerivative = 1 - std::pow(postActivationValue, 2);
}

void Node::propagateBackward() {
    double deltaPushBack = delta * activationDerivative;

    // Set the biasDelta to delta. Because it's an addition
    biasDelta += deltaPushBack;

    // Call propagateBackward for all incomingEdges
    for (auto* edge : inputEdges) {
        edge->propagateBackward(deltaPushBack);
    }
}

void Node::initializeWeightsAndBias(double newBias) {
    bias = newBias;
    size_t N = inputEdges.size();
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (auto* edge : inputEdges) {
        edge->setWeight(distribution(generator) / std::sqrt(N));
    }
}

std::string Node::toString() const {
    std::stringstream ss;
    ss << "[Node - layer: " << layer << ", number: " << number << ", type: "
        << static_cast<int>(nodeType) << "]";
    return ss.str();
}

std::string Node::toDetailedString() const {
    std::stringstream ss;
    ss << "[Node - layer: " << layer << ", number: " << number << ", node type: "
        << static_cast<int>(nodeType) << ", activation type: "
        << static_cast<int>(activationType) << ", n input edges: "
        << inputEdges.size() << ", n output edges: " << outputEdges.size()
        << ", pre value: " << preActivationValue << ", post value: "
        << postActivationValue << ", delta: " << delta << "]";
    return ss.str();
}