#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "Edge.h"
#include "Node.h"
#include "../util/Log.h"
#include <memory>

// Node constructor implementation
Node::Node(int layerValue, int numberValue, NodeType type, ActivationType actType)
    : layer(layerValue), number(numberValue), nodeType(type), activationType(actType),
    preActivationValue(0), postActivationValue(0), delta(0), activationDerivative(0),
    bias(0), biasDelta(0) {
    // Initialization logic here if needed
}

void Node::reset() {
    preActivationValue = 0;
    postActivationValue = 0;
    activationDerivative = 0;
    delta = 0;
    biasDelta = 0;
    for (std::shared_ptr<Edge>& edge : inputEdges) {
        edge->weightDelta = 0;
    }
}

void Node::addOutgoingEdge(std::shared_ptr<Edge> outgoingEdge) {
    outputEdges.push_back(outgoingEdge);
    Log::trace("Node " + toString() + " added outgoing edge to Node " + outgoingEdge->outputNode->toString());
}

void Node::addIncomingEdge(std::shared_ptr<Edge> incomingEdge) {
    inputEdges.push_back(incomingEdge);
    Log::trace("Node " + toString() + " added incoming edge to Node " + incomingEdge->outputNode->toString());
}

void Node::propagateForward() {
    //for (Edge edge : inputEdges) {
    //printf("PreAct0: %g\n", preActivationValue);
    for (size_t i = 0; i < inputEdges.size(); ++i) {
        //printf("Edge: %s %g %g\n", inputEdges[i]->toString().c_str(), inputEdges[i]->weight, inputEdges[i]->inputNode->postActivationValue);
        preActivationValue += inputEdges[i]->weight * inputEdges[i]->inputNode->postActivationValue;
    }
    //printf("PreAct1: %g\n", preActivationValue);
    preActivationValue += bias;
    //printf("PreAct2: %g\n", preActivationValue);

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
    //printf("PostAct: %g\n", postActivationValue);
    activationDerivative = 1;
}

void Node::applySigmoid() {
    postActivationValue = 1.0 / (1.0 + exp(-preActivationValue));
    //printf("PostAct: %g\n", postActivationValue);
    activationDerivative = postActivationValue * (1 - postActivationValue);
}

void Node::applyTanh() {
    postActivationValue = tanh(preActivationValue);
    //printf("PostAct: %g\n", postActivationValue);
    activationDerivative = 1 - (postActivationValue * postActivationValue);
}

int Node::getWeights(int position, std::vector<double>& weights) const {
    int weightCount = 0;

    // The first weight set will be the bias if it is a hidden node
    if (nodeType == NodeType::HIDDEN) {
        weights[position] = bias;
        weightCount = 1;
    }

    for (std::shared_ptr<Edge> edge : outputEdges) {
        weights[position + weightCount] = edge->weight;
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

    for (std::shared_ptr<Edge>& edge : outputEdges) {
        deltas[position + deltaCount] = edge->weightDelta;
        deltaCount++;
    }
    return deltaCount;
}

int Node::setWeights(int position, std::vector<double>& weights) {
    int weightCount = 0;

    // The first weight set will be the bias if it is a hidden node
    if (nodeType == NodeType::HIDDEN) {
        bias = weights[position];
        weightCount = 1;
    }

    for (size_t i = 0; i < outputEdges.size(); ++i) {
        //printf("%s edge weight set to %g\n", outputEdges[i]->toString().c_str(), weights[position + weightCount]);
        outputEdges[i]->weight = weights[position + weightCount];
        //printf("%s %g\n", outputEdges[i]->toString().c_str(), outputEdges[i]->weight);
        weightCount++;
    }
    //for (std::shared_ptr<Edge>& edge : outputEdges) {
    //    printf("Output Edge: %s %g\n", edge->toString().c_str(), edge->weight);
    //}
    //for (std::shared_ptr<Edge>& edge : inputEdges) {
    //    printf("Input Edge: %s %g\n", edge->toString().c_str(), edge->weight);
    //}
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
    //printf("Node: %s \tdeltaPushBack: %g \tdelta: %g \tactDeriv: %g\n", toString().c_str(), deltaPushBack, delta, activationDerivative);
    // Set the biasDelta to delta. Because it's an addition
    biasDelta += deltaPushBack;

    // Call propagateBackward for all incomingEdges
    for (std::shared_ptr<Edge>& edge : inputEdges) {
        edge->propagateBackward(deltaPushBack);
    }
}

void Node::initializeWeightsAndBias(double newBias) {
    bias = newBias;
    size_t N = inputEdges.size();
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (std::shared_ptr<Edge>& edge : inputEdges) {
        edge->weight = distribution(generator) / std::sqrt(N);
    }
}

std::vector<std::shared_ptr<Edge>> Node::getInputEdges() {
    return inputEdges;
}

void Node::setBias(double new_bias){
    bias = new_bias;
}

std::string Node::toString() const {
    std::string ss = "[Node - layer: " + std::to_string(layer) + ", number: " + std::to_string(number) + ", type: "
        + std::to_string(static_cast<int>(nodeType)) + "]";
    return ss;
}

std::string Node::toDetailedString() const {
    //std::stringstream ss;
    std::string ss = "[Node - layer: " + std::to_string(layer) + ", number: " + std::to_string(number) + ", node type: "
        + std::to_string(static_cast<int>(nodeType)) + ", activation type: "
        + std::to_string(static_cast<int>(activationType)) + ", n input edges: "
        + std::to_string(inputEdges.size()) + ", n output edges: " + std::to_string(outputEdges.size())
        + ", pre value: " + std::to_string(preActivationValue) + ", post value: "
        + std::to_string(postActivationValue) + ", delta: " + std::to_string(delta) + "]";
    return ss;
}