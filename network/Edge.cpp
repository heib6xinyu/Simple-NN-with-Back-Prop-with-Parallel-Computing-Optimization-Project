#include "Edge.h"
#include "../util/Log.h" 
#include <iostream>

Edge::Edge(Node* inputNode, Node* outputNode) {
    this->inputNode = inputNode;
    this->outputNode = outputNode;
    Log::debug("Created a new edge with input " + inputNode->toString() + " and output " + outputNode->toString());

    // Initialize the weight and delta to 0
    weight = 0;
    weightDelta = 0;

    inputNode->addOutgoingEdge(this);
    outputNode->addIncomingEdge(this);
}

void Edge::propagateBackward(double delta) {
    // Uncommenting the following may help you debug this method:
    // std::cout << "Edge with output node[layer " << outputNode->layer << ", number " << outputNode->number << "] and input node[layer " << inputNode->layer << ", number " << inputNode->number <<"] weight: " << weight << std::endl;
    // std::cout << "Edge with output node[layer " << outputNode->layer << ", number " << outputNode->number << "] and input node[layer " << inputNode->layer << ", number " << inputNode->number <<"] backpropagating delta: " << delta << std::endl;

    // 1. set the weightDelta to be the updated delta of its outputNode * postActivationValue of inputNode
    weightDelta = delta * inputNode->postActivationValue;

    // 2. the delta of the inputNode to be the product of weight and the updated delta of the its outputNode.
    // += because if multiple edges come into the same inputNode, add them up
    inputNode->delta += weight * delta;

    // std::cout << "Edge with output node[layer " << outputNode->layer << ", number " << outputNode->number << "] and input node[layer " << inputNode->layer << ", number " << inputNode->number <<"] backpropagating weightDelta: " << weightDelta << std::endl;
}

void Edge::setWeight(double new_weight) {
    weight = new_weight;
}

bool Edge::equals(const Edge& other) const {
    return this->inputNode->layer == other.inputNode->layer &&
        this->inputNode->number == other.inputNode->number &&
        this->outputNode->layer == other.outputNode->layer &&
        this->outputNode->number == other.outputNode->number;
}
