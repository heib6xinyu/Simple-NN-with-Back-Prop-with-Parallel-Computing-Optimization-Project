#include "Edge.h"
#include "../util/Log.h" 
#include <iostream>

Edge::Edge(Node* inputNode, Node* outputNode) : weight(0), weightDelta(0), inputNode(inputNode), outputNode(outputNode) {
    Log::debug("Created a new edge with input " + inputNode->toString() + " and output " + outputNode->toString());
}

void Edge::propagateBackward(double delta) {
    // 1. set the weightDelta to be the updated delta of its outputNode * postActivationValue of inputNode
    weightDelta = delta * inputNode->postActivationValue;
    
    // 2. the delta of the inputNode to be the product of weight and the updated delta of the its outputNode.
    // += because if multiple edges come into the same inputNode, add them up
    inputNode->delta += weight * delta;
}

void Edge::setWeight(double new_weight) {
    weight = new_weight;
}

double Edge::getWeight() {
    return weight;
}

bool Edge::equals(Edge other) const {
    return this->inputNode->layer == other.inputNode->layer &&
        this->inputNode->number == other.inputNode->number &&
        this->outputNode->layer == other.outputNode->layer &&
        this->outputNode->number == other.outputNode->number;
}

std::string Edge::toString() {
    return "Edge Input Node: " + inputNode->toString() + " Output Node: " + outputNode->toString();
}
