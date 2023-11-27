#ifndef NODE_H
#define NODE_H

#include <vector>
#include <string>
#include "ActivationType.h"
#include <memory>

// Forward declaration of Edge class to avoid circular dependencies
class Edge;

// Enumerations for Node type and activation functions
enum class NodeType {
    INPUT,
    HIDDEN,
    OUTPUT
};

//enum class ActivationType {
//    LINEAR,
//    SIGMOID,
//    TANH
//};

class Node {
private:
    NodeType nodeType;
    ActivationType activationType;
    double activationDerivative;
    double bias;
    double biasDelta;
    std::vector<std::shared_ptr<Edge>> inputEdges;
    std::vector<std::shared_ptr<Edge>> outputEdges;

    // Helper methods for activation functions
    void applyLinear();
    void applySigmoid();
    void applyTanh();
    void applyDerivativeLinear();
    void applyDerivativeSigmoid();
    void applyDerivativeTanh();

public:
    // Constructor and destructor
    Node(int layerValue, int numberValue, NodeType type, ActivationType actType);

    // Method to reset node state
    void reset();

    // Edge management
    void addOutgoingEdge(std::shared_ptr<Edge> outgoingEdge);
    void addIncomingEdge(std::shared_ptr<Edge> incomingEdge);

    // Propagation methods
    void propagateForward();
    void propagateBackward();

    // Weights and deltas management
    int getWeights(int position, std::vector<double>& weights) const;
    int getDeltas(int position, std::vector<double>& deltas);
    int setWeights(int position, std::vector<double>& weights);
    void setBias(double bias);

    std::vector<std::shared_ptr<Edge>> getInputEdges(); 

    // Initialization of weights and bias
    void initializeWeightsAndBias(double newBias);

    // Utility methods for printing node details
    std::string toString() const;
    std::string toDetailedString() const;

    int layer;
    int number;
    double delta;
    double postActivationValue;
    double preActivationValue;
};

#endif // NODE_H
