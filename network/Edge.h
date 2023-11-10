#ifndef EDGE_H
#define EDGE_H

#include "Node.h" 

class Edge {
public:
    double weight;
    double weightDelta;
    Node* inputNode;
    Node* outputNode;

    Edge(Node* inputNode, Node* outputNode);

    void propagateBackward(double delta);
    bool equals(const Edge& other) const;
};

#endif // EDGE_H
