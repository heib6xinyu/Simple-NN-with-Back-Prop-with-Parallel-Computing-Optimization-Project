#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>
#include <string>

class Instance {
public:
    const std::vector<double> expectedOutputs;
    const std::vector<double> inputs;

    // Constructor declaration
    Instance(const std::vector<double>& expectedOutputs, const std::vector<double>& inputs);

    // Method declarations
    bool equals(const std::vector<double>& otherExpectedOutputs, const std::vector<double>& otherInputs) const;
    bool equals(const Instance& other) const;
    std::string toString() const;
};

#endif // INSTANCE_H
