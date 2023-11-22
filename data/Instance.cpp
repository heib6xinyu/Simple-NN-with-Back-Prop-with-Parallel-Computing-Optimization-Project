#include "Instance.h"
#include <vector>
#include <string>
#include <sstream>

// Constructor that takes vectors for inputs and expected outputs
Instance::Instance(const std::vector<double>& expectedOutputs, const std::vector<double>& inputs)
    : expectedOutputs(expectedOutputs), inputs(inputs) {}

// Compares the expected outputs and inputs of this Instance to another set
bool Instance::equals(const std::vector<double>& otherExpectedOutputs, const std::vector<double>& otherInputs) const {
    if (expectedOutputs != otherExpectedOutputs) return false;
    if (inputs != otherInputs) return false;
    return true;
}

// Compares this Instance to another Instance
bool Instance::equals(const Instance& other) const {
    return equals(other.expectedOutputs, other.inputs);
}

// Generates a readable string representation of this Instance
std::string Instance::toString() const {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < expectedOutputs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << expectedOutputs[i];
    }

    oss << " : ";

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << inputs[i];
    }

    oss << "]";
    return oss.str();
}