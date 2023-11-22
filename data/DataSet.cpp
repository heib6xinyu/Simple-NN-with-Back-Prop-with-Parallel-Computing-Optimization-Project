#include "DataSet.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include "Instance.h"


DataSet::DataSet(const std::string& name, const std::string& filename) : name(name), filename(filename), numberOutputs(-1), numberInputs(-1), numberClasses(0) {
    std::set<double> potentialOutputs;
    std::ifstream file(filename);
    std::string line;
    int lineCount = 0;

    if (!file.is_open()) {
        std::cerr << "ERROR opening DataSet file: '" << filename << "'" << std::endl;
        exit(1);
    }

    while (getline(file, line)) {
        lineCount++;
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments

        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            throw std::runtime_error("Line " + std::to_string(lineCount) + " is not properly formatted.");
        }

        std::string outputPart = line.substr(0, colonPos);
        std::string inputPart = line.substr(colonPos + 1);

        std::vector<double> outputs;
        std::vector<double> inputs;

        std::istringstream osstream(outputPart);
        std::istringstream isstream(inputPart);
        std::string value;

        while (getline(osstream, value, ',')) {
            outputs.push_back(std::stod(value));
        }

        while (getline(isstream, value, ',')) {
            inputs.push_back(std::stod(value));
        }

        if (numberOutputs == -1) {
            numberOutputs = outputs.size();
        }
        else if (outputs.size() != static_cast<size_t>(numberOutputs)) {
            throw std::runtime_error("Inconsistent number of outputs on line " + std::to_string(lineCount));
        }

        if (numberInputs == -1) {
            numberInputs = inputs.size();
        }
        else if (inputs.size() != static_cast<size_t>(numberInputs)) {
            throw std::runtime_error("Inconsistent number of inputs on line " + std::to_string(lineCount));
        }

        for (double output : outputs) {
            potentialOutputs.insert(output);
        }

        instances.emplace_back(outputs, inputs);
    }

    numberClasses = potentialOutputs.size();
}

std::vector<double> DataSet::getInputMeans() {
    std::vector<double> inputMeans(numberInputs, 0.0);
    for (const auto& instance : instances) {
        for (size_t i = 0; i < instance.inputs.size(); ++i) {
            inputMeans[i] += instance.inputs[i];
        }
    }
    for (double& mean : inputMeans) {
        mean /= instances.size();
    }
    return inputMeans;
}

std::vector<double> DataSet::getInputStandardDeviations() {
    std::vector<double> inputMeans = getInputMeans();
    std::vector<double> inputVariances(numberInputs, 0.0);

    for (const auto& instance : instances) {
        for (size_t i = 0; i < instance.inputs.size(); ++i) {
            inputVariances[i] += std::pow(instance.inputs[i] - inputMeans[i], 2);
        }
    }

    for (size_t i = 0; i < inputVariances.size(); ++i) {
        inputVariances[i] = sqrt(inputVariances[i] / (instances.size() - 1));
    }

    return inputVariances;
}

void DataSet::normalize(const std::vector<double>& inputMeans, const std::vector<double>& inputStandardDeviations) {
    for (Instance instance : instances) {
        for (size_t i = 0; i < instance.inputs.size(); ++i) {
            instance.inputs[i] = (instance.inputs[i] - inputMeans[i]) / inputStandardDeviations[i];
        }
    }
}

std::string DataSet::getName() const {
    return name;
}

size_t DataSet::getNumberInstances() const {
    return instances.size();
}

int DataSet::getNumberInputs() const {
    return numberInputs;
}

int DataSet::getNumberOutputs() const {
    return numberOutputs;
}

int DataSet::getNumberClasses() const {
    return numberClasses;
}

void DataSet::shuffle() {
    std::random_shuffle(instances.begin(), instances.end());
}

Instance DataSet::getInstance(int position) const {
    return instances[position];
}

std::vector<Instance> DataSet::getInstances(int position, int numberOfInstances) const {
    int endPosition = std::min(position + numberOfInstances, static_cast<int>(instances.size()));
    return std::vector<Instance>(instances.begin() + position, instances.begin() + endPosition);
}

const std::vector<Instance>& DataSet::getInstances() const {
    return instances;
}
