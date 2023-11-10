// DataSet.h
#ifndef DATASET_H
#define DATASET_H

#include "Instance.h"
#include <string>
#include <vector>

class DataSet {
private:
    std::string name;
    std::string filename;
    std::vector<Instance> instances;
    int numberOutputs;
    int numberInputs;
    int numberClasses;

public:
    // Constructor declaration
    DataSet(const std::string& name, const std::string& filename);

    // Method declarations
    std::vector<double> getInputMeans();
    std::vector<double> getInputStandardDeviations();
    void normalize(const std::vector<double>& inputMeans, const std::vector<double>& inputStandardDeviations);

    // Accessors
    std::string getName() const;
    size_t getNumberInstances() const;
    int getNumberInputs() const;
    int getNumberOutputs() const;
    int getNumberClasses() const;

    // Other functionalities
    void shuffle();
    Instance getInstance(int position) const;
    std::vector<Instance> getInstances(int position, int numberOfInstances) const;
    const std::vector<Instance>& getInstances() const;
};

#endif // DATASET_H
