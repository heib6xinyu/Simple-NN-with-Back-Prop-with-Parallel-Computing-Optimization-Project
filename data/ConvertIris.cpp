#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "util/Logger.h"
class ConvertIris {
public:
    void run() {
        try {
            std::ifstream bufferedReader("./datasets/iris.data");
            std::ofstream bufferedWriter("./datasets/iris.txt");

            if (!bufferedReader.is_open()) {
                Logger::fatal("ERROR opening iris.data file for reading.");
                return;
            }
            if (!bufferedWriter.is_open()) {
                Logger::fatal("ERROR opening iris.txt file for writing.");
                return;
            }

            std::string readLine;
            // Read the file line by line
            while (std::getline(bufferedReader, readLine)) {
                Logger::info(readLine);

                // Skip empty lines or lines starting with '#'
                if (readLine.empty() || readLine[0] == '#') {
                    continue;
                }

                std::istringstream iss(readLine);
                std::string token;
                std::vector<std::string> values;

                while (std::getline(iss, token, ',')) {
                    values.push_back(token);
                }

                std::string sampleClass = values.back(); // The class is the last element
                std::string output;

                // Map the sample class to a numerical value
                output = mapClassToNumber(sampleClass);

                // Check if the mapping was successful
                if (output.empty()) {
                    std::cerr << "ERROR: unknown class in iris.data file: '" << sampleClass << "'\n";
                    std::cerr << "This should not happen.\n";
                    return;
                }

                output += ":";

                // Concatenate the input values for the neural network
                for (size_t i = 0; i < values.size() - 1; ++i) {
                    if (i > 0) output += ",";
                    output += values[i];
                }
                output += "\n";

                Logger::info(output);
                bufferedWriter.write(output.c_str(), output.size());
            }
            bufferedWriter.close();
            bufferedReader.close();
        }
        catch (const std::exception& e) {
            Logger::fatal("ERROR converting iris data file");
            std::cerr << e.what() << '\n';
            return;
        }
    }

private:
    std::string mapClassToNumber(const std::string& sampleClass) {
        if (sampleClass == "Iris-setosa") {
            return "0";
        }
        else if (sampleClass == "Iris-versicolor") {
            return "1";
        }
        else if (sampleClass == "Iris-virginica") {
            return "2";
        }
        return ""; // Return an empty string if class is unknown
    }
};

int main() {
    ConvertIris converter;
    converter.run();
    return 0;
}
