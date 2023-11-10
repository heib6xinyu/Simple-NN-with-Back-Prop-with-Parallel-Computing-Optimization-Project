#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

class ConvertMushroom {
private:
    std::vector<std::vector<std::string>> columns{
        {"b", "c", "x", "f", "k", "s"},
        {"f", "g", "y", "s"},
        {"n", "b", "c", "g", "r", "p", "u", "e", "w", "y"},
        {"t", "f"},
        {"a", "l", "c", "y", "f", "m", "n", "p", "s"},
        {"a", "d", "f", "n"},
        {"c", "w", "d"},
        {"b", "n"},
        {"k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"},
        {"e", "t"},
        {"b", "c", "u", "e", "z", "r", "?"}, 
        {"f", "y", "k", "s"},
        {"f", "y", "k", "s"},
        {"n", "b", "c", "g", "o", "p", "e", "w", "y"},
        {"n", "b", "c", "g", "o", "p", "e", "w", "y"},
        {"p", "u"},
        {"n", "o", "w", "y"},
        {"n", "o", "t"},
        {"c", "e", "f", "l", "n", "p", "s", "z"},
        {"k", "n", "b", "h", "r", "o", "u", "w", "y"},
        {"a", "c", "n", "s", "v", "y"},
        {"g", "l", "m", "p", "u", "w", "d"}
    };

public:
    void run() {
        std::ifstream bufferedReader("./datasets/agaricus-lepiota.data");
        std::ofstream bufferedWriter("./datasets/agaricus-lepiota.txt");

        if (!bufferedReader.is_open()) {
            std::cerr << "ERROR opening agaricus-lepiota.data file for reading." << std::endl;
            return;
        }
        if (!bufferedWriter.is_open()) {
            std::cerr << "ERROR opening agaricus-lepiota.txt file for writing." << std::endl;
            return;
        }

        std::string readLine;
        while (getline(bufferedReader, readLine)) {
            if (readLine.empty() || readLine.front() == '#') {
                continue;
            }

            std::istringstream lineStream(readLine);
            std::string value;
            std::vector<std::string> values;
            while (getline(lineStream, value, ',')) {
                values.push_back(value);
            }

            std::string sampleClass = values.front();
            std::string binaryString;

            binaryString += (sampleClass == "p") ? "1" : "0";
            binaryString += ":";

            for (size_t i = 1; i < values.size(); ++i) {
                for (size_t j = 0; j < columns[i - 1].size(); ++j) {
                    binaryString += (values[i] == columns[i - 1][j]) ? "1" : "0";
                    if (j < columns[i - 1].size() - 1) {
                        binaryString += ",";
                    }
                }
                if (i < values.size() - 1) {
                    binaryString += ",";
                }
            }
            binaryString += "\n";
            bufferedWriter << binaryString;
        }

        bufferedReader.close();
        bufferedWriter.close();
    }
};

int main() {
    ConvertMushroom converter;
    converter.run();
    return 0;
}
