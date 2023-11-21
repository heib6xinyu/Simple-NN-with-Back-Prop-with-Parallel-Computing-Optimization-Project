#ifndef CONVERT_IRIS_H
#define CONVERT_IRIS_H

#include <string>
class ConvertIris {
public:
    void run();

private:
    std::string mapClassToNumber(const std::string& sampleClass);
};

#endif // CONVERT_IRIS_H
