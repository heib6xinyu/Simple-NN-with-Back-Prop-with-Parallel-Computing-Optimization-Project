#ifndef CONVERT_IRIS_H
#define CONVERT_IRIS_H

#include <string>
#include "util/Logger.h"
class ConvertIris {
public:
    void run();

private:
    std::string mapClassToNumber(const std::string& sampleClass);
};

#endif // CONVERT_IRIS_H
