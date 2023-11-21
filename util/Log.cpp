#include "Log.h"
#include <iostream>

Log::Level Log::logLevel = Log::INFO;  // Default level

void Log::setLevel(Log::Level newLevel) {
    logLevel = newLevel;
    std::cout << "Log level set to " << levelToString(newLevel) << std::endl;
}

void Log::fatal(const std::string& message) {
    if (Log::FATAL <= logLevel) std::cout << "[FATAL  ] " << message << std::endl;
}

void Log::error(const std::string& message) {
    if (Log::ERROR <= logLevel) std::cout << "[ERROR  ] " << message << std::endl;
}

void Log::warning(const std::string& message) {
    if (Log::WARNING <= logLevel) std::cout << "[WARNING] " << message << std::endl;
}

void Log::info(const std::string& message) {
    if (Log::INFO <= logLevel) std::cout << "[INFO   ] " << message << std::endl;
}

void Log::debug(const std::string& message) {
    if (Log::DEBUG <= logLevel) std::cout << "[DEBUG  ] " << message << std::endl;
}

void Log::trace(const std::string& message) {
    if (Log::TRACE <= logLevel) std::cout << "[TRACE  ] " << message << std::endl;
}

std::string Log::levelToString(Log::Level level) {
    static const char* levelNames[] = { "NONE", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE", "ALL" };
    return levelNames[level];
}
