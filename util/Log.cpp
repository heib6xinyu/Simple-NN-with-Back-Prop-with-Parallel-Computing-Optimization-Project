#include "Log.h"
#include <iostream>

Log::Level Log::logLevel = Log::INFO;  // Default level

void Log::setLevel(Log::Level newLevel) {
    logLevel = newLevel;
    //std::cout << "Log level set to " << levelToString(newLevel) << std::endl;
}

void Log::fatal(const std::string& message) {
    if (Log::FATAL <= logLevel) { printf("[FATAL  ] %s\n", message.c_str()); }
}

void Log::error(const std::string& message) {
    if (Log::ERROR <= logLevel) { printf("[ERROR  ] %s\n", message.c_str()); }
}

void Log::warning(const std::string& message) {
    if (Log::WARNING <= logLevel) { printf("[WARNING] %s\n", message.c_str()); }
}

void Log::info(std::string message) {
    if (Log::INFO <= logLevel) { printf("[INFO   ] %s\n", message.c_str()); } 
}

void Log::debug(const std::string& message) {
    if (Log::DEBUG <= logLevel) { printf("[DEBUG  ] %s\n", message.c_str()); }
}

void Log::trace(const std::string& message) {
    if (Log::TRACE <= logLevel) { printf("[TRACE  ] %s\n", message.c_str()); }
}

std::string Log::levelToString(Log::Level level) {
    static const char* levelNames[] = { "NONE", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE", "ALL" };
    return levelNames[level];
}
