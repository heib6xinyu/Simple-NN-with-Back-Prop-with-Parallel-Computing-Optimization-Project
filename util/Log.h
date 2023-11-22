#ifndef LOG_H
#define LOG_H

#include <string>

class Log {
public:
    enum Level {
        NONE, FATAL, ERROR, WARNING, INFO, DEBUG, TRACE, ALL
    };

    static void setLevel(Level newLevel);
    static void fatal(const std::string& message);
    static void error(const std::string& message);
    static void warning(const std::string& message);
    static void info(std::string message);
    static void debug(const std::string& message);
    static void trace(const std::string& message);

private:
    static Level logLevel;
    static std::string levelToString(Level level);
};

#endif // LOG_H
