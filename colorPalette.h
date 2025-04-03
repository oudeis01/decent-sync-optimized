#ifndef COLOR_PALETTE_H
#define COLOR_PALETTE_H

#include <string>

class Color {
public:
    // Base colors
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";

    // Formatting
    static constexpr const char* BOLD = "\033[1m";
    static constexpr const char* DIM = "\033[2m";

    // Utility functions
    static std::string style(const std::string& text, 
                            const char* color, 
                            const char* format = RESET) {
        return std::string(format) + color + text + RESET;
    }

    // Predefined styled elements
    static std::string cmdTag() { 
        static const std::string boldYellow = std::string(BOLD) + YELLOW;
        return style("[CMD ]", boldYellow.c_str()); 
    }

    static std::string rcvTag() { 
        static const std::string boldBlue = std::string(BOLD) + BLUE;
        return style("[RCV ]", boldBlue.c_str()); 
    }

    static std::string runTag() { 
        static const std::string boldMagenta = std::string(BOLD) + MAGENTA;
        return style("[RUN ]", boldMagenta.c_str()); 
    }
    
    static std::string errorTag() { 
        static const std::string boldRed = std::string(BOLD) + RED;
        return style("[ERRO]", boldRed.c_str()); 
    }
    
    static std::string successTag() { 
        static const std::string boldGreen = std::string(BOLD) + GREEN;
        return style("[DONE]", boldGreen.c_str()); 
    }
    
    static std::string client(const std::string& ip_port) { 
        return style(ip_port, MAGENTA); 
    }
    
    static std::string value(const std::string& val) { 
        return style(val, CYAN); 
    }
    
    static std::string value(float val) { 
        return style(std::to_string(val), CYAN); 
    }
    
    static std::string value(int val) { 
        return style(std::to_string(val), CYAN); 
    }
    
    static std::string header(const std::string& text) { 
        static const std::string boldBlue = std::string(BOLD) + BLUE;
        return style(text, boldBlue.c_str()); 
    }
};

#endif