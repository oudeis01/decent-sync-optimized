#include "motor.h"
#include <thread>
#include <chrono>

Motor::Motor(int enPin, int dirPin, int stepPin)
    : enPin_(enPin), dirPin_(dirPin), stepPin_(stepPin), isEnabled_(false) {
    gpioSetMode(enPin_, PI_OUTPUT);
    gpioSetMode(dirPin_, PI_OUTPUT);
    gpioSetMode(stepPin_, PI_OUTPUT);
    disable();
}

Motor::~Motor() { disable(); }

void Motor::enable() {
    std::lock_guard<std::mutex> lock(motor_mutex_);
    gpioWrite(enPin_, 0);
    isEnabled_ = true;
}

void Motor::disable() {
    std::lock_guard<std::mutex> lock(motor_mutex_);
    gpioWrite(enPin_, 1);
    isEnabled_ = false;
}

void Motor::rotate(int steps, int delayUs, bool direction) {
    std::lock_guard<std::mutex> lock(motor_mutex_);
    if (!isEnabled_) return;
    
    gpioWrite(dirPin_, direction ? 1 : 0);
    for (int i = 0; i < steps; ++i) {
        gpioWrite(stepPin_, 1);
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        gpioWrite(stepPin_, 0);
        std::this_thread::sleep_for(std::chrono::microseconds(delayUs - 1));
    }
}