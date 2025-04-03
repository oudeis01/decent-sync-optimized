#ifndef MOTOR_H
#define MOTOR_H

#include <pigpio.h>
#include <mutex>

class Motor {
public:
    Motor(int enPin, int dirPin, int stepPin);
    ~Motor();
    void enable();
    void disable();
    void rotate(int steps, int delayUs, bool direction);

private:
    int enPin_;
    int dirPin_;
    int stepPin_;
    bool isEnabled_;
    std::mutex motor_mutex_;
};

#endif