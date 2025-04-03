#ifndef MAIN_H
#define MAIN_H

#include "motor.h"
#include "receiver.h"
#include "sender.h"
#include <pigpio.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <csignal>
#include <iostream>

#define EN_PIN 24
#define DIR_PIN 23
#define STEP_PIN 18

extern std::atomic<bool> shutdown_flag;
extern std::atomic<bool> worker_running;

void signalHandler(int signal);

#endif