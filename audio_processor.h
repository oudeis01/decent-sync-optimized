#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <vector>
#include <portaudio.h>
#include <atomic>
#include <string>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstring>
#include <unistd.h>
#include <csignal>
#include <stdexcept>


class AudioProcessor {
public:
    AudioProcessor(int sampleRate, int framesPerBuffer);
    ~AudioProcessor();
    
    void start();
    void stop();
    void loadModel(const std::string& weightsPath, const std::string& scalePath);
    
    std::atomic<bool> detectionFlag{false};

private:
    PaStream* stream;
    int sampleRate;
    int framesPerBuffer;
    std::vector<float> weights;
    std::vector<float> featureMeans;
    std::vector<float> featureStds;
    float intercept = 0.0f;
    float threshold = 0.7f;

    static int audioCallback(const void* input, void* output,
                            unsigned long frameCount,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void* userData);
    
    std::vector<float> calculateMFCC(const std::vector<float>& audio) const;
    void processAudio(const float* input);
};

#endif