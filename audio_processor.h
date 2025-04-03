#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <vector>
#include <portaudio.h>
#include <atomic>
#include <deque>
#include <string>
#include <cstddef>

class AudioProcessor {
public:
    static constexpr int TARGET_SAMPLE_RATE = 16000;
    static constexpr int SOURCE_SAMPLE_RATE = 44100;
    static constexpr int N_FFT = 512;
    static constexpr int HOP_LENGTH = 256;
    static constexpr int N_MEL = 40;
    static constexpr int N_MFCC = 10;
    static constexpr float DURATION = 0.5f;
    
    AudioProcessor();
    ~AudioProcessor();
    
    void start();
    void stop();
    void loadModel(const std::string& weightsPath, const std::string& scalePath);
    
    std::atomic<bool> detectionFlag{false};

private:
    PaStream* stream;
    std::deque<float> resampleBuffer;
    std::vector<float> audioBuffer;
    std::vector<float> weights;
    std::vector<float> featureMeans;
    std::vector<float> featureStds;
    float intercept = 0.0f;

    static int audioCallback(const void* input, void* output,
                            unsigned long frameCount,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void* userData);
    
    void resampleAndProcess(const float* input, std::size_t frames);
    std::vector<float> calculateMFCC(const std::vector<float>& audio) const;
};

#endif