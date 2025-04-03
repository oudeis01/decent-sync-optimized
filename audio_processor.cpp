#include "audio_processor.h"
#include <portaudio.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <string>

// MFCC constants matching Python training
// for 44.1kHz
constexpr int N_FFT = 1024;       // Increased window size
constexpr int HOP_LENGTH = 512;   // 50% overlap
constexpr int N_MEL = 40;         // More filters for wider frequency range
constexpr int N_MFCC = 10;
constexpr float SAMPLE_RATE = 44100.0f;
constexpr float MEL_LOW = 20.0f;
constexpr float MEL_HIGH = 4000.0f;
constexpr float PRE_EMPHASIS = 0.97f;

// 1. Mel Filter Banks (EXACTLY matching librosa)
const std::vector<std::vector<float>> MEL_FILTERS = [] {
    constexpr float fmin = 0.0f;
    constexpr float fmax = 4000.0f;
    constexpr int n_mels = N_MEL;
    constexpr int n_fft = N_FFT;
    
    std::vector<std::vector<float>> filters(n_mels, 
        std::vector<float>(n_fft/2 + 1, 0.0f));

    // librosa's mel frequencies
    auto mel = [](float f) { return 2595.0f * log10f(1.0f + f/700.0f); };
    auto inv_mel = [](float m) { return 700.0f * (powf(10.0f, m/2595.0f) - 1.0f); };
    
    float max_mel = mel(fmax);
    std::vector<float> mel_points(n_mels + 2);
    for(int i=0; i<n_mels+2; ++i) {
        mel_points[i] = max_mel * i/(n_mels + 1);
    }
    
    // Convert to Hz
    std::vector<float> hz_points(n_mels + 2);
    for(int i=0; i<n_mels+2; ++i) {
        hz_points[i] = inv_mel(mel_points[i]);
    }
    
    // Create triangular filters
    for(int m=1; m<=n_mels; ++m) {
        float left = hz_points[m-1];
        float center = hz_points[m];
        float right = hz_points[m+1];
        
        for(int k=0; k<=n_fft/2; ++k) {
            float freq = (SAMPLE_RATE/2.0f) * k/(n_fft/2);
            
            if(freq >= left && freq <= center) {
                filters[m-1][k] = (freq - left)/(center - left);
            } else if(freq > center && freq <= right) {
                filters[m-1][k] = (right - freq)/(right - center);
            }
            
            // Librosa's area normalization
            filters[m-1][k] *= 2.0f / (right - left);
        }
    }
    
    return filters;
}();

AudioProcessor::AudioProcessor(int sampleRate, int framesPerBuffer) 
    : sampleRate(sampleRate), framesPerBuffer(framesPerBuffer) {
    if(sampleRate != SAMPLE_RATE) {
        throw std::runtime_error("Sample rate must be 16000Hz");
    }
}

AudioProcessor::~AudioProcessor() {
    stop();
}

void AudioProcessor::loadModel(const std::string& weightsPath, const std::string& scalePath) {
    std::ifstream weightsFile(weightsPath);
    std::ifstream scaleFile(scalePath);
    
    if(!weightsFile || !scaleFile) {
        throw std::runtime_error("Failed to open model files");
    }

    // Load SVM weights and intercept
    std::string line;
    std::getline(weightsFile, line);
    intercept = std::stof(line);
    
    weights.clear();
    while(std::getline(weightsFile, line)) {
        weights.push_back(std::stof(line));
    }

    // Load feature scaling parameters
    std::vector<float> scales;
    while(std::getline(scaleFile, line)) {
        scales.push_back(std::stof(line));
    }
    
    if(scales.size() != N_MFCC * 2) {
        throw std::runtime_error("Invalid feature scale file");
    }
    
    featureMeans = std::vector<float>(scales.begin(), scales.begin() + N_MFCC);
    featureStds = std::vector<float>(scales.begin() + N_MFCC, scales.end());
}

void AudioProcessor::start() {
    if(Pa_Initialize() != paNoError) {
        throw std::runtime_error("PortAudio initialization failed");
    }
    
    PaStreamParameters inputParams;
    inputParams.device = 1;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    if(Pa_OpenStream(&stream, &inputParams, nullptr,
                    sampleRate, framesPerBuffer, paClipOff,
                    &AudioProcessor::audioCallback, this) != paNoError) {
        Pa_Terminate();
        throw std::runtime_error("Failed to open audio stream");
    }
    
    if(Pa_StartStream(stream) != paNoError) {
        Pa_CloseStream(stream);
        Pa_Terminate();
        throw std::runtime_error("Failed to start audio stream");
    }
}

void AudioProcessor::stop() {
    if(stream) {
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        stream = nullptr;
    }
}

int AudioProcessor::audioCallback(const void* input, void* output,
                                unsigned long frameCount,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void* userData) {
    AudioProcessor* processor = static_cast<AudioProcessor*>(userData);
    const float* audioData = static_cast<const float*>(input);
    processor->processAudio(audioData);
    return paContinue;
}

// 2. MFCC Calculation (matches librosa exactly)
std::vector<float> AudioProcessor::calculateMFCC(const std::vector<float>& audio) const {
    // Hann window
    std::vector<float> windowed(N_FFT);
    for(size_t i=0; i<N_FFT; ++i) {
        windowed[i] = audio[i] * (0.5f - 0.5f * cosf(2*M_PI*i/(N_FFT-1)));
    }
    
    // FFT magnitude
    std::vector<float> spectrum(N_FFT/2 + 1, 0.0f);
    for(int k=0; k<=N_FFT/2; ++k) {
        float real = 0.0f, imag = 0.0f;
        for(int n=0; n<N_FFT; ++n) {
            float angle = 2*M_PI*k*n/N_FFT;
            real += windowed[n] * cosf(angle);
            imag -= windowed[n] * sinf(angle);
        }
        spectrum[k] = sqrtf(real*real + imag*imag);
    }
    
    // Mel filterbank
    std::vector<float> melEnergies(N_MEL, 0.0f);
    for(int m=0; m<N_MEL; ++m) {
        for(int k=0; k<=N_FFT/2; ++k) {
            melEnergies[m] += spectrum[k] * MEL_FILTERS[m][k];
        }
        melEnergies[m] = logf(melEnergies[m] + 1e-6f);
    }
    
    // DCT-II (librosa uses scipy's DCT-II)
    std::vector<float> mfcc(N_MFCC, 0.0f);
    for(int n=0; n<N_MFCC; ++n) {
        for(int m=0; m<N_MEL; ++m) {
            mfcc[n] += melEnergies[m] * cosf(M_PI * n * (m + 0.5f)/N_MEL);
        }
        mfcc[n] *= sqrtf(2.0f/N_MEL);
    }
    
    return mfcc;
}

std::vector<float> audioBuffer;
// audio_processor.cpp
void AudioProcessor::processAudio(const float* input) {
    // Use the pre-configured framesPerBuffer_
    audioBuffer.insert(audioBuffer.end(), input, input + framesPerBuffer);
    
    // Process in 0.5s chunks (44100 * 0.5 = 22050 samples)
    while(audioBuffer.size() >= 22050) {
        std::vector<float> chunk(audioBuffer.begin(), audioBuffer.begin() + 22050);
        audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + 22050);
        
        auto mfcc = calculateMFCC(chunk);
        
        // Standardize
        for(int i = 0; i < N_MFCC; ++i) {
            mfcc[i] = (mfcc[i] - featureMeans[i]) / featureStds[i];
        }
        
        // SVM decision
        float score = weights[0]; // intercept
        for(int i = 0; i < N_MFCC; ++i) {
            score += mfcc[i] * weights[i+1];
        }
        
        float probability = 1.0f / (1.0f + expf(-score));
        detectionFlag.store(probability > 0.7f);
    }
}