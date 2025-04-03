#include "audio_processor.h"
#include <portaudio.h>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

// MFCC constants matching Python training
constexpr int N_FFT = 512;
constexpr int HOP_LENGTH = 256;
constexpr int N_MEL = 20;
constexpr int N_MFCC = 10;
constexpr float SAMPLE_RATE = 16000.0f;
constexpr float MEL_LOW = 20.0f;
constexpr float MEL_HIGH = 4000.0f;
constexpr float PRE_EMPHASIS = 0.97f;

// Precomputed Mel filter banks
const std::vector<std::vector<float>> MEL_FILTERS = [] {
    std::vector<std::vector<float>> filters(N_MEL, std::vector<float>(N_FFT/2 + 1));
    const float mel_max = 1127.0f * logf(1.0f + MEL_HIGH/700.0f);
    const float mel_min = 1127.0f * logf(1.0f + MEL_LOW/700.0f);
    const float mel_spacing = (mel_max - mel_min)/(N_MEL + 1);
    
    std::vector<float> bin_freqs(N_FFT/2 + 1);
    for(int i = 0; i <= N_FFT/2; i++) {
        bin_freqs[i] = (SAMPLE_RATE/2.0f) * i/(N_FFT/2);
    }
    
    for(int m = 0; m < N_MEL; m++) {
        const float left_mel = mel_min + m * mel_spacing;
        const float center_mel = mel_min + (m + 1) * mel_spacing;
        const float right_mel = mel_min + (m + 2) * mel_spacing;
        
        for(int k = 0; k <= N_FFT/2; k++) {
            const float freq = bin_freqs[k];
            const float mel = 1127.0f * logf(1.0f + freq/700.0f);
            
            if(mel > left_mel && mel < right_mel) {
                filters[m][k] = 1.0f - fabsf(mel - center_mel)/mel_spacing;
            }
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
    inputParams.device = Pa_GetDefaultInputDevice();
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

std::vector<float> AudioProcessor::calculateMFCC(const std::vector<float>& audio) const {
    // 1. Pre-emphasis
    std::vector<float> emphasized(audio.size());
    emphasized[0] = audio[0];
    for(size_t i = 1; i < audio.size(); i++) {
        emphasized[i] = audio[i] - PRE_EMPHASIS * audio[i-1];
    }

    // 2. Windowing (Hann window)
    std::vector<float> windowed(emphasized.size());
    for(size_t i = 0; i < emphasized.size(); i++) {
        float window = 0.5f - 0.5f * cosf(2 * M_PI * i / (emphasized.size() - 1));
        windowed[i] = emphasized[i] * window;
    }

    // 3. FFT magnitude spectrum
    std::vector<float> spectrum(N_FFT/2 + 1);
    for(int k = 0; k <= N_FFT/2; k++) {
        float real = 0.0f;
        float imag = 0.0f;
        for(int n = 0; n < N_FFT; n++) {
            float angle = 2 * M_PI * k * n / N_FFT;
            real += windowed[n] * cosf(angle);
            imag -= windowed[n] * sinf(angle);
        }
        spectrum[k] = sqrtf(real*real + imag*imag);
    }

    // 4. Apply Mel filterbank
    std::vector<float> melEnergies(N_MEL, 0.0f);
    for(int m = 0; m < N_MEL; m++) {
        for(int k = 0; k <= N_FFT/2; k++) {
            melEnergies[m] += MEL_FILTERS[m][k] * spectrum[k];
        }
        melEnergies[m] = logf(melEnergies[m] + 1e-6f);
    }

    // 5. DCT to MFCC
    std::vector<float> mfcc(N_MFCC);
    for(int n = 0; n < N_MFCC; n++) {
        float sum = 0.0f;
        for(int m = 0; m < N_MEL; m++) {
            sum += melEnergies[m] * cosf(M_PI * n * (m + 0.5f) / N_MEL);
        }
        mfcc[n] = sum * sqrtf(2.0f / N_MEL);
    }

    return mfcc;
}

void AudioProcessor::processAudio(const float* input) {
    std::vector<float> buffer(input, input + framesPerBuffer);
    
    // Pad to minimum FFT size
    if(buffer.size() < N_FFT) {
        buffer.resize(N_FFT, 0.0f);
    }
    else if(buffer.size() > N_FFT) {
        buffer.resize(N_FFT);
    }

    // Calculate MFCCs
    auto mfcc = calculateMFCC(buffer);

    // Standardize features
    for(int i = 0; i < N_MFCC; i++) {
        mfcc[i] = (mfcc[i] - featureMeans[i]) / featureStds[i];
    }

    // SVM decision function
    float score = intercept;
    for(int i = 0; i < N_MFCC; i++) {
        score += mfcc[i] * weights[i];
    }

    // Sigmoid probability
    float probability = 1.0f / (1.0f + expf(-score));
    detectionFlag.store(probability > 0.7f);
}