#include "audio_processor.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// Mel filterbank matching librosa's implementation for 16kHz
const std::vector<std::vector<float>> MEL_FILTERS = [] {
    constexpr float fmax = 4000.0f;
    constexpr int n_mels = AudioProcessor::N_MEL;
    constexpr int n_fft = AudioProcessor::N_FFT;
    
    std::vector<std::vector<float>> filters(n_mels, 
        std::vector<float>(n_fft/2 + 1, 0.0f));

    auto mel = [](float f) { return 2595.0f * log10f(1.0f + f/700.0f); };
    auto inv_mel = [](float m) { return 700.0f * (powf(10.0f, m/2595.0f) - 1.0f); };
    
    std::vector<float> hz_points(n_mels + 2);
    const float max_mel = mel(fmax);
    for(int i=0; i<n_mels+2; i++) {
        hz_points[i] = inv_mel(max_mel * i/(n_mels + 1));
    }
    
    for(int m=1; m<=n_mels; m++) {
        const float left = hz_points[m-1];
        const float center = hz_points[m];
        const float right = hz_points[m+1];
        
        for(int k=0; k<=n_fft/2; k++) {
            const float freq = (AudioProcessor::TARGET_SAMPLE_RATE/2.0f) * k/(n_fft/2);
            float weight = 0.0f;
            
            if(freq > left && freq <= center) {
                weight = (freq - left)/(center - left);
            } else if(freq > center && freq < right) {
                weight = (right - freq)/(right - center);
            }
            
            filters[m-1][k] = weight * 2.0f / (right - left);
        }
    }
    
    return filters;
}();

AudioProcessor::AudioProcessor() = default;

AudioProcessor::~AudioProcessor() { stop(); }

void AudioProcessor::loadModel(const std::string& weightsPath, const std::string& scalePath) {
    std::ifstream weightsFile(weightsPath);
    std::ifstream scaleFile(scalePath);
    
    if(!weightsFile || !scaleFile) {
        throw std::runtime_error("Failed to open model files");
    }

    // Load weights
    std::string line;
    std::getline(weightsFile, line);
    intercept = std::stof(line);
    weights.clear();
    while(std::getline(weightsFile, line)) {
        weights.push_back(std::stof(line));
    }

    // Load feature scaling
    std::vector<float> scales;
    while(std::getline(scaleFile, line)) {
        scales.push_back(std::stof(line));
    }
    
    if(scales.size() != N_MFCC * 2) {
        throw std::runtime_error("Invalid feature scale file");
    }
    
    featureMeans = {scales.begin(), scales.begin() + N_MFCC};
    featureStds = {scales.begin() + N_MFCC, scales.end()};
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
                    SOURCE_SAMPLE_RATE, paFramesPerBufferUnspecified, paClipOff,
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
    }
}

int AudioProcessor::audioCallback(const void* input, void* output,
                                unsigned long frameCount,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void* userData) {
    AudioProcessor* processor = static_cast<AudioProcessor*>(userData);
    const auto* audioData = static_cast<const float*>(input);
    processor->resampleAndProcess(audioData, frameCount);
    return paContinue;
}
#include <samplerate.h>

void AudioProcessor::resampleAndProcess(const float* input, std::size_t frames) {
    // Initialize resampler once
    static SRC_STATE* resampler = [](){
        int error;
        SRC_STATE* s = src_new(SRC_SINC_FASTEST, 1, &error);
        if(!s) throw std::runtime_error("Resampler init failed: " + 
                                      std::string(src_strerror(error)));
        return s;
    }();

    // Configure resampling parameters
    constexpr double ratio = static_cast<double>(TARGET_SAMPLE_RATE) 
                           / SOURCE_SAMPLE_RATE;
    
    const double tmp = frames * ratio;
    const long outputFrames = static_cast<long>(tmp) + 128;
    
    // Temporary buffers
    static std::vector<float> inputBuffer;
    static std::vector<float> outputBuffer(outputFrames);

    // Setup resampling data
    SRC_DATA data;
    data.data_in = input;
    data.input_frames = frames;
    data.data_out = outputBuffer.data();
    data.output_frames = outputFrames;
    data.src_ratio = ratio;
    data.end_of_input = 0;

    // Perform resampling
    int error = src_process(resampler, &data);
    if(error) {
        throw std::runtime_error("Resampling failed: " + 
                               std::string(src_strerror(error)));
    }

    // Store resampled audio
    audioBuffer.insert(audioBuffer.end(), 
                      outputBuffer.begin(), 
                      outputBuffer.begin() + data.output_frames_gen);

    // Process in 0.5s chunks (16000 * 0.5 = 8000 samples)
    while(audioBuffer.size() >= 8000) {
        std::vector<float> chunk(audioBuffer.begin(), audioBuffer.begin() + 8000);
        audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + 8000);

        // Calculate MFCCs
        auto mfcc = calculateMFCC(chunk);

        // Feature standardization
        for(int i = 0; i < N_MFCC; ++i) {
            mfcc[i] = (mfcc[i] - featureMeans[i]) / featureStds[i];
        }

        // SVM classification
        float score = intercept;
        for(int i = 0; i < N_MFCC; ++i) {
            score += mfcc[i] * weights[i];
        }

        // Sigmoid activation
        float probability = 1.0f / (1.0f + expf(-score));
        detectionFlag.store(probability > 0.7f);

        // Debug output
        std::cout << "MFCC:";
        for(const auto& v : mfcc) std::cout << " " << v;
        std::cout << "\nScore: " << score 
                << " | Prob: " << probability 
                << " | Detect: " << detectionFlag.load() << "\n";
    }
}

std::vector<float> AudioProcessor::calculateMFCC(const std::vector<float>& audio) const {
    // Pre-emphasis
    std::vector<float> emphasized(audio.size());
    emphasized[0] = audio[0];
    for(size_t i=1; i<audio.size(); i++) {
        emphasized[i] = audio[i] - 0.97f * audio[i-1];
    }

    // Hann window
    std::vector<float> windowed(N_FFT);
    for(int i=0; i<N_FFT; i++) {
        windowed[i] = emphasized[i] * (0.5f - 0.5f * cosf(2*M_PI*i/(N_FFT-1)));
    }

    // FFT magnitude
    std::vector<float> spectrum(N_FFT/2 + 1);
    for(int k=0; k<=N_FFT/2; k++) {
        float real = 0.0f, imag = 0.0f;
        for(int n=0; n<N_FFT; n++) {
            float angle = 2*M_PI*k*n/N_FFT;
            real += windowed[n] * cosf(angle);
            imag -= windowed[n] * sinf(angle);
        }
        spectrum[k] = sqrtf(real*real + imag*imag);
    }

    // Mel energies
    std::vector<float> melEnergies(N_MEL, 0.0f);
    for(int m=0; m<N_MEL; m++) {
        for(int k=0; k<=N_FFT/2; k++) {
            melEnergies[m] += spectrum[k] * MEL_FILTERS[m][k];
        }
        melEnergies[m] = logf(melEnergies[m] + 1e-6f);
    }

    // DCT-II
    std::vector<float> mfcc(N_MFCC);
    for(int n=0; n<N_MFCC; n++) {
        float sum = 0.0f;
        for(int m=0; m<N_MEL; m++) {
            sum += melEnergies[m] * cosf(M_PI * n * (m + 0.5f)/N_MEL);
        }
        mfcc[n] = sum * sqrtf(2.0f/N_MEL);
    }
    std::cout << "C++ MFCC: ";
    for (const auto& val : mfcc) std::cout << val << " ";
    std::cout << "\n";

    return mfcc;
}