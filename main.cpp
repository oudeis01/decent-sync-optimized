#include "main.h"
#include "colorPalette.h"
#include "audio_processor.h"
#include <thread>
#include <chrono>

std::atomic<bool> shutdown_flag(false);
std::atomic<bool> worker_running(true);
std::atomic<bool> audio_detection(false);
std::queue<Command> cmd_queue;
std::mutex queue_mutex;
std::condition_variable cv;
std::atomic<int> cmd_index(0);

void audioProcessingThread(AudioProcessor& processor) {
    while (!shutdown_flag) {
        if (processor.detectionFlag.exchange(false)) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            Command audio_cmd;
            audio_cmd.type = Command::ROTATE;
            audio_cmd.steps = 200;  // Default rotation steps
            audio_cmd.delayUs = 1000;
            audio_cmd.direction = true;
            audio_cmd.index = ++cmd_index;
            cmd_queue.push(audio_cmd);
            std::cout << Color::rcvTag() << " AUDIO DETECTION -> ROTATE ADDED TO QUEUE\n";
            cv.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void commandWorker(Motor& motor, 
                  std::queue<Command>& cmd_queue,
                  std::mutex& queue_mutex,
                  std::condition_variable& cv) {
    while (worker_running && !shutdown_flag) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [&]{ 
            return !cmd_queue.empty() || !worker_running || shutdown_flag; 
        });

        if (!worker_running || shutdown_flag) break;
        
        Command cmd = cmd_queue.front();
        cmd_queue.pop();
        lock.unlock();

        try {
            switch (cmd.type) {
                case Command::ROTATE:
                    std::cout << Color::runTag() << " Processing ROTATE from AUDIO\n";
                    std::cout << Color::runTag() << " Starting rotation - Steps: " 
                            << Color::value(cmd.steps) << ", Delay: " 
                            << Color::value(cmd.delayUs) << "μs\n";
                    motor.rotate(cmd.steps, static_cast<int>(cmd.delayUs), cmd.direction);
                    Sender::sendDone(cmd.senderIp, 12345, cmd.index, std::string("ROTATE"));
                    break;
                case Command::ENABLE:
                    std::cout << Color::runTag() << " Enabling motor\n";
                    motor.enable();
                    Sender::sendDone(cmd.senderIp, 12345, cmd.index, std::string("ENABLE"));
                    break;
                case Command::DISABLE:
                    std::cout << Color::runTag() << " Disabling motor\n";
                    motor.disable();
                    Sender::sendDone(cmd.senderIp, 12345, cmd.index, std::string("DISABLE"));
                    break;
                case Command::EXIT:
                    std::cout << Color::successTag() << " Graceful shutdown initiated via OSC\n";
                    shutdown_flag.store(true);  // Add this line
                    worker_running = false;
                    break;
                case Command::INFO:
                    break;
            }
        } catch (const std::exception& e) {
            std::cerr << Color::errorTag() << " Error executing command: " << e.what() << "\n";
        }
    }
}

void signalHandler(int signal) {
    if (!shutdown_flag.exchange(true)) {
        std::cerr << Color::errorTag() << " Received signal " << signal 
                << ", initiating shutdown..." << std::endl;
    }
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGSEGV, signalHandler);

    if (gpioInitialise() < 0) return 1;

    std::queue<Command> cmd_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<int> cmd_index(0);

    Motor motor(EN_PIN, DIR_PIN, STEP_PIN);
    Receiver receiver(9000, cmd_queue, queue_mutex, cmd_index, cv);
    receiver.start();

    std::thread worker(commandWorker, 
                     std::ref(motor), 
                     std::ref(cmd_queue),
                     std::ref(queue_mutex), 
                     std::ref(cv));

    AudioProcessor audioProcessor;
    try {
        audioProcessor.loadModel("model_weights.txt", "feature_scale.txt");
    } catch (const std::exception& e) {
        std::cerr << "Audio init failed: " << e.what() << std::endl;
        return 1;
    }

    // Start audio thread
    std::thread audio_thread(audioProcessingThread, std::ref(audioProcessor));
    audioProcessor.start();

    // Main monitoring loop
    while (!shutdown_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Cleanup sequence
    worker_running = false;
    shutdown_flag = true;
    cv.notify_one();

    audioProcessor.stop();
    if (audio_thread.joinable()) audio_thread.join();

    receiver.stop(true);

    if (worker.joinable()) {
        worker.join();
    }

    receiver.stop(true);  // Ensure waiting for server thread
    motor.disable();
    gpioTerminate();

    std::cout << Color::successTag() << " Application shutdown complete\n";
    return 0;
}