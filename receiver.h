#ifndef RECEIVER_H
#define RECEIVER_H

#include <lo/lo.h>
#include <queue>
#include <mutex>
#include <atomic>
#include <string>
#include <condition_variable>
#include <thread>
#include <unordered_set>

struct Command {
    int index;
    std::string senderIp;
    int senderPort;
    enum Type { ROTATE, ENABLE, DISABLE, INFO, EXIT} type;
    int steps;
    float delayUs; // Changed to float
    bool direction;
};

class Receiver {
public:
    Receiver(int port, std::queue<Command>& queue, std::mutex& mutex,
             std::atomic<int>& cmdIndex, std::condition_variable& cv);
    ~Receiver();
    std::string getLocalIp() const;
    void start();
    void stop(bool wait_for_thread = true);
    
private:
    static int oscHandler(const char *path, const char *types, 
                         lo_arg **argv, int argc, lo_message msg, void *user_data);

    std::atomic<bool> running_{false};
    int port_;
    std::queue<Command>& commandQueue_;
    std::mutex& queueMutex_;
    std::atomic<int>& commandIndex_;
    std::condition_variable& cv_;
    lo_server_thread server_thread_;
    std::unordered_set<std::string> connected_clients_;
};

#endif