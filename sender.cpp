#include "sender.h"
#include "colorPalette.h"
#include <lo/lo.h>
#include <iostream>

constexpr int RESPONSE_PORT = 12345;

void Sender::sendAck(const std::string& ip, int port, int index) {
    lo_address addr = lo_address_new(ip.c_str(), std::to_string(port).c_str());
    lo_message msg = lo_message_new();
    lo_message_add_int32(msg, index);
    lo_send_message(addr, "/ack", msg);
    lo_message_free(msg);
    lo_address_free(addr);
}

void Sender::sendDone(const std::string& ip, int port, int index, std::string type_name) {
    lo_address addr = lo_address_new(ip.c_str(), std::to_string(port).c_str());
    lo_message msg = lo_message_new();
    lo_message_add_int32(msg, index);
    lo_send_message(addr, "/done", msg);
    lo_message_free(msg);
    lo_address_free(addr);
    std::cout << Color::successTag() << " Completed command #" <<
    Color::value(index) <<  " " << type_name <<"\n";
}

void Sender::sendInfo(const std::string& ip, int port, const std::queue<Command>& queue) {
    lo_address addr = lo_address_new(ip.c_str(), std::to_string(RESPONSE_PORT).c_str());
    lo_message msg = lo_message_new();
    std::string qstr = "";

    if (queue.empty()) {
        lo_message_add_string(msg, "no commands in the queue");
        lo_send_message(addr, "/info", msg);
    } else {
        std::queue<Command> q_copy = queue;
        while (!q_copy.empty()) {
            const Command& cmd = q_copy.front();
            switch (cmd.type) {
                case Command::ROTATE:
                    qstr += "rotate " + std::to_string(cmd.steps) + ", " + std::to_string(cmd.delayUs) + ", " + std::to_string(cmd.direction) + "\n";
                    break;
                case Command::ENABLE:
                    qstr += "enable motor #" + std::to_string(cmd.index) + "\n";
                    break;
                case Command::DISABLE:
                    qstr += "disable motor #" + std::to_string(cmd.index) + "\n";
                    break;
                case Command::INFO:
                    qstr += "get info for motor #" + std::to_string(cmd.index) + "\n";
                    break;
                case Command::EXIT:
                    qstr += "exit\n";
                    break;
            }
            q_copy.pop();
        }
        lo_message_add_string(msg, qstr.c_str());
        lo_send_message(addr, "/info", msg);

    }
    lo_message_free(msg);
    lo_address_free(addr);
}