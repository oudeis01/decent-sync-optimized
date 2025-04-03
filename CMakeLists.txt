cmake_minimum_required(VERSION 3.12)
project(motor_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBLO REQUIRED liblo)
include_directories(${LIBLO_INCLUDE_DIRS})

find_library(PIGPIO_LIBRARY pigpio)
if(NOT PIGPIO_LIBRARY)
    message(FATAL_ERROR "pigpio library not found!")
endif()

set(SOURCES
    main.cpp
    motor.cpp
    receiver.cpp
    sender.cpp
)

add_executable(motor_control ${SOURCES})

target_link_libraries(motor_control
    ${PIGPIO_LIBRARY}
    ${LIBLO_LIBRARIES}
    pthread
    rt
    m
)

target_compile_options(motor_control PRIVATE 
    -Wall
    -O2
    -Wno-unused-parameter
)