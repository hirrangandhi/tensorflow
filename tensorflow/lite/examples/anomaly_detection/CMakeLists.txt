cmake_minimum_required(VERSION 3.10)
project(anomaly_app)

set(CMAKE_CXX_STANDARD 14)

add_executable(anomaly_app anomaly_app.cc anomaly_app.h)

# TensorFlow Lite and nlohmann/json include directories
include_directories(
    /path/to/tensorflow/lite
    ${CMAKE_SOURCE_DIR}/nlohmann_json
)

# Link TensorFlow Lite and any other required libraries
target_link_libraries(anomaly_app
    tensorflow-lite
    # Add other required libraries here
)

# For nlohmann/json (header-only), no need to link
# If installed system-wide, you can use find_package:
# find_package(nlohmann_json CONFIG REQUIRED)
# target_link_libraries(anomaly_app PRIVATE nlohmann_json::nlohmann_json)
