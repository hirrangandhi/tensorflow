# Anomaly Detection Application

This application uses TensorFlow Lite autoencoder models to detect anomalies in system metrics. It reads input data from a JSON file, runs inference using CPU and memory models, and outputs anomaly results to a CSV file with human-readable explanations.

## Features
- Parses system metrics from JSON input (e.g., cpu_usage, memory, load averages)
- Runs inference using TFLite autoencoder models for CPU and memory
- Computes reconstruction error and flags anomalies using percentile thresholds
- Generates human-readable explanations based on feature medians
- Outputs timestamped results to CSV
- Supports hardware delegates (GPU, NNAPI, XNNPACK)
- All major parameters are configurable via command-line flags

## Requirements
- TensorFlow Lite (C++ API)
- nlohmann/json (for JSON parsing)
- C++14 or newer
- Bazel or CMake build system

## Build Instructions

### Bazel
1. Ensure Bazel is installed and configured.
2. Add nlohmann/json to your WORKSPACE as an external dependency (see project docs).
3. Run:
   ```sh
   bazel build //anomaly_detection:anomaly_app
   ```

### CMake
1. Update `CMakeLists.txt` with correct include/library paths for TensorFlow Lite and nlohmann/json.
2. Run:
   ```sh
   mkdir build && cd build
   cmake ..
   make
   ```

## Usage

```
./anomaly_app --cpu_model=<cpu_model.tflite> --memory_model=<memory_model.tflite> --input=<input_json> --output_csv=<output_csv> [--threads=<num_threads>] [--allow_fp16=<0|1>] [delegate flags]
```

## Example
./anomaly_app --input=July9_10.json --output_csv=test.csv --cpu_model=cpu_anomaly_model.tflite --memory_model=memory_anomaly_model.tflite

- `--cpu_model`: Path to the CPU anomaly TFLite model file
- `--memory_model`: Path to the memory anomaly TFLite model file
- `--input`: Path to the input JSON file (e.g., July9_10.json)
- `--output_csv`: Path to output CSV file (required)
- `--threads`: Number of threads for inference (default: 1)
- `--allow_fp16`: Allow float16 precision (default: 0)
- Delegate flags: See help output for hardware delegate options (GPU, NNAPI, XNNPACK)

## Input Format
The input JSON should contain timestamped entries, each with a `system_params` object. Example:

```json
{
  "2025-07-09-11:35:40.325": {
    "system_params": {
      "cpu_usage": "5.000",
      "free_memory_mb": "22.020",
      "loadavg_15min": "0.100",
      "slab_memory_mb": "1.000",
      "available_mem_mb": "20.000",
      "cached_mem_mb": "2.000",
      "slab_unreclaim_mb": "0.500"
    }
  },
  ...
}
```

## Output
The output CSV will contain:
- `timestamp`: Timestamp from input
- `cpu_anomaly`: True/False if CPU anomaly detected
- `cpu_explanation`: Human-readable explanation (if anomaly)
- `memory_anomaly`: True/False if memory anomaly detected
- `memory_explanation`: Human-readable explanation (if anomaly)

## License
Apache 2.0
