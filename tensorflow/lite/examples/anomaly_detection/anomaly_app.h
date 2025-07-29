#ifndef ANOMALY_APP_H
#define ANOMALY_APP_H

#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <sstream>
#include "nlohmann_json/json.hpp"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

struct Record {
    std::string timestamp;
    float cpu_usage;
    float loadavg_15min;
    float free_memory_mb;
    float slab_memory_mb;
    float available_mem_mb;
    float cached_mem_mb;
    float slab_unreclaim_mb;
    float used_memory_mb;
};

class MinMaxScaler {
public:
    std::vector<float> min, max;
    void fit(const std::vector<std::vector<float>>& data);
    std::vector<std::vector<float>> transform(const std::vector<std::vector<float>>& data) const;
    std::vector<float> inverse_transform(const std::vector<float>& row) const;
};

class DelegateProviders {
 public:
  DelegateProviders();
  bool InitFromCmdlineArgs(int* argc, const char** argv);
  void MergeSettingsIntoParams(bool allow_fp16, int num_threads);
  std::vector<tflite::tools::ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates() const;
  std::string GetHelpMessage(const std::string& cmdline) const;
 private:
  tflite::tools::ToolParams params_;
  tflite::tools::ProvidedDelegateList delegate_list_util_;
  std::vector<tflite::Flag> flags_;
};

std::vector<std::vector<float>> predict_tflite(const std::string& model_path, const std::vector<std::vector<float>>& data, int num_threads, bool allow_fp16, DelegateProviders& delegate_providers);
float percentile(const std::vector<float>& data, float pct);
std::vector<float> median(const std::vector<std::vector<float>>& data);
std::string get_dynamic_reason_with_median(const std::vector<float>& abs_errors, const std::vector<float>& scaled_row, const std::vector<std::string>& feature_names, const MinMaxScaler& scaler, const std::vector<float>& medians, int top_n = 2);

struct Settings {
    std::string cpu_model_path;
    std::string mem_model_path;
    std::string json_path;
    std::string output_csv;
    int num_threads = 1;
    bool allow_fp16 = false;
    // Delegate-related flags
    bool use_gpu = false;
    bool use_nnapi = false;
    bool use_xnnpack = false;
    // You can add more delegate flags as needed
};

void display_usage(const DelegateProviders& delegate_providers);

#endif // ANOMALY_APP_H
