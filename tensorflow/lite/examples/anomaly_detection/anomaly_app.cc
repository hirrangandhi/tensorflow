/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "anomaly_app.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <cmath>
#include "nlohmann_json/json.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

using json = nlohmann::json;

// DelegateProviders implementation
DelegateProviders::DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
    delegate_list_util_.AppendCmdlineFlags(flags_);
    params_.RemoveParam("help");
    delegate_list_util_.RemoveCmdlineFlag(flags_, "help");
}

bool DelegateProviders::InitFromCmdlineArgs(int* argc, const char** argv) {
    return tflite::Flags::Parse(argc, argv, flags_);
}

void DelegateProviders::MergeSettingsIntoParams(bool allow_fp16, int num_threads) {
    if (params_.HasParam("use_gpu")) {
        params_.Set<bool>("use_gpu", true);
        if (params_.HasParam("gpu_precision_loss_allowed")) {
            params_.Set<bool>("gpu_precision_loss_allowed", allow_fp16);
        }
    }
    if (params_.HasParam("use_nnapi")) {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", allow_fp16);
    }
    if (params_.HasParam("use_xnnpack")) {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<int>("num_threads", num_threads);
    }
}

std::vector<tflite::tools::ProvidedDelegateList::ProvidedDelegate> DelegateProviders::CreateAllDelegates() const {
    return delegate_list_util_.CreateAllRankedDelegates();
}

std::string DelegateProviders::GetHelpMessage(const std::string& cmdline) const {
    return tflite::Flags::Usage(cmdline, flags_);
}

// MinMaxScaler implementation
void MinMaxScaler::fit(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;
    size_t n = data[0].size();
    min.assign(n, std::numeric_limits<float>::max());
    max.assign(n, std::numeric_limits<float>::lowest());
    for (const auto& row : data) {
        for (size_t i = 0; i < n; ++i) {
            if (row[i] < min[i]) min[i] = row[i];
            if (row[i] > max[i]) max[i] = row[i];
        }
    }
}
std::vector<std::vector<float>> MinMaxScaler::transform(const std::vector<std::vector<float>>& data) const {
    std::vector<std::vector<float>> scaled;
    for (const auto& row : data) {
        std::vector<float> s(row.size());
        for (size_t i = 0; i < row.size(); ++i) {
            if (max[i] > min[i]) {
                s[i] = (row[i] - min[i]) / (max[i] - min[i]);
            } else {
                // sklearn: if min==max, output 0 for all
                s[i] = 0.0f;
            }
        }
        scaled.push_back(s);
    }
    return scaled;
}
std::vector<float> MinMaxScaler::inverse_transform(const std::vector<float>& row) const {
    std::vector<float> orig(row.size());
    for (size_t i = 0; i < row.size(); ++i) {
        if (max[i] > min[i]) {
            orig[i] = row[i] * (max[i] - min[i]) + min[i];
        } else {
            // sklearn: if min==max, always return min
            orig[i] = min[i];
        }
    }
    return orig;
}

// TFLite inference for a batch with delegates
std::vector<std::vector<float>> predict_tflite(const std::string& model_path, const std::vector<std::vector<float>>& data, int num_threads, bool allow_fp16, DelegateProviders& delegate_providers) {
    std::vector<std::vector<float>> results;
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return results;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk || !interpreter) {
        std::cerr << "Failed to construct interpreter." << std::endl;
        return results;
    }
    interpreter->SetAllowFp16PrecisionForFp32(allow_fp16);
    interpreter->SetNumThreads(num_threads);
    // Apply delegates
    auto delegates = delegate_providers.CreateAllDelegates();
    for (auto& delegate : delegates) {
        if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
            std::cerr << "Failed to apply delegate." << std::endl;
            return results;
        }
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return results;
    }
    int input_idx = interpreter->inputs()[0];
    int feature_dim = data[0].size();
    int batch_size = data.size();
    // Resize input tensor to {batch_size, feature_dim}
    std::vector<int> new_shape = {static_cast<int>(batch_size), static_cast<int>(feature_dim)};
    if (interpreter->ResizeInputTensor(input_idx, new_shape) != kTfLiteOk) {
        std::cerr << "Failed to resize input tensor for batch." << std::endl;
        return results;
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors after resize for batch." << std::endl;
        return results;
    }
    float* input_tensor = interpreter->typed_tensor<float>(input_idx);
    // Copy all data into input tensor
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            input_tensor[i * feature_dim + j] = data[i][j];
        }
    }
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter for batch." << std::endl;
        return results;
    }
    int output_idx = interpreter->outputs()[0];
    float* output_tensor = interpreter->typed_output_tensor<float>(output_idx);
    // Output tensor shape: [batch_size, feature_dim]
    for (int i = 0; i < batch_size; ++i) {
        std::vector<float> output_row(feature_dim);
        for (int j = 0; j < feature_dim; ++j) {
            output_row[j] = output_tensor[i * feature_dim + j];
        }
        results.push_back(output_row);
    }
    return results;
}

// Percentile calculation
// numpy-like percentile with linear interpolation
float percentile(const std::vector<float>& data, float pct) {
    if (data.empty()) return 0.0f;
    std::vector<float> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    float rank = pct * (sorted.size() - 1);
    size_t low = static_cast<size_t>(std::floor(rank));
    size_t high = static_cast<size_t>(std::ceil(rank));
    if (high >= sorted.size()) high = sorted.size() - 1;
    if (low == high) return sorted[low];
    float weight = rank - low;
    return sorted[low] * (1.0f - weight) + sorted[high] * weight;
}

// Median calculation
std::vector<float> median(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return {};
    size_t n = data[0].size();
    std::vector<float> medians(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> col;
        for (const auto& row : data) col.push_back(row[i]);
        std::sort(col.begin(), col.end());
        size_t mid = col.size() / 2;
        if (col.size() % 2 == 0) {
            // numpy: average of two middle values for even
            medians[i] = (col[mid - 1] + col[mid]) / 2.0f;
        } else {
            medians[i] = col[mid];
        }
    }
    return medians;
}

// Explanation generator
std::string get_dynamic_reason_with_median(const std::vector<float>& abs_errors, const std::vector<float>& scaled_row, const std::vector<std::string>& feature_names, const MinMaxScaler& scaler, const std::vector<float>& medians, int top_n) {
    // Get top_n error indices
    std::vector<size_t> idxs(abs_errors.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) { return abs_errors[a] > abs_errors[b]; });
    std::vector<std::string> reasons;
    std::vector<float> unscaled_row = scaler.inverse_transform(scaled_row);
    for (int i = 0; i < top_n && i < (int)abs_errors.size(); ++i) {
        size_t idx = idxs[i];
        std::string feat = feature_names[idx];
        // Replace underscores with spaces to match Python output
        std::replace(feat.begin(), feat.end(), '_', ' ');
        float actual = unscaled_row[idx];
        float median_val = medians[idx];
        std::string direction = (actual > median_val) ? "High" : "Low";
        std::ostringstream oss;
        oss << direction << " " << feat << " (value: " << std::fixed << std::setprecision(2) << actual << ", normal ≈ " << median_val << ")";
        reasons.push_back(oss.str());
    }
    std::ostringstream out;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i > 0) out << ", ";
        out << reasons[i];
    }
    return out.str();
}

void display_usage(const DelegateProviders& delegate_providers) {
    std::cout << "\n" << delegate_providers.GetHelpMessage("anomaly_app")
              << "\t--cpu_model: Path to CPU anomaly model (required)\n"
              << "\t--memory_model: Path to Memory anomaly model (required)\n"
              << "\t--input: Path to input JSON file (required)\n"
              << "\t--output_csv: Path to output CSV file (required)\n"
              << "\t--threads: Number of threads\n"
              << "\t--allow_fp16: Allow float16 precision\n"
              << "\t--help: Print this help message\n";
}

int main(int argc, char* argv[]) {
    float TOTAL_RAM_MB = 751.42f;
    Settings settings;
    DelegateProviders delegate_providers;
    // --- Parse CLI ---
    std::vector<tflite::Flag> flags = {
        tflite::Flag::CreateFlag("cpu_model", &settings.cpu_model_path, "Path to CPU anomaly model (required)"),
        tflite::Flag::CreateFlag("memory_model", &settings.mem_model_path, "Path to Memory anomaly model (required)"),
        tflite::Flag::CreateFlag("input", &settings.json_path, "Path to input JSON file (required)"),
        tflite::Flag::CreateFlag("output_csv", &settings.output_csv, "Path to output CSV file (required)"),
        tflite::Flag::CreateFlag("threads", &settings.num_threads, "Number of threads"),
        tflite::Flag::CreateFlag("allow_fp16", &settings.allow_fp16, "Allow float16 precision")
    };
    // Add delegate flags
    for (const auto& f : delegate_providers.CreateAllDelegates()) {
        // No-op, flags are already appended in DelegateProviders constructor
    }
    if (!tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flags)) {
        std::cerr << "Failed to parse command-line flags." << std::endl;
        display_usage(delegate_providers);
        return 1;
    }
    // Check required flags
    if (settings.json_path.empty() || settings.cpu_model_path.empty() || settings.mem_model_path.empty() || settings.output_csv.empty()) {
        std::cerr << "Error: All of --input, --cpu_model, --memory_model, and --output_csv must be provided." << std::endl;
        display_usage(delegate_providers);
        return 1;
    }
    delegate_providers.MergeSettingsIntoParams(settings.allow_fp16, settings.num_threads);

    // --- Load JSON Data ---
    std::ifstream file(settings.json_path);
    if (!file.is_open()) {
        std::cerr << "Unable to open JSON file: " << settings.json_path << std::endl;
        return 1;
    }
    json j;
    file >> j;

    // --- Extract Records ---
    std::vector<Record> records;
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string timestamp = it.key();
        // Strip whitespace from timestamp to match Python
        timestamp.erase(0, timestamp.find_first_not_of(" \t\n\r"));
        timestamp.erase(timestamp.find_last_not_of(" \t\n\r") + 1);
        const auto& sys = it.value().contains("system_params") ? it.value()["system_params"] : json{};
        Record rec;
        rec.timestamp = timestamp;
        rec.cpu_usage = std::stof(sys.value("cpu_usage", "0.0"));
        rec.loadavg_15min = std::stof(sys.value("loadavg_15min", "0.0"));
        rec.free_memory_mb = std::stof(sys.value("free_memory_mb", "0.0"));
        rec.slab_memory_mb = std::stof(sys.value("slab_memory_mb", "0.0"));
        rec.available_mem_mb = std::stof(sys.value("available_mem_mb", "0.0"));
        rec.cached_mem_mb = std::stof(sys.value("cached_mem_mb", "0.0"));
        rec.slab_unreclaim_mb = std::stof(sys.value("slab_unreclaim_mb", "0.0"));
        rec.used_memory_mb = TOTAL_RAM_MB - rec.free_memory_mb;
        records.push_back(rec);
    }
    // Sort records by timestamp to match Python output order
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
        return a.timestamp < b.timestamp;
    });

    // --- Feature Preparation ---
    std::vector<std::string> cpu_feature_names = {"cpu_usage", "loadavg_15min"};
    std::vector<std::string> mem_feature_names = {"used_memory_mb", "slab_memory_mb", "available_mem_mb", "cached_mem_mb", "slab_unreclaim_mb"};
    std::vector<std::vector<float>> cpu_features, mem_features;
    for (const auto& rec : records) {
        cpu_features.push_back({rec.cpu_usage, rec.loadavg_15min});
        mem_features.push_back({rec.used_memory_mb, rec.slab_memory_mb, rec.available_mem_mb, rec.cached_mem_mb, rec.slab_unreclaim_mb});
    }
    MinMaxScaler cpu_scaler, mem_scaler;
    cpu_scaler.fit(cpu_features);
    mem_scaler.fit(mem_features);
    auto cpu_input = cpu_scaler.transform(cpu_features);
    auto mem_input = mem_scaler.transform(mem_features);

    // --- Run TFLite Inference ---
    auto cpu_recon = predict_tflite(settings.cpu_model_path, cpu_input, settings.num_threads, settings.allow_fp16, delegate_providers);
    auto mem_recon = predict_tflite(settings.mem_model_path, mem_input, settings.num_threads, settings.allow_fp16, delegate_providers);

    // --- Reconstruction Errors ---
    std::vector<std::vector<float>> cpu_abs_errors, mem_abs_errors;
    std::vector<float> cpu_mse, mem_mse;
    for (size_t i = 0; i < cpu_input.size(); ++i) {
        std::vector<float> cpu_err(cpu_input[i].size()), mem_err(mem_input[i].size());
        float cpu_sum = 0.0f, mem_sum = 0.0f;
        for (size_t j = 0; j < cpu_input[i].size(); ++j) {
            cpu_err[j] = std::abs(cpu_input[i][j] - cpu_recon[i][j]);
            cpu_sum += (cpu_input[i][j] - cpu_recon[i][j]) * (cpu_input[i][j] - cpu_recon[i][j]);
        }
        for (size_t j = 0; j < mem_input[i].size(); ++j) {
            mem_err[j] = std::abs(mem_input[i][j] - mem_recon[i][j]);
            mem_sum += (mem_input[i][j] - mem_recon[i][j]) * (mem_input[i][j] - mem_recon[i][j]);
        }
        cpu_abs_errors.push_back(cpu_err);
        mem_abs_errors.push_back(mem_err);
        cpu_mse.push_back(cpu_sum / cpu_input[i].size());
        mem_mse.push_back(mem_sum / mem_input[i].size());
    }

    // --- Percentile Thresholds ---
    float cpu_threshold = percentile(cpu_mse, 0.95f);
    float mem_threshold = percentile(mem_mse, 0.95f);

    // --- Median Calculation ---
    auto cpu_medians = median(cpu_features);
    auto mem_medians = median(mem_features);

    // --- Explanation Generation ---
    std::vector<std::string> cpu_reason, mem_reason;
    for (size_t i = 0; i < records.size(); ++i) {
        cpu_reason.push_back(get_dynamic_reason_with_median(cpu_abs_errors[i], cpu_input[i], cpu_feature_names, cpu_scaler, cpu_medians));
        mem_reason.push_back(get_dynamic_reason_with_median(mem_abs_errors[i], mem_input[i], mem_feature_names, mem_scaler, mem_medians));
    }
    std::vector<std::string> cpu_explanation, mem_explanation;
    for (size_t i = 0; i < records.size(); ++i) {
        if (cpu_mse[i] > cpu_threshold)
            cpu_explanation.push_back("CPU anomaly. Cause: " + cpu_reason[i]);
        else
            cpu_explanation.push_back("");
        if (mem_mse[i] > mem_threshold)
            mem_explanation.push_back("Memory anomaly. Cause: " + mem_reason[i]);
        else
            mem_explanation.push_back("");
    }

    // --- Output CSV ---
    std::ofstream out_csv(settings.output_csv);
    out_csv << "timestamp,cpu_anomaly,cpu_explanation,memory_anomaly,memory_explanation\n";
    for (size_t i = 0; i < records.size(); ++i) {
        // Remove leading/trailing spaces from timestamp and fields
        out_csv << records[i].timestamp << ","
                << (cpu_mse[i] > cpu_threshold ? "True" : "False") << ","
                << (cpu_explanation[i].empty() ? "" : ('"' + cpu_explanation[i] + '"')) << ","
                << (mem_mse[i] > mem_threshold ? "True" : "False") << ","
                << (mem_explanation[i].empty() ? "" : ('"' + mem_explanation[i] + '"')) << "\n";
    }
    out_csv.close();
    std::cout << "Inference complete. Output written to: " << settings.output_csv << std::endl;
    return 0;
}
