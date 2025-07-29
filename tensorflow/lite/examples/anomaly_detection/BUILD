cc_binary(
    name = "anomaly_app",
    srcs = [
        "anomaly_app.cc",
        "anomaly_app.h",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:profile_buffer",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//tensorflow/lite/tools/delegates:tflite_execution_providers",
        ":nlohmann_json",  # Local target for nlohmann/json
    ],
    copts = [
        "-std=c++14",
        "-Inlohmann_json",  # Add include path for nlohmann/json
    ],
)

cc_library(
    name = "nlohmann_json",
    hdrs = glob(["nlohmann_json/*.hpp"]),
    includes = ["nlohmann_json"],
    visibility = ["//visibility:public"],
)
