/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_DOT_LIBRARY_REWRITER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_DOT_LIBRARY_REWRITER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "tsl/platform/protobuf.h"

namespace xla::cpu {

struct DotLibraryRewriterOptions {
  bool use_onednn = false;
  bool use_xnnpack = false;
  const tsl::protobuf::RepeatedField<int>* onednn_fusion_types = nullptr;
  const tsl::protobuf::RepeatedField<int>* xnn_fusion_types = nullptr;
};

// Rewrites suitable Dot operations into library fusions.
class DotLibraryRewriter : public HloModulePass {
 public:
  explicit DotLibraryRewriter(
      const TargetMachineFeatures* target_machine_features,
      const DotLibraryRewriterOptions& options)
      : target_machine_features_(target_machine_features),
        options_(std::move(options)) {}
  ~DotLibraryRewriter() override = default;

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::string_view name() const override { return "dot-library-rewriter"; }

 private:
  const TargetMachineFeatures* target_machine_features_;
  const DotLibraryRewriterOptions options_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_DOT_LIBRARY_REWRITER_H_
