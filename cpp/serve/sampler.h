/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.h
 * \brief The header for runtime module of sampler functions.
 */

#ifndef MLC_LLM_SERVE_SAMPLER_H_
#define MLC_LLM_SERVE_SAMPLER_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "../base.h"
#include "event_trace_recorder.h"
#include "model.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief The base class of runtime sampler.
 * Its main function is `SampleTokens`, which takes a batch of
 * logits and corresponding configuration, and sample one token
 * for each instance of the batch.
 */
class SamplerObj : public Object {
 public:
  /*!
   * \brief Compute the probability distribution of the input logits.
   * \param logits_on_device The logits to compute probability distribution for.
   * \param model The LLM model which contains the softmax
   * function on device that might be used to compute probability distribution.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the given model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \return The probability distribution of the input logits.
   */
  virtual NDArray ComputeProb(NDArray logits_on_device, Model model,
                              Array<RequestModelState> request_mstates,
                              Array<GenerationConfig> generation_cfg) = 0;

  /*!
   * \brief Sample tokens from a batch of input probability distributions.
   * \param probs The input batch of probability distributions.
   * \param generation_cfg The generation config.
   * \return The sampled tokens, one for each instance of the batch.
   */
  virtual std::vector<int32_t> SampleTokenFromProbs(
      NDArray probs, Array<RequestModelState> request_mstates,
      Array<GenerationConfig> generation_cfg, std::vector<NDArray>* output_prob_dist = nullptr,
      std::vector<float>* output_token_probs = nullptr) = 0;

  /*!
   * \brief Sample tokens from the input batch of logits.
   * \param logits_on_device The logits to sample tokens from.
   * \param model The LLM model which contains the softmax
   * function on device that might be used to compute probability distribution.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the given model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \param output_prob_dist The output probability distribution
   * \param output_token_probs The output token probabilities
   * \return The sampled tokens, one for each request in the batch.
   */
  virtual std::vector<int32_t> SampleTokens(NDArray logits_on_device, Model model,
                                            Array<RequestModelState> request_mstates,
                                            Array<GenerationConfig> generation_cfg,
                                            std::vector<NDArray>* output_prob_dist = nullptr,
                                            std::vector<float>* output_token_probs = nullptr) = 0;

  static constexpr const char* _type_key = "mlc.serve.Sampler";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(SamplerObj, Object);
};

class Sampler : public ObjectRef {
 public:
  /*!
   * \brief Create the runtime sampler module.
   * \param sampler_kind The sampler name denoting which sampler to create.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created runtime module.
   */
  TVM_DLL static Sampler Create(std::string sampler_kind,
                                Optional<EventTraceRecorder> trace_recorder);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Sampler, ObjectRef, SamplerObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_SAMPLER_H_
