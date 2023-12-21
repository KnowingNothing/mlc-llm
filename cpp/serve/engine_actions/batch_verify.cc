/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_spec_decode.cc
 */

#include <cmath>
#include <exception>

#include "../../random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs verification for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 */
class BatchVerifyActionObj : public EngineActionObj {
 public:
  explicit BatchVerifyActionObj(Array<Model> models, Sampler sampler, KVCacheConfig kv_cache_config,
                                int max_single_sequence_length,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        sampler_(std::move(sampler)),
        kv_cache_config_(std::move(kv_cache_config)),
        max_single_sequence_length_(max_single_sequence_length),
        trace_recorder_(std::move(trace_recorder)),
        rng_(RandomGenerator::GetInstance()) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run spec decode when there are two models (llm+ssm) and >=1 running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    // the id of different models
    const int draft_model_id = 1;
    const int verify_model_id = 0;

    auto [requests, rstates, draft_lengths, total_draft_length] = GetDraftsToVerify(estate);
    ICHECK_EQ(requests.size(), rstates.size());
    ICHECK_EQ(requests.size(), draft_lengths.size());
    if (requests.empty()) {
      return {};
    }

    int num_requests = requests.size();
    Array<String> request_ids = requests.Map([](const Request& request) { return request->id; });
    auto tstart = std::chrono::high_resolution_clock::now();

    // - Get embedding and run verify.
    Array<NDArray> embeddings;
    std::vector<int64_t> request_internal_ids;
    std::vector<int> verify_lengths;
    int total_verify_length = total_draft_length;
    embeddings.reserve(num_requests);
    request_internal_ids.reserve(num_requests);
    verify_lengths.resize(num_requests);
    for (int i = 0; i < num_requests; ++i) {
      RequestModelState verify_mstate = rstates[i]->mstates[verify_model_id];
      RequestModelState draft_mstate = rstates[i]->mstates[draft_model_id];
      ICHECK_EQ(draft_mstate->GetDraftLength(), draft_lengths[i]);
      // one committed token and all the draft tokens
      verify_lengths[i] = draft_lengths[i];
      ICHECK(draft_mstate->draft_output_tokens.size() ==
             draft_mstate->draft_output_token_prob.size());
      ICHECK(draft_mstate->draft_output_tokens.size() ==
             draft_mstate->draft_output_prob_dist.size());
      request_internal_ids.push_back(verify_mstate->internal_id);
      RECORD_EVENT(trace_recorder_, requests[i]->id, "verify start embedding");
      std::vector<int> draft_tokens = {draft_mstate->committed_tokens.back()};
      draft_tokens.insert(draft_tokens.end(), draft_mstate->draft_output_tokens.begin(),
                          draft_mstate->draft_output_tokens.end());
      if (draft_tokens.size() > 1U) {
        draft_tokens.pop_back();
      } else {
        verify_lengths[i] += 1;
        total_verify_length += 1;
      }
      embeddings.push_back(models_[verify_model_id]->TokenEmbed(
          {IntTuple{draft_tokens.begin(), draft_tokens.end()}}));
      RECORD_EVENT(trace_recorder_, requests[i]->id, "verify finish embedding");
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start verify");
    NDArray logits =
        models_[verify_model_id]->BatchVerify(embeddings, request_internal_ids, verify_lengths);
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify");
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], total_verify_length);

    std::vector<int> cum_verify_lengths = {0};
    for (int i = 0; i < num_requests; ++i) {
      cum_verify_lengths.push_back(cum_verify_lengths.back() + verify_lengths[i]);
    }

    // - Verify tokens.
    logits = logits.CreateView({total_verify_length, 1, logits->shape[2]}, logits->dtype);
    std::vector<RequestModelState> mstates;
    std::vector<GenerationConfig> generation_configs;
    int cur_req_id = 0;
    for (int i = 0; i < total_verify_length; ++i) {
      if (i >= cum_verify_lengths[cur_req_id + 1]) {
        cur_req_id += 1;
      }
      Request request = requests[cur_req_id];
      mstates.push_back(estate->GetRequestState(request)->mstates[verify_model_id]);
      generation_configs.push_back(request->generation_cfg);
    }
    // for (int i = 0; i < num_requests; ++i) {
    //   Request request = requests[i];
    //   mstates.push_back(estate->GetRequestState(request)->mstates[verify_model_id]);
    //   generation_configs.push_back(request->generation_cfg);
    // }
    logits = sampler_->ComputeProb(logits, models_[verify_model_id], mstates, generation_configs);

    ICHECK(logits.IsContiguous());
    ICHECK(logits.DataType() == DataType::Float(32));

    if (logits->device.device_type != kDLCPU) {
      logits = logits.CopyTo(DLDevice{kDLCPU, 0});
    }

    ICHECK(logits->device.device_type == kDLCPU);

    int64_t ndata = logits->shape[logits->ndim - 1];
    const float* __restrict p_probs =
        static_cast<float*>(__builtin_assume_aligned(logits->data, 4));

    // this part is parallelizable
    // TODO: parallelize this part
    for (int i = 1; i <= num_requests; ++i) {
      RequestModelState draft_mstate = rstates[i - 1]->mstates[draft_model_id];
      int verify_start = cum_verify_lengths[i - 1];
      int verify_end = cum_verify_lengths[i];
      int accept_length = VerifyAndAcceptTokens(
          estate, requests[i - 1], p_probs, verify_start, verify_end, ndata,
          draft_mstate->draft_output_tokens, draft_mstate->draft_output_token_prob,
          draft_mstate->draft_output_prob_dist);
      estate->stats.total_accepted_length += accept_length;
      // minus one because the last draft token has no kv cache entry
      int rollback_length = verify_end - verify_start - accept_length - 1;
      // in case of all accepted
      rollback_length = std::max(0, rollback_length);
      // rollback kv cache
      if (rollback_length > 0) {
        models_[verify_model_id]->PopNFromKVCache(
            rstates[i - 1]->mstates[verify_model_id]->internal_id, rollback_length);
        models_[draft_model_id]->PopNFromKVCache(
            rstates[i - 1]->mstates[draft_model_id]->internal_id, rollback_length);
      }
    }

    // clear the draft model states
    for (int i = 0; i < num_requests; ++i) {
      rstates[i]->mstates[draft_model_id]->draft_output_tokens.clear();
      rstates[i]->mstates[draft_model_id]->draft_output_token_prob.clear();
      rstates[i]->mstates[draft_model_id]->draft_output_prob_dist.clear();
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return requests;
  }

 private:
  bool CanDecode(int num_requests) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_requests <= num_available_pages;
  }

  /*! \brief Check if the drafts can be verified under conditions. */
  bool CanVerify(EngineState estate, int num_verify_req, int total_draft_length,
                 int num_required_pages, int num_available_pages) {
    int num_running_requests = estate->running_queue.size();
    ICHECK_LE(num_running_requests, kv_cache_config_->max_num_sequence);

    // No exceeding of the maximum allowed requests that can
    // run simultaneously.
    if (num_running_requests + num_verify_req > kv_cache_config_->max_num_sequence) {
      return false;
    }

    // NOTE: The conditions are heuristic and can be revised.
    // Cond 1: total input length <= max allowed single sequence length.
    // Cond 2: at least one verify can be performed.
    // Cond 3: number of total tokens does not exceed the limit
    int new_batch_size = num_running_requests + num_verify_req;
    return total_draft_length <= max_single_sequence_length_ &&
           num_required_pages <= num_available_pages &&
           estate->stats.current_total_seq_len + total_draft_length <=
               kv_cache_config_->max_total_sequence_length;
  }

  /*!
   * \brief Decide whether to run verify for the draft of each request.
   * \param estate The engine state.
   * \return The drafts to verify, together with their respective
   * state and input length.
   */
  std::tuple<Array<Request>, Array<RequestState>, std::vector<int>, int> GetDraftsToVerify(
      EngineState estate) {
    const int verify_model_id = 0;
    const int draft_model_id = 1;

    // - Try to verify pending requests.
    std::vector<Request> verify_requests;
    std::vector<RequestState> rstates;
    std::vector<int> draft_lengths;
    int total_draft_length = 0;
    int total_required_pages = 0;
    int num_available_pages = models_[verify_model_id]->GetNumAvailablePages();

    int req_id = 1;
    for (; req_id <= static_cast<int>(estate->running_queue.size()); ++req_id) {
      Request request = estate->running_queue[req_id - 1];
      RequestState rstate = estate->GetRequestState(request);
      int draft_length = rstate->mstates[draft_model_id]->GetDraftLength();
      int num_require_pages =
          (draft_length + kv_cache_config_->page_size - 1) / kv_cache_config_->page_size;
      total_draft_length += draft_length;
      total_required_pages += num_require_pages;
      if (CanVerify(estate, req_id, total_draft_length, total_required_pages,
                    num_available_pages)) {
        verify_requests.push_back(request);
        rstates.push_back(rstate);
        draft_lengths.push_back(draft_length);
      } else {
        total_draft_length -= draft_length;
        total_required_pages -= num_require_pages;
        break;
      }
    }
    // preempt all the remaining requests
    // TODO: can we remove requests that are in the middle positions?
    while (req_id <= static_cast<int>(estate->running_queue.size())) {
      PreemptLastRunningRequest(estate);
      req_id += 1;
    }

    return {verify_requests, rstates, draft_lengths, total_draft_length};
  }

  /*!
   * \brief Preempt the last running requests from `running_queue`,
   * moving it from running request set to the foremost of waiting
   * request queue.
   */
  void PreemptLastRunningRequest(EngineState estate) {
    Request request = estate->running_queue.back();

    // Remove from models.
    // - Clear model speculation draft.
    // - Update `inputs` for future prefill.
    RequestState rstate = estate->GetRequestState(request);
    RECORD_EVENT(trace_recorder_, rstate->request->id, "preempt");
    estate->stats.current_total_seq_len -=
        request->input_total_length + rstate->mstates[0]->committed_tokens.size() - 1;
    for (RequestModelState mstate : rstate->mstates) {
      mstate->draft_output_tokens.clear();
      mstate->draft_output_token_prob.clear();
      mstate->draft_output_prob_dist.clear();
      ICHECK(mstate->inputs.empty());
      ICHECK(!mstate->committed_tokens.empty());

      Array<Data> inputs = request->inputs;
      if (const auto* token_input = inputs.back().as<TokenDataNode>()) {
        // Merge the TokenData so that a single time TokenEmbed is needed.
        std::vector<int> token_ids{token_input->token_ids->data,
                                   token_input->token_ids->data + token_input->token_ids.size()};
        token_ids.insert(token_ids.end(), mstate->committed_tokens.begin(),
                         mstate->committed_tokens.end());
        inputs.Set(inputs.size() - 1, TokenData(token_ids));
      } else {
        inputs.push_back(TokenData(mstate->committed_tokens));
      }
      mstate->inputs = std::move(inputs);
    }
    RemoveRequestFromModel(estate, rstate->mstates[0]->internal_id, models_);

    // Move from running queue to the front of waiting queue.
    estate->running_queue.erase(estate->running_queue.end() - 1);
    estate->waiting_queue.insert(estate->waiting_queue.begin(), request);
  }

  void AcceptToken(EngineState estate, RequestState rstate, int token) {
    // the id of different models
    const int draft_model_id = 1;
    const int verify_model_id = 0;
    RequestModelState verify_mstate = rstate->mstates[verify_model_id];
    RequestModelState draft_mstate = rstate->mstates[draft_model_id];
    verify_mstate->committed_tokens.push_back(token);
    draft_mstate->committed_tokens.push_back(token);
    // update total token length
    estate->stats.current_total_seq_len += 1;
  }

  int VerifyAndAcceptTokens(EngineState estate, Request request, const float* __restrict p_probs,
                            int verify_start, int verify_end, int64_t ndata,
                            std::vector<int>& draft_output_tokens,
                            std::vector<float>& draft_output_token_prob,
                            std::vector<NDArray>& draft_output_prob_dist) {
    int accept_length = 0;
    // the id of different models
    const int draft_model_id = 1;
    const int verify_model_id = 0;
    RequestState rstate = estate->GetRequestState(request);
    RequestModelState verify_mstate = rstate->mstates[verify_model_id];
    RequestModelState draft_mstate = rstate->mstates[draft_model_id];
    if (!draft_output_tokens.size()) {
      // no draft tokens
      // sample a new token
      NDArray new_prob_dist = NDArray::Empty({1, ndata}, {kDLFloat, 32, 1}, {kDLCPU, 0});
      new_prob_dist.CopyFromBytes(&p_probs[0], ndata * sizeof(float));
      std::vector<int32_t> next_tokens =
          sampler_->SampleTokenFromProbs(new_prob_dist, {verify_mstate}, {request->generation_cfg});
      ICHECK(next_tokens.size() == 1U);
      AcceptToken(estate, rstate, next_tokens[0]);
      return 0;
    }

    ICHECK(verify_end - verify_start == (int)draft_output_tokens.size());
    for (int i = verify_start; i < verify_end; ++i) {
      int cur_token_idx = i - verify_start;
      int cur_token = draft_output_tokens[cur_token_idx];
      float p_value = p_probs[i * ndata + cur_token];
      float q_value = draft_output_token_prob[cur_token_idx];
      const float eps = 1e-9;
      if (p_value >= q_value) {
        AcceptToken(estate, rstate, cur_token);
      } else {
        float r = rng_.GetRandomNumber();
        if (r < p_value / (q_value + eps)) {
          AcceptToken(estate, rstate, cur_token);
        } else {
          // nomarlize a new probability distribution
          double sum_v = 0.0;
          std::vector<float> new_prob;
          new_prob.resize(ndata);

          NDArray q_dist = draft_output_prob_dist[cur_token_idx];
          if (q_dist->device.device_type != kDLCPU) {
            q_dist = q_dist.CopyTo(DLDevice{kDLCPU, 0});
          }

          ICHECK(q_dist->device.device_type == kDLCPU);

          ICHECK(q_dist->ndim == 1);
          ICHECK(ndata == q_dist->shape[q_dist->ndim - 1]);
          const float* __restrict p_qdist =
              static_cast<float*>(__builtin_assume_aligned(q_dist->data, 4));

          // this part is parallelizable
          // TODO: parallelize this part
          for (int j = 0; j < ndata; ++j) {
            new_prob[j] = p_probs[i * ndata + j] - p_qdist[j];
            new_prob[j] = new_prob[j] < 0 ? 0 : new_prob[j];
            // new_prob[j] = std::exp(new_prob[j]);
            sum_v += new_prob[j];
          }
          for (int j = 0; j < ndata; ++j) {
            new_prob[j] /= sum_v;
          }

          // sample a new token from the new distribution
          NDArray new_prob_dist = NDArray::Empty({1, ndata}, {kDLFloat, 32, 1}, {kDLCPU, 0});
          new_prob_dist.CopyFromBytes(new_prob.data(), ndata * sizeof(float));
          std::vector<int32_t> next_tokens = sampler_->SampleTokenFromProbs(
              new_prob_dist, {verify_mstate}, {request->generation_cfg});
          ICHECK(next_tokens.size() == 1U);
          AcceptToken(estate, rstate, next_tokens[0]);
          break;
        }
      }
      accept_length += 1;
    }

    return accept_length;
  }
  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The kv cache config. */
  KVCacheConfig kv_cache_config_;
  /*! \brief The maximum allowed length of a single sequence. */
  int max_single_sequence_length_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Random number generator. */
  RandomGenerator& rng_;
};

EngineAction EngineAction::BatchVerify(Array<Model> models, Sampler sampler,
                                       KVCacheConfig kv_cache_config,
                                       int max_single_sequence_length,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchVerifyActionObj>(
      std::move(models), std::move(sampler), std::move(kv_cache_config),
      std::move(max_single_sequence_length), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
