/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_spec_decode.cc
 */

#include "../config.h"
#include "../model.h"
#include "../sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs draft proposal for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 */
class BatchDraftActionObj : public EngineActionObj {
 public:
  explicit BatchDraftActionObj(Array<Model> models, Sampler sampler,
                               Optional<EventTraceRecorder> trace_recorder, int draft_length = 4)
      : models_(std::move(models)),
        sampler_(std::move(sampler)),
        trace_recorder_(std::move(trace_recorder)),
        draft_length_(draft_length) {}

  Array<Request> Step(EngineState estate) final {
    // - Only run spec decode when there are two models (llm+ssm) and >=1 running requests.
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt requests when decode cannot apply.
    while (!CanDecode(estate->running_queue.size())) {
      PreemptLastRunningRequest(estate);
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running requests at a time.
    int num_requests = estate->running_queue.size();

    // The first model doesn't get involved in draft proposal.
    for (int model_id = 1; model_id < models_.size(); ++model_id) {
      // draft_length_ rounds of draft proposal.
      for (int draft_id = 0; draft_id < draft_length_; ++draft_id) {
        // Collect
        // - the last committed token,
        // - the request states,
        // - the sampling parameters,
        // of each request.
        std::vector<int> input_tokens;
        Array<String> request_ids;
        std::vector<int64_t> request_internal_ids;
        Array<RequestModelState> mstates;
        Array<GenerationConfig> generation_cfg;
        input_tokens.reserve(num_requests);
        request_ids.reserve(num_requests);
        request_internal_ids.reserve(num_requests);
        mstates.reserve(num_requests);
        generation_cfg.reserve(num_requests);
        for (Request request : estate->running_queue) {
          RequestState rstate = estate->GetRequestState(request);
          if (draft_id == 0) {
            // The first draft proposal uses the last committed token.
            input_tokens.push_back(rstate->mstates[model_id]->committed_tokens.back());
          } else {
            input_tokens.push_back(rstate->mstates[model_id]->draft_output_tokens.back());
          }
          request_ids.push_back(request->id);
          request_internal_ids.push_back(rstate->mstates[0]->internal_id);
          mstates.push_back(rstate->mstates[model_id]);
          generation_cfg.push_back(request->generation_cfg);
        }

        // - Compute embeddings.
        RECORD_EVENT(trace_recorder_, request_ids,
                     "start propsal embedding for model_id " + std::to_string(model_id) +
                         " draft_id " + std::to_string(draft_id));
        NDArray embeddings =
            models_[model_id]->TokenEmbed({IntTuple{input_tokens.begin(), input_tokens.end()}});
        RECORD_EVENT(trace_recorder_, request_ids,
                     "finish proposal embedding for model_id " + std::to_string(model_id) +
                         " draft_id " + std::to_string(draft_id));
        ICHECK_EQ(embeddings->ndim, 3);
        ICHECK_EQ(embeddings->shape[0], 1);
        ICHECK_EQ(embeddings->shape[1], num_requests);
        embeddings =
            embeddings.CreateView({num_requests, 1, embeddings->shape[2]}, embeddings->dtype);

        // - Invoke model decode.
        RECORD_EVENT(trace_recorder_, request_ids,
                     "start proposal decode for model_id " + std::to_string(model_id) +
                         " draft_id " + std::to_string(draft_id));
        NDArray logits = models_[model_id]->BatchDecode(embeddings, request_internal_ids);
        RECORD_EVENT(trace_recorder_, request_ids,
                     "finish proposal decode for model_id " + std::to_string(model_id) +
                         " draft_id " + std::to_string(draft_id));
        ICHECK_EQ(logits->ndim, 3);
        ICHECK_EQ(logits->shape[0], embeddings->shape[0]);
        ICHECK_EQ(logits->shape[1], 1);

        // - Sample tokens.
        RECORD_EVENT(trace_recorder_, request_ids,
                     "start proposal sampling for model_id " + std::to_string(model_id) +
                         " draft_id " + std::to_string(draft_id));
        std::vector<NDArray> prob_dist;
        std::vector<float> token_probs;
        std::vector<int32_t> next_tokens = sampler_->SampleTokens(
            logits, models_[model_id], mstates, generation_cfg, &prob_dist, &token_probs);
        RECORD_EVENT(trace_recorder_, request_ids,
                     "finish proposal sampling for model_id " + std::to_string(model_id) +
                         " draft_id " + std::to_string(draft_id));
        ICHECK_EQ(next_tokens.size(), num_requests);

        // - Update the draft tokens, prob dist, token probs of states.
        for (int i = 0; i < num_requests; ++i) {
          mstates[i]->draft_output_tokens.push_back(next_tokens[i]);
          mstates[i]->draft_output_prob_dist.push_back(prob_dist[i]);
          mstates[i]->draft_output_token_prob.push_back(token_probs[i]);
          estate->stats.total_draft_length += 1;
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return {};
  }

 private:
  /*! \brief Check if the input requests can be decoded under conditions. */
  bool CanDecode(int num_requests) {
    // The first model is not involved in draft proposal.
    for (int model_id = 1; model_id < models_.size(); ++model_id) {
      // Check if the model has enough available pages.
      int num_available_pages = models_[model_id]->GetNumAvailablePages();
      if (num_requests > num_available_pages) {
        return false;
      }
    }
    return true;
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

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  /*! \brief Draft proposal length */
  int draft_length_;
};

EngineAction EngineAction::BatchDraft(Array<Model> models, Sampler sampler,
                                      Optional<EventTraceRecorder> trace_recorder,
                                      int draft_length) {
  return EngineAction(make_object<BatchDraftActionObj>(std::move(models), std::move(sampler),
                                                       std::move(trace_recorder), draft_length));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
