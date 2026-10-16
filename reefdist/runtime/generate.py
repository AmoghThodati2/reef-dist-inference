"""Generation loop run by every simulated NPU: prefill (optionally chunked), decode, spec_verify.

Passes, in order:
  prefill      ceil(prompt_len / prefill_chunk) passes over the prompt
  decode       gen_tokens - 1 single-token passes (token 1 comes from the prefill logits)
  spec_verify  one pass over draft_tokens draft tokens against the prompt's KV cache

Sampling (greedy argmax) happens on the NPUs that hold logits. If those are not the
NPUs that embed tokens (pipeline parallel), the sampled tokens are sent along
``plan["token_routes"]`` at the start of the next pass (counted under layer "sample").
"""
import torch

from ..strategies.base import StepInput


def run_workload(strategy, npu_id, kernels, comm, tracer, job):
    plan, wl = strategy.plan_data, job["workload"]
    B = wl["batch_size"]
    prompt = job["prompt_ids"]
    P = prompt.shape[1]
    is_input = npu_id in plan["input_npus"]
    has_logits = npu_id in plan["logits_npus"]
    send_to = [d for s, d in plan["token_routes"] if s == npu_id]
    recv_from = [s for s, d in plan["token_routes"] if d == npu_id]
    cache = kernels.create_kv_cache()
    passes = []

    def step(phase, ids, T, all_logits=False, route=False):
        tracer.begin_pass(len(passes))
        passes.append({"pass": len(passes), "phase": phase, "tokens": T, "context": cache.seq_len + T})
        if route:
            with tracer.scope("sample"):
                for dst in send_to:
                    comm.send(ids, dst)
                for src in recv_from:
                    ids = comm.recv((B, T), torch.long, src)
        x = StepInput(B, T, ids if is_input else None, all_logits)
        logits = strategy.forward(npu_id, x, cache, kernels, comm)
        cache.advance(T)
        return logits

    out = {"logits": {}, "tokens": {}}
    chunk = wl["prefill_chunk"] or P
    prefill_logits = []
    for s in range(0, P, chunk):
        logits = step("prefill", prompt[:, s:s + chunk], min(chunk, P - s), all_logits=job["all_logits"])
        if has_logits:
            prefill_logits.append(logits)
    token = None
    if has_logits:
        out["logits"]["prefill"] = torch.cat(prefill_logits, 1) if job["all_logits"] else logits[:, -1]
        token = logits[:, -1].argmax(-1)
        out["tokens"]["first"] = token

    if wl["mode"] == "decode":
        tokens, step_logits = [token], []
        for _ in range(wl["gen_tokens"] - 1):
            ids = token[:, None] if token is not None else None
            logits = step("decode", ids, 1, route=True)
            if has_logits:
                step_logits.append(logits[:, -1])
                token = logits[:, -1].argmax(-1)
                tokens.append(token)
        if has_logits:
            out["logits"]["decode"] = step_logits
            out["tokens"]["decode"] = torch.stack(tokens, 1)
    elif wl["mode"] == "spec_verify":
        draft = job["draft_ids"]
        logits = step("verify", draft, draft.shape[1], all_logits=True)
        if has_logits:
            out["logits"]["verify"] = logits
            out["tokens"]["verify"] = logits.argmax(-1)

    out["passes"] = passes
    out["kv_cache_elements"] = cache.num_elements()
    return out
