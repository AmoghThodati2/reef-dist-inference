"""Hugging Face reference outputs for a workload (correctness only, never timed).

Mirrors exactly what runtime.generate does, using HF's own forward with its KV cache:
  prefill -> logits of the last prompt token (or all, if requested), first greedy token
  decode  -> gen_tokens greedy tokens and the logits of each decode step
  spec_verify -> draft = HF's greedy continuation (draft_tokens long, a stand-in for a
                 perfect draft model); one verify pass of the draft against the prompt cache
"""
import torch


@torch.no_grad()
def hf_reference(model, prompt_ids, workload, all_logits=False):
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits
    ref = {"prefill_logits": logits if all_logits else logits[:, -1]}
    first = logits[:, -1].argmax(-1)
    ref["first_token"] = first
    mode = workload["mode"]
    if mode == "decode" or mode == "spec_verify":
        n = workload["gen_tokens"] if mode == "decode" else workload["draft_tokens"]
        tokens, step_logits, p = [first], [], past
        for _ in range(n - 1):
            o = model(tokens[-1][:, None], past_key_values=p, use_cache=True)
            p = o.past_key_values
            step_logits.append(o.logits[:, -1])
            tokens.append(o.logits[:, -1].argmax(-1))
        tokens = torch.stack(tokens, 1)
        if mode == "decode":
            ref["decode_tokens"] = tokens
            ref["decode_logits"] = step_logits
        else:
            ref["draft_tokens"] = tokens
            o = model(tokens, past_key_values=past, use_cache=True)
            ref["verify_logits"] = o.logits
            ref["verify_tokens"] = o.logits.argmax(-1)
    return ref
