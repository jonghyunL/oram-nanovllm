import torch
import time
import os
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

def prefill(path):
    # Benchmark actual model
    # llm_engine = LLMEngine(path, max_num_batched_tokens=16384)
    llm_engine = LLMEngine(path, max_num_batched_tokens=32768)
    sampling_params = SamplingParams(max_tokens=30)
    test_input = list(range(1024*3))
    print (test_input)
    for _ in range(1):
        llm_engine.add_request(test_input, sampling_params)
    
    scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
    print(scheduled_reqs, has_prefill)
    exit()

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    end_time = time.time()

    profiled_latency = (end_time - start_time) * 1000 / 10

    print(f"Profiled latency: {profiled_latency} ms")

def decode(path):
    # Benchmark actual model
    llm_engine = LLMEngine(path, max_num_batched_tokens=16384, max_model_len=16384)
    sampling_params = SamplingParams(max_tokens=30)
    for _ in range(8):
        llm_engine.add_request(list(range(16000)), sampling_params)
        scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
        print (len(scheduled_reqs), len(scheduled_reqs[0].token_ids), has_prefill)
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)
        # llm_engine.scheduler.update(scheduled_reqs, token_ids)

    scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
    print (len(scheduled_reqs), len(scheduled_reqs[0].token_ids), has_prefill)
    print (scheduled_reqs[0].token_ids[:5], has_prefill)
    exit()
    print([(r.num_scheduled_tokens, r.num_computed_tokens) for r in scheduled_reqs])

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        token_ids = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

    torch.cuda.synchronize()
    end_time = time.time()

    profiled_latency = (end_time - start_time) * 1000 / 10

    print(f"Profiled latency: {profiled_latency} ms")


if __name__ == "__main__":
    path = os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
# prefill(path)
decode(path)
