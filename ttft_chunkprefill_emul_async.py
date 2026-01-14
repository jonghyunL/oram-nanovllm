import os, sys, time, random, statistics, math, asyncio
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor


def main(model_type: str, input_len):
    
    path_dict = {
        "llama3": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "qwen3": "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "gemma2": "~/.cache/huggingface/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6",            
    }
    path_to_model = path_dict[model_type]
    path = os.path.expanduser(path_to_model)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=30)

    token_dur = []
    init_dur = []
    prefill_dur = []
    prefill_nt = []
    decode_dur = []
    decode_nt = []
    
    # Define LLM Engine
    llm_engine = LLMEngine(path, max_num_batched_tokens = 32768)
    
    # Thread pool for blocking tokenizer operations
    executor = ThreadPoolExecutor(max_workers=1)
    
    def gpu_prefill_step(scheduled_reqs, has_prefill):
        """GPU kernel execution - runs on CUDA"""
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        _ = llm_engine.model_runner.run(scheduled_reqs, has_prefill)
        end_evt.record()

        # Block here so timing is accurate
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / 1000.0  # seconds

    def tokenizer_step(i, chunk_iter, prompt, sampling_params):
        """CPU tokenization - blocking operation"""
        t0 = time.perf_counter()

        if i != chunk_iter:
            _ = llm_engine.tokenizer.vocab_size
            llm_engine.add_request(prompt[i], sampling_params)
            next_scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
        else:
            next_scheduled_reqs, has_prefill = [], False

        tok_time = time.perf_counter() - t0
        return tok_time, next_scheduled_reqs, has_prefill

    async def concurrent_gpu_tok_step(scheduled_reqs, has_prefill, i, chunk_iter, prompt, sampling_params, loop):
        """
        Run GPU prefill and tokenization concurrently using asyncio.
        GPU runs on CUDA (blocking but in thread).
        Tokenizer runs on CPU (blocking but in thread).
        We await both and return when both complete.
        """
        # Run both operations concurrently
        gpu_task = loop.run_in_executor(None, gpu_prefill_step, scheduled_reqs, has_prefill)
        tok_task = loop.run_in_executor(executor, tokenizer_step, i, chunk_iter, prompt, sampling_params)
        
        # Wait for both to complete and gather results
        prf_time = await gpu_task
        tok_time, next_scheduled_reqs, has_prefill = await tok_task
        
        return prf_time, tok_time, next_scheduled_reqs, has_prefill

    async def run_async_benchmark():
        """Main async benchmark loop"""
        loop = asyncio.get_event_loop()
        
        # warmup
        llm_engine.add_request(list(range(1024*3)), sampling_params)
        scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
        for _ in range(10):
            _ = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

        for iteration in range(30):
            try:
                # Load tokenizer
                t0 = time.perf_counter()
                tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
                init_dur.append(time.perf_counter() - t0)
                del tokenizer

                # Prepare prompt chunks
                chunk_len = 3000 
                prompt = [random.randint(0, 10000) for _ in range(input_len)]
                new_prompt = []
                if input_len < chunk_len:
                    prompt = " ".join(map(str, prompt))
                else:
                    chunk_iter = math.ceil(input_len / chunk_len) 
                    for i in range(chunk_iter):
                        start = i * chunk_len
                        end = min((i + 1) * chunk_len, input_len)
                        tmp = " ".join(map(str, prompt[start:end]))
                        new_prompt.append(tmp)
                    prompt = new_prompt 
                chunk_iter = len(prompt)

                # First tokenization (synchronous to set baseline)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                llm_engine.add_request(prompt[0], sampling_params)
                torch.cuda.synchronize()
                token_dur.append(time.perf_counter() - t0)
                print(f"Token 0: {token_dur[-1]:.4f}s")
                
                scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
                tot_scheduled_reqs = scheduled_reqs
                current_prefill = 0 

                # Concurrent GPU + tokenizer steps using asyncio.gather
                tasks = []
                for i in range(1, chunk_iter + 1):
                    task = concurrent_gpu_tok_step(
                        scheduled_reqs, has_prefill, i, chunk_iter, prompt, 
                        sampling_params, loop
                    )
                    tasks.append(task)

                # Execute all tasks concurrently and gather results
                results = await asyncio.gather(*tasks)

                # Process results in order
                for i, (prf_time, tok_time, next_scheduled_reqs, has_prefill) in enumerate(results, start=1):
                    current_prefill += prf_time if prf_time > tok_time else tok_time
                    scheduled_reqs = next_scheduled_reqs
                    print(f"Chunk {i}: GPU={prf_time:.4f}s, Tok={tok_time:.4f}s, Total prefill={current_prefill:.4f}s")

                # Emulation part
                while llm_engine.scheduler.running:
                    _ = llm_engine.scheduler.running.popleft()
                llm_engine.scheduler.running.append(tot_scheduled_reqs)

                # Decode phase
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                token_ids = llm_engine.model_runner.run(tot_scheduled_reqs, False)
                torch.cuda.synchronize()
                dd = time.perf_counter() - t0
                pd = current_prefill
                dnt = len(token_ids)
                pnt = len(tot_scheduled_reqs[0].token_ids)

                prefill_dur.append(pd)
                prefill_nt.append(pnt)
                decode_dur.append(dd)
                decode_nt.append(dnt)

                _ = llm_engine.tokenizer.vocab_size
                await asyncio.sleep(5)  # Use async sleep

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                break

        # Print statistics
        if len(token_dur) > 1:
            init_duration = statistics.mean(init_dur[1:])
            token_duration = statistics.mean(token_dur[1:])
            prefill_duration = statistics.mean(prefill_dur[1:])
            prefill_nt_mean = statistics.mean(prefill_nt[1:])
            decode_duration = statistics.mean(decode_dur[1:])
            decode_nt_mean = statistics.mean(decode_nt[1:])
            decode_per_tok = decode_duration / decode_nt_mean if decode_nt_mean > 0 else 0
            
            print("\n" + "="*70)
            print("RESULTS SUMMARY")
            print("="*70)
            print(f"Init duration: {init_duration:.4f}s")
            print(f"Token duration: {token_duration:.4f}s")
            print(f"Prefill duration: {prefill_duration:.4f}s")
            print(f"Prefill num tokens: {prefill_nt_mean:.0f}")
            print(f"Decode duration: {decode_duration:.4f}s")
            print(f"Decode num tokens: {decode_nt_mean:.0f}")
            print(f"Decode per token: {decode_per_tok:.6f}s")
            print("="*70)

    # Run the async event loop
    try:
        asyncio.run(run_async_benchmark())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python test_async_concurrent.py <model_type> (llama3 | qwen3 | gemma2) <input_len>")
    model_type = sys.argv[1]    
    input_len = int(sys.argv[2])

    main(model_type, input_len)

