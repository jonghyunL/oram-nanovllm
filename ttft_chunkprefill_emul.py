import os, sys, time, random, statistics, math 
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import torch


def main(model_type: str, input_len):
    
    path_dict = {
        "llama3": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "qwen3": "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "phi3": "~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f3c06aed622e14ca0abf5115094e4fc9a9948f36",    
        "gemma3": "~/.cache/huggingface/hub/models--gghfez--gemma-3-4b-novision/snapshots/23c1ca31428ceec00e9f630628453bcebf633dfe",
    }
    path_to_model = path_dict[model_type]
    path = os.path.expanduser(path_to_model)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=30)

    # prompts = [
    #     "introduce yourself",
    #     "list all prime numbers within 100",
    # ]
    random.seed(0)
    # prompt = [prompt]
    
    token_dur = []
    init_dur = []
    prefill_dur = []
    prefill_nt = []
    decode_dur = []
    decode_nt = []
    # llama3000
    # Qwen3 1700
    # phi3 1500
    # gemma 5000
    
    # Define LLM Engine
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    llm_engine = LLMEngine(path, max_num_batched_tokens = 32768)
    
    # warmup
    llm_engine.add_request(list(range(1024*3)), sampling_params)
    scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
    for _ in range(5):
        _ = llm_engine.model_runner.run(scheduled_reqs, has_prefill)

        

    for _ in range(10):
        try:
            # warmup tokenizer
            # prompt = "This is warmup"
            # llm_engine.tokenizer.encode(prompt)
            
            t0 = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True) 
            init_dur.append(time.perf_counter() - t0)
            del tokenizer
            ## Llama3
            # chunk_len = 3000 
            ## Qwen3 1700
            ## Phi3 1500
            ## Gemma3 5000
            chunk_len = 5000 
            prompt = [random.randint(0,10000) for _ in range(input_len)]
            new_prompt = []
            if input_len < chunk_len:
                prompt = [" ".join(map(str, prompt))]
            else:
                chunk_iter = math.ceil(input_len / chunk_len) 
                for i in range(chunk_iter):
                    start = i * chunk_len
                    end = min((i + 1) * chunk_len, input_len)
                    tmp = " ".join(map(str, prompt[start:end]))
                    new_prompt.append(tmp)
                prompt = new_prompt 
            chunk_iter = len(prompt)

            # tokenization 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm_engine.add_request(prompt[0], sampling_params )
            scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
            torch.cuda.synchronize()
            token_dur.append(time.perf_counter() - t0)
            # print (len (scheduled_reqs), len(scheduled_reqs[0].token_ids), has_prefill)
            print (init_dur[-1], token_dur[-1])
            
            
            current_prefill = 0 
            next_scheduled_reqs=None
            tot_scheduled_reqs = scheduled_reqs
            # make this into actually running two threads
            for i in range(1, chunk_iter+1): 
                
                # Measure Prefill time
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = llm_engine.model_runner.run(scheduled_reqs, has_prefill)
                prf_time = time.perf_counter() - t0

                # measure Tokenizer + Rebuild time for next
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if i != chunk_iter:
                    _ = llm_engine.tokenizer.vocab_size
                    llm_engine.add_request(prompt[i], sampling_params)
                    next_scheduled_reqs, has_prefill = llm_engine.scheduler.schedule()
                    # print(len(next_scheduled_reqs), len(next_scheduled_reqs[0].token_ids), has_prefill)
                    tot_scheduled_reqs[0].append_tokens(next_scheduled_reqs[0].token_ids)
                    # print(len(tot_scheduled_reqs[0].token_ids))
                else:
                    # next_scheduled_reqs = []
                    has_prefill = False
                torch.cuda.synchronize()
                tok_time = time.perf_counter() - t0

                current_prefill  += max(prf_time, tok_time)
                scheduled_reqs = next_scheduled_reqs
                print (i, prf_time , tok_time,current_prefill)

            while llm_engine.scheduler.running:
                _ = llm_engine.scheduler.running.popleft()
            llm_engine.scheduler.running.append(tot_scheduled_reqs)
            # print(len(llm_engine.scheduler.waiting), len(llm_engine.scheduler.running))
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            token_ids = llm_engine.model_runner.run(tot_scheduled_reqs, False)
            torch.cuda.synchronize()

            dd = time.perf_counter() - t0
            pd = current_prefill
            dnt = len(token_ids)
            # pnt = len(tot_scheduled_reqs[0].token_ids)
            # print(init_dur[0], token_dur[0],pnt, pd, dnt, dd)
            print(init_dur[0], token_dur[0], pd, dnt, dd)

            prefill_dur.append(pd)
            # prefill_nt.append(pnt)
            decode_dur.append(dd)
            decode_nt.append(dnt)
            _ = llm_engine.tokenizer.vocab_size
            time.sleep(3)
            torch.cuda.synchronize()
            


        except:
            # print (prefill_nt[0])
            init_duration = statistics.mean(init_dur[1:])
            token_duration = statistics.mean(token_dur[1:])
            prefill_duration = statistics.mean(prefill_dur[1:])
            # _prefill_nt = statistics.mean(prefill_nt[1:])
            decode_duration = statistics.mean(decode_dur[1:])
            print(decode_nt)
            _decode_nt = statistics.mean(decode_nt[1:])
            decode_per_tok = decode_duration/_decode_nt
            # print(_prefill_nt, _decode_nt)
            print(init_duration, token_duration, prefill_duration, decode_per_tok)


    # print (prefill_nt[0])
    # init_duration = statistics.mean(init_dur[1:])
    token_duration = statistics.mean(token_dur[1:])
    prefill_duration = statistics.mean(prefill_dur[1:])
    # _prefill_nt = statistics.mean(prefill_nt[1:])
    decode_duration = statistics.mean(decode_dur[1:])
    print(decode_nt)
    _decode_nt = statistics.mean(decode_nt[1:])
    decode_per_tok = decode_duration/_decode_nt
    # print (_prefill_nt, _decode_nt)
    # print(_decode_nt)
    print (token_dur)
    print(prefill_dur)
    print(decode_dur)
    # print(init_duration, token_duration, prefill_duration, decode_per_tok)
    print(init_dur[0], token_duration, prefill_duration, decode_per_tok)

    '''
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    '''

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python example.py <model_type> (llama | qwen3 | gemma2) <input_len>")
    model_type = sys.argv[1]    
    input_len = int(sys.argv[2])

    main(model_type, input_len)
