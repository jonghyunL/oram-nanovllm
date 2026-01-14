import os, sys, time, random, statistics 
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


def main(model_type: str, input_len):
    
    path_dict = {
        "llama3": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "qwen3": "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "gemma3": "~/.cache/huggingface/hub/models--gghfez--gemma-3-4b-novision/snapshots/23c1ca31428ceec00e9f630628453bcebf633dfe",
        "phi3": "~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/f3c06aed622e14ca0abf5115094e4fc9a9948f36",
        }
    path_to_model = path_dict[model_type]
    path = os.path.expanduser(path_to_model)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

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
    
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    for _ in range(5):
        try:
            time.sleep(2)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokenizer1 = AutoTokenizer.from_pretrained(path, use_fast=True)
            init_dur.append(time.perf_counter() - t0)
            del tokenizer1
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
            

            # time.sleep(10)
            # llama3.1: 10000 (V) --> 1000000 --> 1000000000000 --> 10000
            # qwen3 10000 --> 1000 --> 100 --> 1000000
            prompt = [random.randint(100, 10000) for _ in range(input_len)]
            prompt = " ".join(map(str, prompt))    

            tok_len = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # print(tokenizer.chat_template is not None)
            # prompt = "tiktoken is a fast BPE tokeniser."
            '''
            if tokenizer.chat_template is not None:
                prompts = [tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                )]
            else: 
                # print (prompt)
                prompts=tokenizer.encode(prompt, truncation=True, max_length=None, return_tensors="pt")
                print (prompts)
            '''
            # prompts=tokenizer.encode(prompt)
            prompts=tokenizer.encode(prompt, max_length=None, return_tensors="pt")
            torch.cuda.synchronize()
            token_dur.append(time.perf_counter() - t0)
            # prompt = [[ random.randint(0, 10000) for _ in range( len(prompts[0]) ) ]]
            prompt = prompts.tolist()
            
            # print(len(prompts[0]))
            # print(time.perf_counter() - t0)
            # print(prompts)
            outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
            pd, pnt, dd, dnt = llm.returnStat()
            prefill_dur.append(pd)
            prefill_nt.append(pnt)
            decode_dur.append(dd)
            decode_nt.append(dnt)
            llm.resetStat()

        except:
            # print (prefill_nt[0])
            init_duration = statistics.mean(init_dur[1:])
            token_duration = statistics.mean(token_dur[1:])
            prefill_duration = statistics.mean(prefill_dur[1:])
            prefill_nt = statistics.mean(prefill_nt[1:])
            decode_duration = statistics.mean(decode_dur[1:])
            decode_nt = statistics.mean(decode_nt[1:])
            decode_per_tok = decode_duration/decode_nt
            print (prefill_nt, decode_nt)
            print(init_duration, token_duration, prefill_duration, decode_per_tok)


    # print (prefill_nt[0])
    init_duration = statistics.mean(init_dur[1:])
    token_duration = statistics.mean(token_dur[1:])
    prefill_duration = statistics.mean(prefill_dur[1:])
    prefill_nt = statistics.mean(prefill_nt[1:])
    decode_duration = statistics.mean(decode_dur[1:])
    decode_nt = statistics.mean(decode_nt[1:])
    decode_per_tok = decode_duration/decode_nt
    print (init_dur)
    print (prefill_nt, decode_nt)
    print (token_dur[1:])
    print (prefill_dur[1:])
    print (decode_dur[1:])
    print(init_duration, token_duration, prefill_duration, decode_per_tok)

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
