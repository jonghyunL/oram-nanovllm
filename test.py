import os, sys, time, random 
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main(model_type: str, input_len: int):
    
    path_dict = {
        # "llama3": "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "llama3": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "qwen3": "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "gemma2": "~/.cache/huggingface/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6",
        # "gemma3": "~/.cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"

        }
    path_to_model = path_dict[model_type]
    path = os.path.expanduser(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    random.seed(0)
    prompt = [[random.randint(0, 10000) for _ in range(100)]]

    _ = llm.generate(prompt, sampling_params, use_tqdm=False)
    llm.resetStat()
    # prompt = " ".join(map(str, prompt))    
    prompt = [[random.randint(0, 10000) for _ in range(input_len)]]
    
    # print(prompt[0])
    # print(prompts)
    outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
    pd, pnt, dd, dnt = llm.returnStat()
    print (pd, pnt, dd, dnt)

    '''
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    '''

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python example.py <model_type> (llama | qwen3 | gemma2) <input_len>, use_tqdm")
    model_type = sys.argv[1]  
    input_len = int(sys.argv[2])

    main(model_type, input_len)
