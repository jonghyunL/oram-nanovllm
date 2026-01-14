import os, sys, time, random 
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main(model_type: str):
    
    path_dict = {
        "llama": "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        # "qwen3": "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "qwen3": "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "gemm2": "~/.cache/huggingface/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6",
    }
    path_to_model = path_dict[model_type]
    path = os.path.expanduser(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1)

    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompt = [[random.randint(0, 10000) for _ in range(8500)]]
    # prompt = " ".join(map(str, prompt))    

    t0 = time.perf_counter()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    print(time.perf_counter() - t0)
    # print(prompts)
    outputs = llm.generate(prompt, sampling_params)
    '''
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    '''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python example.py <model_type> (llama | qwen3 | gemma2)")
    model_type = sys.argv[1]    

    main(model_type)
