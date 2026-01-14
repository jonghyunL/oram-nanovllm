#!/bin/bash

# python3 ttft.py llama3 500  >> no_cache/llama3_naive.out
# python3 ttft.py llama3 1000 >> no_cache/llama3_naive.out
# python3 ttft.py llama3 1500 >> no_cache/llama3_naive.out
# python3 ttft.py llama3 2000 >> no_cache/llama3_naive.out
python3 ttft.py llama3 2500 >> no_cache/llama3_naive.out
python3 ttft.py llama3 3000 >> no_cache/llama3_naive.out
python3 ttft.py llama3 3500 >> no_cache/llama3_naive.out
python3 ttft.py llama3 4000 >> no_cache/llama3_naive.out

python3 ttft.py qwen3 500  >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 1000 >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 1500 >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 2000 >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 2500 >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 3000 >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 3500 >> no_cache/qwen3_naive.out
python3 ttft.py qwen3 4000 >> no_cache/qwen3_naive.out

python3 ttft.py gemma2 500   >> no_cache/gemma2_naive.out
python3 ttft.py gemma2 1000  >> no_cache/gemma2_naive.out
python3 ttft.py gemma2 1500  >> no_cache/gemma2_naive.out
# python3 ttft.py gemma2 2000  >> no_cache/gemma2_naive.out
# python3 ttft.py gemma2 2500  >> no_cache/gemma2_naive.out
# python3 ttft.py gemma2 3000  >> no_cache/gemma2_naive.out
# python3 ttft.py gemma2 3500  >> no_cache/gemma2_naive.out
# python3 ttft.py gemma2 4000  >> no_cache/gemma2_naive.out


