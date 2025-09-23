# Sci2Pol
Sci2Pol-Bench and Sci2Pol-Corpus, the first benchmark and training dataset for evaluating and fine-tuning large language models (LLMs) on policy brief generation from a scientific paper.

# Sci2Pol-Bench: A Benchmark for LLM Policy Brief Generation from Scientific Research

<img width="673" alt="Screenshot 2025-05-16 at 3 24 05 AM" src="https://github.com/user-attachments/assets/e23908cb-e73e-43de-9b69-ffad7d7c2334" />



## Setup environment

We use Python 3.10.4.

```
conda create -n Sci2Pol python=3.10.4
conda activate Sci2Pol
```

```
pip install -r requirements.txt
```

### Running the code needs inferenci API for LLMs, some of them are not free.

## Dataset

### Huggerface link

https://huggingface.co/datasets/Weimin2000/Sci2Pol_Bench

### Download Dataset from Huggerface

```
from huggingface_hub import hf_hub_download

task_files = [f"task{i}.jsonl" for i in range(1, 20)]

for file in task_files:
    hf_hub_download(
        repo_id="Weimin2000/Sci2Pol_Bench",
        filename=file,
        repo_type="dataset",
        local_dir="./sci2pol_data"  # Optional: local output dir
    )
```

## LLM Inference for Different Task

Example: evaluate the response of grok-3-beta on task1.

```
python LLM_infer.py --model grok/grok-3-beta --task task1
```

### Evaluated Models
'meta-llama/llama-3.1-8b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'meta-llama/llama-4-maverick', 'mistralai/mistral-large-2411', 'qwen/qwen-3-8b', 'qwen/qwen-3-235b-a22b', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1', 'google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'grok/grok-3-beta', 'gpt/gpt-4o', 'google/gemini-2.5-pro-preview-05-06', 'anthropic/claude-3-7-sonnet'

```
python LLM_infer.py --model anthropic/claude-3-7-sonnet --task task1
```

### Tasks
task1 ~ task19

## Evaluation of Inference Results for All 19 Tasks (Use Gemini-2.5-Pro as Judge)

```
python Eval.py --model grok/grok-3-beta
```
