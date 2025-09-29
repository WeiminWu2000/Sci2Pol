# [Sci2Pol: Evaluating and Fine-tuning LLMs on Scientific-to-Policy Brief Generation](https://arxiv.org/abs/2509.21493)

The first benchmark and training dataset for evaluating and fine-tuning large language models (LLMs) on policy brief generation from a scientific paper.

<img width="673" alt="Screenshot 2025-05-16 at 3 24 05 AM" src="https://github.com/user-attachments/assets/e23908cb-e73e-43de-9b69-ffad7d7c2334" />

## 1. Dataset Sources

### (i) Sci2Pol-Bench (18 tasks): 

https://huggingface.co/datasets/Weimin2000/Sci2Pol-Bench

### (ii) Sci2Pol-Corpus: 

639 scientific paper-policy brief pairs before and after in-context polishing.

Before in-context polishing: https://huggingface.co/datasets/Weimin2000/Sci2Pol-Corpus/blob/main/Sci2Pol_Corpus_wo_polish.json

After in-context polishing:https://huggingface.co/datasets/Weimin2000/Sci2Pol-Corpus/blob/main/Sci2Pol_Corpus_w_polish.json


## 2. Setup environment

We use Python 3.10.4.

```
conda create -n Sci2Pol python=3.10.4
conda activate Sci2Pol
```

```
pip install -r requirements.txt
```

## 3. LLM Evaluation on Sci2Pol-Bench

### (i) Download Sci2Pol Dataset from Huggerface

```
from huggingface_hub import hf_hub_download

task_files = [f"task{i}.jsonl" for i in range(1, 20)]

for file in task_files:
    hf_hub_download(
        repo_id="Weimin2000/Sci2Pol-Bench",
        filename=file,
        repo_type="dataset",
        local_dir="./sci2pol_data"  # Optional: local output dir
    )
```

### (ii) LLM Inference for Different Tasks

Example: evaluate the response of grok-3-beta on task1.

```
python LLM_infer.py --model grok/grok-3-beta --task task1
```

#### Evaluated Models
'meta-llama/llama-3.1-8b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'meta-llama/llama-4-maverick', 'mistralai/mistral-large-2411', 'qwen/qwen-3-8b', 'qwen/qwen-3-235b-a22b', 'deepseek/deepseek-chat-v3-0324', 'deepseek/deepseek-r1', 'google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'grok/grok-3-beta', 'gpt/gpt-4o', 'anthropic/claude-3-7-sonnet'


#### Tasks
task1 ~ task18

### (iii) Evaluation of Inference Results for All 18 Tasks (Use Gemini-2.5-Pro as Judge)

```
python Eval.py --model grok/grok-3-beta
```
