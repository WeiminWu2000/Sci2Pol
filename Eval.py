
import json
import os
import re
import time
import tqdm
import argparse
import numpy as np
from sklearn.metrics import f1_score
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description="Evaluate model results.")
parser.add_argument("--task", type=str, required=False, default=None, help="Specific task to evaluate (e.g., 'task1', 'task16'). If not specified, evaluates all tasks.")
parser.add_argument("--response_folder", required=False, default="Inference_Results", type=str, help="Response folder path for inference results")
parser.add_argument("--output_folder", required=False, default="Evaluation_Results", type=str, help="Output folder path for evaluation results")
parser.add_argument("--dataset_folder", required=False, default="../sci2pol_data", type=str, help="Dataset folder path")
args = parser.parse_args()
specific_task = args.task
response_folder = args.response_folder
output_folder = args.output_folder
dataset_folder = args.dataset_folder
os.makedirs(output_folder, exist_ok=True)

# Configure Gemini API
# Replace with your actual API key
genai.configure(api_key='YOUR_API_KEY')
gemini_model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")

def bootstrap_scores(scores, B=1000, rng_seed=42):
    """
    Returns (mean_score, standard_error) computed with B bootstrap resamples.
    """
    if not scores:
        return 0.0, 0.0
    
    rng = np.random.default_rng(rng_seed)
    n = len(scores)
    boot = np.empty(B, dtype=np.float32)

    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        boot[b] = np.mean([scores[i] for i in idx])
    return float(np.mean(scores)), float(boot.std(ddof=1))

def LLM_prompt_sum(task_type, source_passage, summary):
    prompt = f'''You are a scientific expert evaluating a summary that restates a **{task_type}** described in a scientific paper and uses the policy-brief style sentences.
    
You are a strict and critical evaluator of summaries. Evaluate the summary on the following dimensions using a 1-5 scale (1 = very poor, 5 = excellent). Be conservative in your judgments: do not give high scores unless the summary is genuinely outstanding.

(1) Clarity: whether the summary is reader-friendly and expresses ideas clearly  
(2) Accuracy: whether the summary contains the same information as the source document  
(3) Coverage: how well the summary covers the important information from the source document  
(4) Overall quality: how good the summary is overall at representing the source document; a good summary is a shorter piece of text that has the essence of the original and tries to convey the same information as the source document

Return only a JSON object in this format:

{{
  'clarity': <1-5>,
  'accuracy': <1-5>,
  'coverage': <1-5>,
  'overall_quality': <1-5>
}}

---

Source Passage:
{source_passage}

Summary:
{summary}
'''
    return prompt

def LLM_prompt_brief(sci_paper, expert_brief, llm_brief):
    prompt = f'''You are a **strict** policy-brief evaluator. Given the full scientific PAPER, an EXPERT-written reference brief, and a CANDIDATE brief, grade the CANDIDATE on four dimensions and produce a compact JSON report.

### Evaluation dimensions & conservative 0-5 rubric
Start each score at 0 and add points only when the brief clearly meets the criterion.  
Reserve 4 or 5 for **near-flawless** performance; 3 means "solid but with notable gaps"; 2 or below signals clear problems.  
0 = disastrous 1 = poor 2 = fair 3 = good 4 = very good 5 = excellent  

1. ContextualDepth: Does the CANDIDATE capture the study's essential quantitative findings, methods, and broader context (e.g., raw-material outlook, scenario count) **without missing key facts or adding fluff**?  
2. HallucinationRisk: Are **all** claims traceable to the PAPER (or universally known)? Deduct heavily for any unsupported number or causal claim.  
3. ReadabilityTone: Is the brief concise, logically ordered, written in active voice, and appropriate for policymakers? Penalize lengthy sentences or jargon.  
4. Actionability: Are policy implications concrete, tied directly to evidence, and immediately useful? Vague or speculative advice ≤ 2.

### Output format (MUST be valid JSON; numeric scores only, no prose)
{{{{
  "contextual_depth": <0-5>,
  "hallucination_risk": <0-5>,
  "readability_tone": <0-5>,
  "actionability": <0-5>
}}}}

### Strict-grading instructions
• Score conservatively: if unsure, choose the **lower** score.  
• Base each score only on comparisons between PAPER and CANDIDATE; EXPERT_BRIEF is reference context.  
• Do **not** include explanations, comments, or extra keys—return exactly the JSON object above.  

BEGIN EVALUATION
### PAPER
{sci_paper}
### EXPERT_BRIEF
{expert_brief}
### CANDIDATE_BRIEF
{llm_brief}
END INPUT
'''
    return prompt

def LLM_prompt_policy_problem_importance(sci_paper):
    prompt = f'''You are a **strict** policy-brief evaluator. Given the full scientific PAPER, assign an **importance score** to each structural component for effectively communicating the policy problem, based only on the PAPER.

Each of the following component types represents a possible building block of a well-structured policy problem:

1. background — what drives the problem (e.g., scientific, environmental, or economic context)  
2. existing_problem — the current obstacle, mismatch, or challenge  s
3. consequence — potential risks if the problem is not addressed  
4. attention_problem — the key policy issue or question requiring urgent attention  
5. supporting_detail — clarification or elaboration of any of the above

### Scoring Instructions
• Assign an **importance score** between 0.0 and 1.0 for each component.  
• A **higher score** means the component is essential for understanding the policy problem described in the PAPER.  
• A **lower score** means the component is optional, minor, or not clearly relevant.  
• If a component is not justified by the PAPER, assign 0.0.

### Strict-grading instructions
• Score conservatively: if unsure, choose the **lower** score.  
• Base each score **only** on the PAPER—do not rely on external references or assumptions.  
• Do **not** include explanations, comments, or extra keys—return exactly the JSON object below.

### Output Format (MUST be valid JSON)
Return a dictionary with the five components and float scores rounded to one decimals:

{{{{
  "background": <float>,
  "existing_problem": <float>,
  "consequence": <float>,
  "attention_problem": <float>,
  "supporting_detail": <float>
}}}}

BEGIN EVALUATION  
### PAPER  
{sci_paper}  
END INPUT
'''
    return prompt

def LLM_prompt_policy_problem_quality(sci_paper, llm_problem):
    prompt = f'''You are a **strict** policy-brief evaluator. Given the full scientific PAPER and the CANDIDATE's policy problem paragraph, assign **quality scores** to five specific aspects of how well the problems are conveyed in CANDIDATE_POLICY_PROBLEM.

Each sentence in the CANDIDATE may contribute to one or more of the following **component types**:

1. background — what drives the problem (e.g., scientific, environmental, or economic context)  
2. existing_problem — the current obstacle, mismatch, or challenge  
3. consequence — potential risks if the problem is not addressed  
4. attention_problem — the key policy issue or question requiring urgent attention  
5. supporting_detail — clarification or elaboration of any of the above

### Scoring Instructions
• Assign a **quality score** between 0.0 and 1.0 for each component.  
• A **higher score** means the content is clear, logical, and strongly aligned with the PAPER.  
• A **lower score** means the content is vague, incorrect, poorly structured, or missing.  
• If a component is not addressed, assign 0.0.

### Strict-grading instructions
• Score conservatively: if unsure, choose the **lower** score.  
• Base each score **only** on comparisons between PAPER and CANDIDATE.  
• Do **not** include explanations, comments, or extra keys—return exactly the JSON object below.  
• Only evaluate content in the CANDIDATE_POLICY_PROBLEM.

### Output Format (MUST be valid JSON)
Return a dictionary with the five components and float scores rounded to one decimals:

{{{{
  "background": <float>,
  "existing_problem": <float>,
  "consequence": <float>,
  "attention_problem": <float>,
  "supporting_detail": <float>
}}}}

BEGIN EVALUATION  
### PAPER  
{sci_paper}  
### CANDIDATE_POLICY_PROBLEM  
{llm_problem}  
END INPUT
'''
    return prompt

def LLM_prompt_findings(sci_paper, llm_findings):
    prompt = f'''You are a **strict** policy-brief evaluator. Given the full scientific PAPER and the CANDIDATE's findings section, assign **quality scores** to five specific aspects of how well the findings are conveyed in CANDIDATE_FINDINGS.

Evaluate the following five criteria:

1. completeness — does the section include **all important findings** from the PAPER?  
2. importance — are the findings mentioned in the section actually **important** according to the PAPER?  
3. accuracy — are the described findings **factually correct** and consistent with the PAPER?  
4. summarizing_findings — does the section effectively **emphasize and summarize** the key messages or implications from the data, rather than just listing facts?  
5. specification_to_findings — does the section **clarify the scope, context, or limitations** of the findings, including conditions under which they apply?

### Scoring Instructions
• Assign a **score between 0.0 and 1.0** for each of the five criteria above.  
• A **higher score** means the section performs well on that criterion.  
• A **lower score** means the section is vague, misleading, incomplete, or missing that dimension.  

### Strict-grading instructions
• Score conservatively: if unsure, choose the **lower** score.  
• Base each score **only** on comparisons between PAPER and CANDIDATE.  
• Do **not** include explanations, comments, or extra keys—return exactly the JSON object below.

### Output Format (MUST be valid JSON)
Return a dictionary with the five components and float scores rounded to one decimals:

{{{{
  "completeness": <float>,
  "importance": <float>,
  "accuracy": <float>,
  "summarizing_findings": <float>,
  "specification_to_findings": <float>
}}}}

BEGIN EVALUATION  
### PAPER  
{sci_paper}  
### CANDIDATE_FINDINGS  
{llm_findings}  
END INPUT
'''
    return prompt

def LLM_prompt_method(sci_paper, llm_method):
    prompt = f'''You are a **strict** policy-brief evaluator. Given the full scientific PAPER and the CANDIDATE's methods section, assign **quality scores** to three core aspects of how the methodology is described.

Each score should be a float between 0.0 and 1.0. 

Evaluate the following three criteria:

1. clarity_and_purpose — Is the method described in a clear, structured way that highlights **what was done and why**, rather than simply listing tools or data sources?

2. technicality_appropriateness — Is the level of technical detail appropriate for a policy audience without excessive jargon, complexity, or irrelevant detail?

3. explanation_of_terms — Are technical terms, models, or data sources **explained** in accessible language and context without unexplained acronyms or unclear references?

### Scoring Instructions
• Assign a **score between 0.0 and 1.0** for each of the three criteria above.  
• A **higher score** means the section performs well on that criterion.  
• A **lower score** means the section is vague, overly technical, unexplained, or missing that dimension.  

### Strict-grading instructions
• Score conservatively: if unsure, choose the **lower** score.  
• Base each score **only** on comparisons between PAPER and CANDIDATE.  
• Do **not** include explanations, comments, or extra keys—return exactly the JSON object below.

### Output Format (MUST be valid JSON)
Return a dictionary with all three component scores (rounded to one decimal):

{{{{
  "clarity_and_purpose": <float>,
  "technicality_appropriateness": <float>,
  "explanation_of_terms": <float>
}}}}

BEGIN EVALUATION  
### PAPER  
{sci_paper}  
### CANDIDATE_METHOD  
{llm_method}  
END INPUT
'''
    return prompt

def LLM_prompt_implication(sci_paper, llm_implication):
    prompt = f'''You are a **strict** policy-brief evaluator. Given the full scientific PAPER and the CANDIDATE's policy implications section, assign **quality scores** to the following four criteria.

Evaluate the following four dimensions:

1. accuracy — Are the implications **explicitly supported** from the PAPER without speculative or hallucinated claims?

2. coverage — Does the section capture **all major implications** stated in the PAPER?

3. conciseness_and_distinctness — Are the listed implications **concise and non-redundant**? Each point should make a **distinct contribution**.

4. alignment_with_paper_intent — Does the implication reflect the **main message or takeaway** of the PAPER?  
   The brief should communicate the paper's **core emphasis**—such as a recommendation, a warning, a scientific insight, or a call to awareness.   

### Scoring Instructions
• Assign a **score between 0.0 and 1.0** for each of the four criteria above.  
• A **higher score** means the section performs well on that criterion.  
• A **lower score** means the section is vague, incorrect, redundant, or misaligned. 

### Strict-grading instructions
• Score conservatively: if unsure, choose the **lower** score.  
• Base each score **only** on comparisons between PAPER and CANDIDATE.  
• Do **not** include explanations, comments, or extra keys—return exactly the JSON object below.

### Output Format (MUST be valid JSON)
Return a dictionary with four component scores, each rounded to one decimal:

{{{{
  "accuracy": <float>,
  "coverage": <float>,
  "conciseness_and_distinctness": <float>,
  "alignment_with_paper_intent": <float>
}}}}

BEGIN EVALUATION  
### PAPER  
{sci_paper}  
### CANDIDATE_IMPLICATION  
{llm_implication}  
END INPUT
'''
    return prompt

# Determine which tasks to evaluate
if specific_task:
    # Validate the specific task
    all_tasks = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8', 'task9', 'task10', 'task11', 'task12', 'task13', 'task14', 'task15', 'task16', 'task17', 'task18']
    if specific_task not in all_tasks:
        print(f"❌ Error: Invalid task '{specific_task}'. Valid tasks are: {', '.join(all_tasks)}")
        exit(1)
    tasks_to_evaluate = [specific_task]
    print(f"🎯 Evaluating specific task: {specific_task}")
else:
    tasks_to_evaluate = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8', 'task9', 'task10', 'task11', 'task12', 'task13', 'task14', 'task15', 'task16', 'task17', 'task18']
    print(f"🔄 Evaluating all tasks: {len(tasks_to_evaluate)} tasks")

for task in tasks_to_evaluate:
# for task in ['task11']:
    load_path = os.path.join(response_folder, f'{task}_response.jsonl')
    if not os.path.exists(load_path):
        print(f"⚠️ File not found: {load_path}. Skipping...")
        continue

    with open(load_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    save_dir = output_folder
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{task}_eval.json')

    if task in ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task16', 'task17', 'task18']:
        y_true = [d['expected'] for d in data]
        y_pred = [d['response'] for d in data]

        # Clean the data - filter out None values and ensure we have valid predictions
        valid_pairs = []
        for true_val, pred_val in zip(y_true, y_pred):
            if pred_val is not None and true_val is not None:
                valid_pairs.append((true_val, pred_val))
        
        if not valid_pairs:
            print(f"⚠️ No valid predictions found for {task}")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'f1_micro': 0.0, 
                    'f1_macro': 0.0,
                    'accuracy_se': 0.0,
                    'num_samples': len(y_true),
                    'num_valid_samples': 0
                }, f, indent=2)
            continue
        
        # Extract cleaned true and predicted values
        y_true_clean = [pair[0] for pair in valid_pairs]
        y_pred_clean = [pair[1] for pair in valid_pairs]

        f1_micro = f1_score(y_true_clean, y_pred_clean, average='micro')
        f1_macro = f1_score(y_true_clean, y_pred_clean, average='macro')
        
        # Calculate F1 scores for individual samples to compute standard error
        individual_f1_scores = []
        for i in range(len(y_true_clean)):
            # For individual samples, we can't compute micro/macro F1 meaningfully
            # Instead, we'll compute accuracy for each sample (1 if correct, 0 if wrong)
            individual_f1_scores.append(1.0 if y_true_clean[i] == y_pred_clean[i] else 0.0)
        
        # Calculate standard error for accuracy (which is equivalent to F1 for binary cases)
        _, accuracy_se = bootstrap_scores(individual_f1_scores)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'f1_micro': f1_micro, 
                'f1_macro': f1_macro,
                'accuracy_se': accuracy_se,
                'num_samples': len(y_true),
                'num_valid_samples': len(valid_pairs)
            }, f, indent=2)

    elif task in ['task7', 'task8', 'task9', 'task10']:
        # Load the source dataset for the task
        initial_data_path = os.path.join(dataset_folder, f'{task}.jsonl')
        with open(initial_data_path, 'r', encoding='utf-8') as f:
            initial_data = [json.loads(line) for line in f]
    
        # Build a map: id -> query (source passage)
        id_to_query = {d['id']: d['query'] for d in initial_data}
    
        # Prepare save path
        LLM_judge_save_path = os.path.join(output_folder, f'{task}_LLM_judge_response.jsonl')
        os.makedirs(os.path.dirname(LLM_judge_save_path), exist_ok=True)
    
        # Build list of records for evaluation
        preds_query = []
        for d in data:
            id_ = d['idx']
            query = id_to_query.get(id_, "")
            pred = d['response']
            preds_query.append({'id': id_, 'query': query, 'pred': pred})
    
        task_type_map = {
            'task7': 'Policy Problem',
            'task8': 'Scientific Research Findings',
            'task9': 'Scientific Research Study Methods',
            'task10': 'Policy Implications'
        }
        task_type = task_type_map[task]
    
        def send_request(Id, query, pred):
            prompt = LLM_prompt_sum(task_type, query, pred)
            try:
                response = gemini_model.generate_content(prompt)
                return {'idx': Id, 'response': response.text}
            except Exception as e:
                return {'idx': Id, 'response': f'ERROR: {e}'}
    
        # Run requests in parallel
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(send_request, sample['id'], sample['query'], sample['pred'])
                for sample in preds_query
            ]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)
    
        # Save responses as they complete
        with open(LLM_judge_save_path, 'w', encoding='utf-8') as fout:
            for record in results:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                fout.flush()
    
        # Parse LLM judge results
        parsed_results = []
        for record in results:
            idx = record['idx']
            content_str = record['response']
    
            if content_str.startswith('```json'):
                content_str = content_str.strip('` \n')[len('json'):].strip()
    
            try:
                parsed_scores = json.loads(content_str)
                parsed_results.append({
                    'id': idx,
                    **parsed_scores
                })
            except json.JSONDecodeError as e:
                print(f'[Error] Failed to parse JSON for idx {idx}: {e}')
    
        # Calculate dataset score with standard error
        scores = []
        for item in parsed_results:
            try:
                total_score = (item['clarity'] + item['accuracy'] + item['coverage'] + item['overall_quality']) * 5
                scores.append(total_score)
            except KeyError:
                print(f"[Warning] Incomplete score data for id {item['id']}")

        if scores:
            mean_score, std_error = bootstrap_scores(scores, B=1000)
        else:
            mean_score, std_error = 0.0, 0.0

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'LLM judge score': mean_score,
                'standard_error': std_error,
                'num_valid_scores': len(scores)
            }, f, indent=2)

    elif task in ['task11', 'task12', 'task13', 'task14']:
        # Load the source dataset for tasks 12-15
        initial_data_path = os.path.join(dataset_folder, 'task16.jsonl')  # Use task16 dataset for paper extraction
        with open(initial_data_path, 'r', encoding='utf-8') as f:
            initial_data = [json.loads(line) for line in f]

        # Extract the scientific paper content from the query
        initial_query = {}
        for d in initial_data:
            match = re.search(
                r'Scientific Research Paper:\s*(.+?)\n{3}',  # stop at exactly three newlines
                d['query'],
                re.DOTALL
            )
            if match:
                initial_query[d['id']] = match.group(1).strip()
            else:
                initial_query[d['id']] = d['query']  # fallback to full query

        # Prepare save path
        LLM_judge_save_path = os.path.join(output_folder, f'{task}_LLM_judge_response.jsonl')
        os.makedirs(os.path.dirname(LLM_judge_save_path), exist_ok=True)

        # Build list of records for evaluation
        preds_query = []
        for d in data:
            id_ = d['idx'] if 'idx' in d else d['id']  # Handle both idx and id fields
            llm_section = d['response']  # This is the section content (problem/findings/methods/implication)
            sci_paper = initial_query.get(id_, "")
            preds_query.append({
                'id': id_, 
                'sci_paper': sci_paper,
                'llm_section': llm_section
            })

        # Determine which evaluation functions to use based on task
        task_prompts = {
            'task11': [
                ('problem_importance', LLM_prompt_policy_problem_importance, False),
                ('problem_quality', LLM_prompt_policy_problem_quality, True)
            ],
            'task12': [('findings', LLM_prompt_findings, True)],
            'task13': [('methods', LLM_prompt_method, True)],
            'task14': [('implication', LLM_prompt_implication, True)]
        }

        def send_request_tasks_12_15(Id, sci_paper, llm_section, eval_name, prompt_function, requires_llm_section):
            if requires_llm_section:
                prompt = prompt_function(sci_paper, llm_section)
            else:
                prompt = prompt_function(sci_paper)
            try:
                response = gemini_model.generate_content(prompt)
                return {'idx': Id, 'eval_name': eval_name, 'response': response.text}
            except Exception as e:
                return {'idx': Id, 'eval_name': eval_name, 'response': f'ERROR: {e}'}

        # Run requests in parallel for all evaluations for this task
        results = []
        evaluations = task_prompts[task]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for sample in preds_query:
                for eval_name, prompt_function, requires_llm_section in evaluations:
                    futures.append(executor.submit(
                        send_request_tasks_12_15, 
                        sample['id'], 
                        sample['sci_paper'], 
                        sample['llm_section'],
                        eval_name,
                        prompt_function,
                        requires_llm_section
                    ))
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)

        # Save responses as they complete
        with open(LLM_judge_save_path, 'w', encoding='utf-8') as fout:
            for record in results:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                fout.flush()

        # Parse LLM judge results and organize by sample ID
        sample_scores = {}
        for record in results:
            idx = record['idx']
            eval_name = record['eval_name']
            content_str = record['response']

            if content_str.startswith('```json'):
                content_str = content_str.strip('` \n')[len('json'):].strip()

            try:
                parsed_scores = json.loads(content_str)
                if idx not in sample_scores:
                    sample_scores[idx] = {}
                # Store scores with eval_name prefix
                for key, value in parsed_scores.items():
                    sample_scores[idx][f"{eval_name}_{key}"] = value
            except json.JSONDecodeError as e:
                print(f'[Error] Failed to parse JSON for idx {idx}, eval {eval_name}: {e}')

        # Calculate dataset score based on task-specific formula
        scores = []
        for idx, scores_dict in sample_scores.items():
            try:
                if task == 'task11':  # Problem section
                    prob_imp = {
                        'background': scores_dict.get('problem_importance_background', 0),
                        'existing_problem': scores_dict.get('problem_importance_existing_problem', 0),
                        'consequence': scores_dict.get('problem_importance_consequence', 0),
                        'attention_problem': scores_dict.get('problem_importance_attention_problem', 0),
                        'supporting_detail': scores_dict.get('problem_importance_supporting_detail', 0)
                    }
                    
                    prob_qual = {
                        'background': scores_dict.get('problem_quality_background', 0),
                        'existing_problem': scores_dict.get('problem_quality_existing_problem', 0),
                        'consequence': scores_dict.get('problem_quality_consequence', 0),
                        'attention_problem': scores_dict.get('problem_quality_attention_problem', 0),
                        'supporting_detail': scores_dict.get('problem_quality_supporting_detail', 0)
                    }
                    
                    section_score = (
                        prob_imp['background'] * prob_qual['background'] +
                        prob_imp['existing_problem'] * prob_qual['existing_problem'] +
                        prob_imp['consequence'] * prob_qual['consequence'] +
                        prob_imp['attention_problem'] * prob_qual['attention_problem'] +
                        prob_imp['supporting_detail'] * prob_qual['supporting_detail']
                    )
                    # Normalize to 100-point scale (max possible is 5.0)
                    section_score = (section_score / 5.0) * 100

                elif task == 'task12':  # Findings section
                    findings = {
                        'completeness': scores_dict.get('findings_completeness', 0),
                        'importance': scores_dict.get('findings_importance', 0),
                        'accuracy': scores_dict.get('findings_accuracy', 0),
                        'summarizing_findings': scores_dict.get('findings_summarizing_findings', 0),
                        'specification_to_findings': scores_dict.get('findings_specification_to_findings', 0)
                    }
                    
                    section_score = (
                        findings['completeness'] + findings['importance'] + findings['accuracy'] +
                        findings['summarizing_findings'] + findings['specification_to_findings']
                    )
                    # Normalize to 100-point scale (max possible is 5.0)
                    section_score = (section_score / 5.0) * 100

                elif task == 'task13':  # Methods section
                    methods = {
                        'clarity_and_purpose': scores_dict.get('methods_clarity_and_purpose', 0),
                        'technicality_appropriateness': scores_dict.get('methods_technicality_appropriateness', 0),
                        'explanation_of_terms': scores_dict.get('methods_explanation_of_terms', 0)
                    }
                    
                    section_score = (
                        methods['clarity_and_purpose'] * 2 + methods['technicality_appropriateness'] * 2 +
                        methods['explanation_of_terms']
                    )
                    # Normalize to 100-point scale (max possible is 5.0)
                    section_score = (section_score / 5.0) * 100

                elif task == 'task14':  # Implications section
                    implications = {
                        'accuracy': scores_dict.get('implication_accuracy', 0),
                        'coverage': scores_dict.get('implication_coverage', 0),
                        'conciseness_and_distinctness': scores_dict.get('implication_conciseness_and_distinctness', 0),
                        'alignment_with_paper_intent': scores_dict.get('implication_alignment_with_paper_intent', 0)
                    }
                    
                    section_score = (
                        implications['accuracy'] + implications['coverage'] + 
                        implications['conciseness_and_distinctness'] + implications['alignment_with_paper_intent']
                    )
                    # Normalize to 100-point scale (max possible is 4.0)
                    section_score = (section_score / 4.0) * 100

                scores.append(section_score)
            except KeyError as e:
                print(f"[Warning] Incomplete score data for id {idx} in {task}: {e}")

        if scores:
            mean_score, std_error = bootstrap_scores(scores, B=1000)
        else:
            mean_score, std_error = 0.0, 0.0

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'LLM judge score': mean_score,
                'standard_error': std_error,
                'num_valid_scores': len(scores)
            }, f, indent=2)

    elif task in ['task15']:
        # Load the source dataset for task16
        initial_data_path = os.path.join(dataset_folder, f'{task}.jsonl')
        with open(initial_data_path, 'r', encoding='utf-8') as f:
            initial_data = [json.loads(line) for line in f]

        # Extract the scientific paper content from the query
        initial_query = {}
        for d in initial_data:
            match = re.search(
                r'Scientific Research Paper:\s*(.+?)\n{3}',  # stop at exactly three newlines
                d['query'],
                re.DOTALL
            )
            if match:
                initial_query[d['id']] = match.group(1).strip()
            else:
                initial_query[d['id']] = d['query']  # fallback to full query

        # Prepare save path
        LLM_judge_save_path = os.path.join(output_folder, f'{task}_LLM_judge_response.jsonl')
        os.makedirs(os.path.dirname(LLM_judge_save_path), exist_ok=True)

        # Build list of records for evaluation
        preds_query = []
        for d in data:
            id_ = d['idx'] if 'idx' in d else d['id']  # Handle both idx and id fields
            expert_brief = d['expected']
            llm_brief = d['response']
            sci_paper = initial_query.get(id_, "")
            preds_query.append({
                'id': id_, 
                'sci_paper': sci_paper,
                'expert_brief': expert_brief, 
                'llm_brief': llm_brief
            })

        def send_request_task16(Id, sci_paper, expert_brief, llm_brief):
            prompt = LLM_prompt_brief(sci_paper, expert_brief, llm_brief)
            try:
                response = gemini_model.generate_content(prompt)
                return {'idx': Id, 'response': response.text}
            except Exception as e:
                return {'idx': Id, 'response': f'ERROR: {e}'}

        # Run requests in parallel
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(send_request_task16, sample['id'], sample['sci_paper'], 
                              sample['expert_brief'], sample['llm_brief'])
                for sample in preds_query
            ]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)

        # Save responses as they complete
        with open(LLM_judge_save_path, 'w', encoding='utf-8') as fout:
            for record in results:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                fout.flush()

        # Parse LLM judge results
        parsed_results = []
        for record in results:
            idx = record['idx']
            content_str = record['response']

            if content_str.startswith('```json'):
                content_str = content_str.strip('` \n')[len('json'):].strip()

            try:
                parsed_scores = json.loads(content_str)
                parsed_results.append({
                    'id': idx,
                    **parsed_scores
                })
            except json.JSONDecodeError as e:
                print(f'[Error] Failed to parse JSON for idx {idx}: {e}')

        # Calculate dataset score with standard error (task16 uses 0-5 scale, multiply by 5 for 0-100)
        scores = []
        for item in parsed_results:
            try:
                total_score = (item['contextual_depth'] + item['hallucination_risk'] + 
                             item['readability_tone'] + item['actionability']) * 5
                scores.append(total_score)
            except KeyError:
                print(f"[Warning] Incomplete score data for id {item['id']}")

        if scores:
            mean_score, std_error = bootstrap_scores(scores, B=1000)
            num_60 = sum([1 for score in scores if score >= 60])
            num_80 = sum([1 for score in scores if score >= 80])
        else:
            mean_score, std_error = 0.0, 0.0
            num_60, num_80 = 0, 0

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'mean_score': mean_score,
                'standard_error': std_error,
                'num_60': num_60,
                'num_80': num_80,
                'num_valid_scores': len(scores)
            }, f, indent=2)

    print(f"✅ Saved eval results to {save_path}")
    

if specific_task:
    print(f"🎉 Task {specific_task} evaluation finished.")
else:
    print("🎉 All tasks finished.")
