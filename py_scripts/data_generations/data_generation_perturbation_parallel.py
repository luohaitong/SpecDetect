import nltk
import torch
import pandas as pd
import numpy as np
import time
import argparse
import tqdm
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import warnings
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')

# 模型名称映射
model_fullnames = {
    'gptj_6b': 'gpt-j-6b',
    'gptneo_2.7b': 'gpt-neo-2.7B',
    'gpt2_xl': 'gpt2-xl',
    'opt_2.7b': 'opt-2.7b',
    'bloom_7b': 'bloom-7b1',
    'falcon_7b': 'falcon-7b',
    'gemma_7b': "gemma-7b",
    'llama1_13b': 'Llama-13b',
    'llama2_13b': 'Llama-2-13B-fp16',
    'llama3_8b': 'Llama-3-8B',
    'opt_13b': 'opt-13b',
    'phi2': 'phi-2',
    "mgpt": 'mGPT',
    'qwen1.5_7b': 'Qwen/Qwen1.5-7B',
    'yi1.5_6b': '01-ai/Yi-1.5-6B',
    't5': 't5-3b'
}

# 用于匹配 <extra_id_*> 的正则表达式
pattern = re.compile(r"<extra_id_\d+>")

def load_model(model_name, gpu_id):
    """
    """
    device = f"cuda:{gpu_id}"
    model_fullname = model_fullnames[model_name]
    model_path = f"pretrain_models/{model_fullname}"

    print(f'[GPU-{gpu_id}] Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gptj' in model_name:
        model_kwargs.update(dict(revision='float16'))

    if 't5' in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs).to(device)
    else:
        # 注意：这要求单个模型能完整放入一张GPU的显存中
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.to(device)
    
    print(f'[GPU-{gpu_id}] Model loaded to {device}')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = f"pretrain_models/{model_fullname}"
    optional_tok_kwargs = {}
    if "opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'
    if "t5" in model_fullname:
        optional_tok_kwargs["model_max_length"] = 512
    tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if '13b' in model_fullname:
            tokenizer.pad_token_id = 0
    return tokenizer

# --- 数据加载和保存函数 (已修改) ---
def load_human_machine_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
    print(f"Raw data loaded from {data_file}")
    return data

def save_human_machine_perturbation_data(data, args, process_id):
    if args.main_results:
        output_basename = f"{args.output_file}_perturbation_{args.n_perturbations}"
    else:
        output_basename = f"{args.output_file}_perturbation_{args.n_perturbations}_{args.scenario}"
    
    args_file = f"{output_basename}_part_{process_id}.args.json"
    data_file = f"{output_basename}_part_{process_id}.raw_data.json"
        
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
    print(f"Process {process_id} finished. Raw data written into {data_file}")

def save_human_machine_regeneration_data(data, args, process_id):
    output_basename = f"{args.output_file}_regeneration_{args.n_regenerations}_{args.scenario}"
    args_file = f"{output_basename}_part_{process_id}.args.json"
    data_file = f"{output_basename}_part_{process_id}.raw_data.json"

    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
    print(f"Process {process_id} finished. Raw data written into {data_file}")

# --- 核心扰动和重写逻辑 (未修改) ---
def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    buffer_size = 1
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def replace_masks(args, mask_model, mask_tokenizer, texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(mask_model.device)
    outputs = mask_model.generate(**tokens, max_length=args.max_length, min_length=args.min_length, do_sample=True, top_p=args.mask_top_p,
                                  num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    tokens = [x.split(' ') for x in masked_texts]
    n_expected = count_masks(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    span_length = args.span_length
    pct = args.pct_words_masked
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts

def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    chunk_size = 1024
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], ceil_pct=ceil_pct))
    return outputs

def trim_to_shorter_length(texta, textb):
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def sample_from_model(args, model, tokenizer, texts):
    texts = [t.split(' ') for t in texts]
    texts = [' '.join(t[: int(len(t) * args.truncate_ratio)]) for t in texts]
    all_encoded = tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(model.device)
    model.eval()
    decoded = ['' for _ in range(len(texts))]
    tries = 0
    m = 0
    while m < args.min_length:
        if tries != 0:
            print(f"min words: {m}, needed {args.min_length}, regenerating (try {tries})")
        sampling_kwargs = {'temperature': args.temperature}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        outputs = model.generate(**all_encoded, min_length=args.min_length, max_length=args.max_length, do_sample=True,
                                   **sampling_kwargs, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        m = min(len(x.split()) for x in decoded) if decoded else 0
        tries += 1
    return decoded

def rewrite_texts_(args, model, tokenizer, texts):
    rewrote_texts = sample_from_model(args, model, tokenizer, texts)
    rewrote_texts = [trim_to_shorter_length(o, s)[1] for o, s in zip(texts, rewrote_texts)]
    return rewrote_texts

def rewrite_texts(args, model, tokenizer, texts):
    chunk_size = 10000
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(rewrite_texts_(args, model, tokenizer, texts[i:i + chunk_size]))
    return outputs

# --- 数据生成主函数 (已修改) ---
def generate_samples(data, args, model, tokenizer, gpu_id=0):
    model.eval()
    n_samples = len(data["original"])
    perturbs_rewrites = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"[GPU-{gpu_id}] Perturb or Rewrite text"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        if args.Generation_methods == "Perturbation":
            n_perturbations = args.n_perturbations
            p_sampled_text = perturb_texts(args, model, tokenizer, [sampled_text for _ in range(n_perturbations)])
            p_original_text = perturb_texts(args, model, tokenizer, [original_text for _ in range(n_perturbations)])
            perturbs_rewrites.append({
                "original": original_text,
                "sampled": sampled_text,
                "perturbed_sampled": p_sampled_text,
                "perturbed_original": p_original_text
            })
        else: # Rewrite
            n_regenerations = args.n_regenerations
            r_sampled_text = rewrite_texts(args, model, tokenizer, [sampled_text for _ in range(n_regenerations)])
            r_original_text = rewrite_texts(args, model, tokenizer, [original_text for _ in range(n_regenerations)])
            perturbs_rewrites.append({
                "original": original_text,
                "sampled": sampled_text,
                "rewrote_sampled": r_sampled_text,
                "rewrote_original": r_original_text
            })
    return perturbs_rewrites

# --- 新增的 Worker 函数 ---
def worker(process_id, gpu_id, data_chunk, args):
    """
    每个子进程执行的工作函数
    """
    print(f"Process {process_id} starting, assigned to GPU {gpu_id}")
    try:
        torch.cuda.set_device(gpu_id)

        if args.Generation_methods == "Perturbation":
            model_name_to_load = args.model
        else:
            model_name_to_load = args.rewrite_model
        
        model = load_model(model_name_to_load, gpu_id)
        tokenizer = load_tokenizer(model_name_to_load)
        
        new_data = generate_samples(data_chunk, args, model, tokenizer, gpu_id)
        
        if args.Generation_methods == "Perturbation":
            save_human_machine_perturbation_data(new_data, args, process_id)
        else:
            save_human_machine_regeneration_data(new_data, args, process_id)
            
        print(f"Process {process_id} on GPU {gpu_id} finished successfully.")
    except Exception as e:
        print(f"!!!!!! Process {process_id} on GPU {gpu_id} failed with error: {e} !!!!!!")


# --- 主执行模块 (已修改) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Generation_methods', type=str, default="Perturbation", choices=["Perturbation", "Rewrite"])
    parser.add_argument('--output_file', type=str, default="datasets/perturbation_data_detectgpt_npr/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="datasets/human_llm_data_for_experiment/xsum_llama3_8b")
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--model', type=str, default="t5")
    parser.add_argument('--truncate_ratio', type=float, default=0.5)
    parser.add_argument('--rewrite_model', type=str, default="llama3_8b")
    parser.add_argument('--n_regenerations', type=int, default=10)
    parser.add_argument('--n_prompt_tokens', type=int, default=30)
    parser.add_argument('--do_top_k', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true', default=False)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--scenario', type=str, default='white', choices=["black", "white"])
    parser.add_argument('--main_results', action='store_true', default=False)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=65)
    args = parser.parse_args()

    # ---- 并行处理启动逻辑 ----
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs available. Please run on a machine with CUDA-enabled GPUs.")
        exit()
    print(f"Found {n_gpus} GPUs. Starting parallel processing.")

    data = load_human_machine_data(args.dataset_file)
    data_keys = list(data.keys())
    total_samples = len(data[data_keys[0]])
    indices = list(range(total_samples))
    
    chunk_indices = np.array_split(indices, n_gpus)
    data_chunks = []
    for i in range(n_gpus):
        chunk = {}
        for key in data_keys:
            # 确保即使数据块为空，key也存在
            chunk[key] = [data[key][j] for j in chunk_indices[i]] if len(chunk_indices[i]) > 0 else []
        data_chunks.append(chunk)

    mp.set_start_method('spawn', force=True)
    processes = []
    for i in range(n_gpus):
        if len(data_chunks[i][data_keys[0]]) == 0:
            print(f"Skipping GPU {i} as there is no data to process.")
            continue
        p = mp.Process(target=worker, args=(i, i, data_chunks[i], args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n" + "="*50)
    print("All processes have finished.")
    print("You can now merge the output files using the 'merge_results.py' script.")
    print("="*50)