import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
import time
import os
from scoring_methods import fastMDE
from utils.metrics import get_roc_metrics, get_precision_recall_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTJConfig, GPTJForCausalLM
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import warnings
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.signal import stft
import csv
import pandas as pd
warnings.filterwarnings('ignore')

# os.chdir("......") # cache_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
model_fullnames = {  'gptj_6b': 'gpt-j-6b', # https://huggingface.co/EleutherAI/gpt-j-6b/tree/main
                     'gptneo_2.7b': 'gpt-neo-2.7B', # https://huggingface.co/EleutherAI/gpt-neo-2.7B/tree/main
                     'gpt2_xl': 'gpt2-xl',# https://huggingface.co/openai-community/gpt2-xl/tree/main
                     'opt_2.7b': 'opt-2.7b', # https://huggingface.co/facebook/opt-2.7b/tree/main
                     'bloom_7b': 'bloom-7b1', # https://huggingface.co/bigscience/bloom-7b1/tree/main
                     'falcon_7b': 'falcon-7b', # https://huggingface.co/tiiuae/falcon-7b/tree/main
                     'gemma_7b': "gemma-7b", # https://huggingface.co/google/gemma-7b/tree/main
                     'llama1_13b': 'Llama-13b', # https://huggingface.co/huggyllama/llama-13b/tree/main
                     'llama2_13b': 'Llama-2-13B-fp16', # https://huggingface.co/TheBloke/Llama-2-13B-fp16/tree/main
                    #  'llama3_8b': 'meta-llama/Meta-Llama-3-8B-Instruct', # https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main
                     'llama3_8b': 'Llama-3-8B', # https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main
                     'opt_13b': 'opt-13b', # https://huggingface.co/facebook/opt-13b/tree/main
                     'phi2': 'phi-2', # https://huggingface.co/microsoft/phi-2/tree/main
                     "mgpt": 'mGPT', # https://huggingface.co/ai-forever/mGPT/tree/main
                     'qwen1.5_7b': 'Qwen1.5-7B', # https://huggingface.co/Qwen/Qwen1.5-7B/tree/main
                     'yi1.5_6b': 'Yi-1.5-6B',
                      'phi-4': 'phi-4',
                     'Qwen3-1.7B': 'Qwen3-1.7B',
                     'Qwen3-4B': 'Qwen3-4B',
                     'Qwen3-8B': 'Qwen3-8B',
                      'falcon3-10b': 'falcon3-10b',
                     'falcon3-7b': 'falcon3-7b',
                     'gemma3-12b': 'gemma3-12b',
                     'gemma3-1b': 'gemma3-1b',
                     'gemma3-4b': 'gemma3-4b',
                     'falcon3-3b': 'falcon3-3b'} # https://huggingface.co/01-ai/Yi-1.5-6B/tree/main 

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "pretrain_models/" + model_fullname
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gptj' in model_name:
        model_kwargs.update(dict(revision='float16'))
    if 'falcon3-3b' in model_name:
        model_kwargs.update(dict(torch_dtype = torch.float32))

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs, device_map="auto")
    # model = GPTJForCausalLM.from_pretrained(model_path, **model_kwargs, device_map="auto")
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    # model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    # model_path = "/pretrain_models/" + model_fullname
    # if model_fullname == 'llama3_8b':
    #     model_pathname = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_path = "pretrain_models/" + model_fullname

    optional_tok_kwargs = {}
    if "opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'

    base_tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def load_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    path_prefix = "."
    data_file = f"{input_file}.raw_data.json"
    # data_file = path_prefix + data_file
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def get_samples(logits, labels, args):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = args.n_samples
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    # lprobs = torch.softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood

def get_lastde(log_likelihood, args):
    embed_size = args.embed_size
    epsilon = int(args.epsilon * log_likelihood.shape[1])
    tau_prime = args.tau_prime

    templl = log_likelihood.mean(dim=1)
    aggmde = fastMDE.get_tau_multiscale_DE(ori_data = log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime)
    lastde = templl
    return lastde

def get_sampling_discrepancy(logits_ref, logits_score, labels, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)


    # lastde
    lastde_x = get_lastde(log_likelihood_x, args)
    sampled_lastde = get_lastde(log_likelihood_x_tilde, args)

    miu_tilde = sampled_lastde.mean()
    sigma_tilde = sampled_lastde.std()
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.cpu().item()

def get_sampling_discrepancy_frequency(logits_ref, logits_score, labels, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    # lastde
    lastde_x = get_freq_features(log_likelihood_x.cpu().numpy())
    sampled_lastde = get_freq_features(log_likelihood_x_tilde.cpu().numpy())

    miu_tilde = sampled_lastde.mean(axis=0)
    sigma_tilde = sampled_lastde.std(axis=0)
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy

def get_log_likelihood(logits_score, labels):
    # assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    # if logits_ref.size(-1) != logits_score.size(-1):
    #     # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
    #     vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
    #     logits_ref = logits_ref[:, :, :vocab_size]
    #     logits_score = logits_score[:, :, :vocab_size]

    # samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    # log_likelihood_x_tilde = get_likelihood(logits_score, samples)


    # # lastde
    # lastde_x = get_lastde(log_likelihood_x, args)
    # sampled_lastde = get_lastde(log_likelihood_x_tilde, args)

    # miu_tilde = sampled_lastde.mean()
    # sigma_tilde = sampled_lastde.std()
    # discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return log_likelihood_x.cpu().numpy().tolist()  # Convert to numpy and then to list for JSON serialization

def analyze_frequency(logits_original, logits_sampled):

    if len(logits_original) != len(logits_sampled):
        min_length = min(len(logits_original), len(logits_sampled))
        logits_original = logits_original[:min_length]
        logits_sampled = logits_sampled[:min_length]

    N=len(logits_original)
    x = np.arange(N)             # 频率个数
    half_x = x[range(int(N/2))]  #取一半区间

    # 对logits_original做频谱分析
    fft_y_orig = fft(logits_original)
    abs_y_orig = np.abs(fft_y_orig)
    normalization_y_orig = abs_y_orig / N
    normalization_half_y_orig = normalization_y_orig[range(int(N/2))]

    # 对logits_sampled做频谱分析
    fft_y_sampled = fft(logits_sampled)
    abs_y_sampled = np.abs(fft_y_sampled)
    normalization_y_sampled = abs_y_sampled / N
    normalization_half_y_sampled = normalization_y_sampled[range(int(N/2))]

    # 计算x从30之后的两个频率分量在频域中的能量占比（能量为幅度的平方）
    if N // 2 > 32:
        power_half_y_orig = normalization_half_y_orig ** 2
        power_half_y_sampled = normalization_half_y_sampled ** 2
        energy_total_orig = np.sum(power_half_y_orig)
        energy_total_sampled = np.sum(power_half_y_sampled)
        energy_high_orig = np.sum(power_half_y_orig[30:])
        energy_high_sampled = np.sum(power_half_y_sampled[30:])
        ratio_orig = energy_high_orig / energy_total_orig if energy_total_orig > 0 else 0
        ratio_sampled = energy_high_sampled / energy_total_sampled if energy_total_sampled > 0 else 0
        print(f"Original: Energy ratio of high freq: {ratio_orig:.4f}")
        print(f"Sampled:  Energy ratio of high freq: {ratio_sampled:.4f}")
    else:
        print("Sequence too short for freq 30/31 analysis.")
        
    plt.figure(figsize=(12, 6))
    # 原始logits时域
    plt.subplot(231)
    plt.plot(x, logits_original, label='original')
    plt.title('original')

    # 采样logits时域
    plt.subplot(232)
    plt.plot(x, logits_sampled, label='sampled', color='orange')
    plt.title('sampled')

    # 原始logits频谱
    plt.subplot(234)
    plt.plot(half_x, normalization_half_y_orig, 'blue')
    plt.title('original spectrum', fontsize=9, color='blue')

    # 采样logits频谱
    plt.subplot(235)
    plt.plot(half_x, normalization_half_y_sampled, 'orange')
    plt.title('sampled spectrum', fontsize=9, color='orange')

    # 对比频谱
    plt.subplot(233)
    plt.plot(half_x, normalization_half_y_orig, 'blue', label='original')
    plt.plot(half_x, normalization_half_y_sampled, 'orange', label='sampled')
    plt.title('spectrum compare')
    plt.legend()

    plt.tight_layout()
    plt.savefig("frequency_analysis.png")
    plt.show()
    plt.close()
 
def _stft_power(x, nperseg=64, window='hann', db_scale=True):
    """内部工具：返回 STFT 频率、时间、功率谱（可选 dB）"""
    x= np.squeeze(np.array(x))
    f, t, Zxx = stft(
        x, fs=1.0, window=window, nperseg=nperseg, boundary=None, padded=True
    )
    Sxx = np.abs(Zxx) ** 2
    # if db_scale:
    #     Sxx = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx

def get_high_freq_area2(logits):

    # logits = logits[0]
    logits = np.squeeze(logits)  # 确保是1维数组
    N=len(logits)
    x = np.arange(N)             # 频率个数
    half_x = x[range(int(N/2))]  #取一半区间

    # 对logits_original做频谱分析
    fft_y_orig = fft(logits)

    # print("fft_y_orig shape:", fft_y_orig.shape)

    abs_y_orig = np.abs(fft_y_orig)
    normalization_y_orig = abs_y_orig / N
    normalization_half_y_orig = normalization_y_orig[range(int(N/2))]

    # 计算x从30之后的两个频率分量在频域中的能量占比（能量为幅度的平方）
    if N // 2 > 30:
        power_half_y_orig = normalization_half_y_orig ** 2
        energy_total_orig = np.sum(power_half_y_orig)
        energy_high_orig = np.sum(power_half_y_orig[int(0.3 * normalization_half_y_orig.shape[0]):])
        ratio_orig = energy_high_orig / energy_total_orig if energy_total_orig > 0 else 0

    templl = logits.mean()  # 计算平均值

    return ratio_orig

def get_freq_features(log_likelihood):

    # embed_size = 4
    # epsilon = int(8 * log_likelihood.shape[1])
    # tau_prime = 15
    
    # aggmde = fastMDE.get_tau_multiscale_DE(ori_data = torch.tensor(log_likelihood), embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime).cpu().numpy()
    
    log_likelihood = np.swapaxes(log_likelihood, 0, 2)  # 先交换第0和第3维
    log_likelihood = np.squeeze(log_likelihood, axis=-1)

    # templl = log_likelihood.mean(axis=1)  # 计算平均值
    log_likelihood = log_likelihood - np.mean(log_likelihood, axis=1, keepdims=True)
    # log_likelihood = log_likelihood - np.expand_dims(templl, axis=-1)  # 减去平均值

    N = log_likelihood.shape[1]

    # fft analysis
    fft_y_orig = fft(log_likelihood)

    power_half_y = (((np.abs(fft_y_orig))/N) ** 2)[:,:(int(N/2))]

    # abs_y_orig = np.abs(fft_y_orig)
    # normalization_y_orig = abs_y_orig / N
    # normalization_half_y_orig = normalization_y_orig[:,:(int(N/2))]
    # power_half_y = normalization_half_y_orig ** 2

    # energy_total = np.sum(power_half_y, axis=1)
    energy_mean = np.mean(power_half_y, axis=1)
    # kurtosis = scipy.stats.kurtosis(power_half_y,axis=1)

    # stft analysis
    energy_total_stft = []
    mean_flux_stft = []
    for i in range(len(log_likelihood)):
        log_likelihood_tem = np.expand_dims(log_likelihood[i],axis=0)

        f, t, Sxx = _stft_power(log_likelihood_tem, nperseg=20, window='hann', db_scale=False)
        energies = np.sum(Sxx, axis=0)  # 按频率轴求和，形状 (len(t),)
        total_energy_stft = np.mean(energies)  # 所有窗口的均值
        fluxes = np.sqrt(np.sum((Sxx[:, 1:] - Sxx[:, :-1]) ** 2, axis=0))
        mean_flux = np.mean(fluxes) if len(fluxes) > 0 else 0
        energy_total_stft.append(total_energy_stft)
        mean_flux_stft.append(mean_flux)
    energy_total_stft = np.array(energy_total_stft)
    mean_flux_stft = np.array(mean_flux_stft)

    # print(energy_total)
    # print(np.exp(templl), energy_total)

    # stft多尺度多样entropy
    # DE_final = []
    # for i in range(len(log_likelihood)):
    #     log_likelihood_tem = np.expand_dims(log_likelihood[i],axis=0)

    #     DE = fastMDE.get_tau_multiscale_DE_stft(ori_data=log_likelihood_tem, epsilon=epsilon, tau_prime=tau_prime)
    #     DE_final.append(DE.cpu().numpy())
    # DE_final = np.array(DE_final)
    # DE_final = DE_final[:,0]
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))


    # scores = np.concatenate(
    #     [
    #         # templl[:, np.newaxis],
    #         - energy_total[:, np.newaxis],
    #         # - [:, np.newaxis],
    #         # - energy_mean / [:, np.newaxis],
    #         # kurtosis[:, np.newaxis],
    #         # -energy_total_stft[:, np.newaxis],
    #         # -(energy_total * energy_total/mean_flux_stft)[:, np.newaxis]
    #     ],
    #     axis = 1  # 在第二个维度拼接
    # )
    scores = -mean_flux_stft[:, np.newaxis]
    return scores

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()

    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name)
        reference_model = load_model(args.reference_model_name)
        reference_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # evaluate criterion
    name = "specdetect_flux++"
    criterion_fn = get_sampling_discrepancy_frequency

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    # time1=time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits_ref, logits_score, labels, args)
            # original_likelihood = get_log_likelihood(logits_score, labels)
            # tokenized_ori = tokenized
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)  
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, args)
            # sampled_likelihood = get_log_likelihood(logits_score, labels)
            # tokenized_sample = tokenized

        # analyze_frequency(original_likelihood[0], sampled_likelihood[0])
        # result
        # results.append({
        #                 "tokenized_original": tokenized_ori.input_ids.cpu().numpy().tolist(),
        #                 "tokenized_sampled": tokenized_sample.input_ids.cpu().numpy().tolist(),
        #                 "original_logits": original_likelihood,
        #                 "sampled_logits": sampled_likelihood})
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
    # time2=time.time()
    # time_per_sample_ms = ((time2 - time1) / n_samples) * 1000
    # print(f"Time per sample for {name}: {time_per_sample_ms:.2f} ms")

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]
                   }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    
    # results
    # results_file = os.getcwd() + f'{args.output_file}.{name}.json'
    # results_file = f'{args.output_file}.{name}.json'
    # results = { 'name': f'{name}_threshold',
    #             'info': {'n_samples': n_samples},
    #             # 'predictions': predictions,
    #             # 'raw_results': results,
    #             # 'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
    #             # 'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
    #             'roc_auc': roc_auc,
    #             'pr_auc': pr_auc,
    #             # 'loss': 1 - pr_auc
    #             }
    # with open(results_file, 'w') as fout:
    #     json.dump(results, fout)
    #     print(f'Results written into {results_file}')
    results = {'dataset':args.dataset_file.split("/")[-1],
        'model':args.reference_model_name,
        'feature index': name,
        'auc':round(roc_auc,4),
        'prauc':round(pr_auc,4),
        'n_samples': args.n_samples}

    # results_file = f'{args.output_file}/{name}.csv'
    # with open(results_file, 'a', newline='') as fout:
    #     writer = csv.DictWriter(fout, fieldnames=results.keys())
        
    #     # 若文件为空，先写入表头
    #     if fout.tell() == 0:
    #         writer.writeheader()
        
    #     # 写入数据行
    #     writer.writerow(results)
    #     print(f'Results appended to {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="experiment_results/spec/xsum_gpt4turbo")
    parser.add_argument('--dataset_file', type=str, default="datasets/human_llm_data_for_experiment/xsum_gpt4turbo")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--reference_model_name', type=str, default="gptj_6b")
    parser.add_argument('--scoring_model_name', type=str, default="gptj_6b")
    parser.add_argument('--embed_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=8)
    parser.add_argument('--tau_prime', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)