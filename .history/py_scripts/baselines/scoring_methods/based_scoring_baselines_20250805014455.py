import torch
import torch.nn.functional as F
from scoring_methods import fastMDE
import numpy as np
from scipy.fft import fft
from scipy.signal import stft

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_rank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 # convert to 1-indexed rank
    return -ranks.mean().item()

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_entropy(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()

# Log-Likelihood Log-Rank Ratio
def get_lrr(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    likelihood = get_likelihood(logits, labels)
    logrank = get_logrank(logits, labels)
    return likelihood / logrank

def get_lastde(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    templl = log_likelihood.mean(dim=1)

    # open-source
    # embed_size = 3
    # epsilon = 10 * log_likelihood.shape[1] 
    # tau_prime = 5

    # closed-source
    embed_size = 3
    epsilon = 1 * log_likelihood.shape[1] 
    tau_prime = 15


    aggmde = fastMDE.get_tau_multiscale_DE(ori_data = log_likelihood, embed_size=embed_size, epsilon=epsilon, tau_prime=tau_prime)
    lastde = templl / aggmde 
    return lastde.cpu().item()

# def get_specdetect(logits, labels):
#     assert logits.shape[0] == 1
#     assert labels.shape[0] == 1
#     labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
#     lprobs = torch.log_softmax(logits, dim=-1)
#     log_likelihood = lprobs.gather(dim=-1, index=labels)

#     # log_likelihood = log_likelihood - np.mean(log_likelihood, axis=1, keepdims=True)
#     log_likelihood = log_likelihood.squeeze(-1)  # remove the last dimension
#     templl = log_likelihood.mean(dim=1)
#     log_likelihood -= log_likelihood.mean(dim=1)

#     N = log_likelihood.shape[1]

#     # fft analysis
#     fft_y_orig = fft(log_likelihood.cpu().numpy())

#     power_half_y = (((np.abs(fft_y_orig))/N) ** 2)[:,:(int(N/2))]

#     # abs_y_orig = np.abs(fft_y_orig)
#     # normalization_y_orig = abs_y_orig / N
#     # normalization_half_y_orig = normalization_y_orig[:,:(int(N/2))]
#     # power_half_y = normalization_half_y_orig ** 2

#     # energy_total = np.sum(power_half_y, axis=1)
#     energy_mean = np.mean(power_half_y, axis=1)

#     return templl.cpu().numpy()*energy_mean

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

def get_specdetect(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)

    log_likelihood = log_likelihood - np.mean(log_likelihood, axis=1, keepdims=True)
    # log_likelihood = log_likelihood.squeeze(-1)  # remove the last dimension
    log_likelihood -= log_likelihood.mean(dim=1)

    N = log_likelihood.shape[1]

    # fft analysis
    fft_y_orig = fft(log_likelihood.cpu().numpy())

    power_half_y = (((np.abs(fft_y_orig))/N) ** 2)[:,:(int(N/2))]

    energy_total = np.sum(power_half_y, axis=1)

    return -energy_total