########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################
import random

import numpy as np
import paddle
import paddle.nn.functional as F


def top_k_logits(logits, k):
    v, ix = paddle.topk(logits, k)
    out = logits.clone()
    out[out < v[:, -1].unsqueeze(-1)] = -1e4
    return out


def top_p_probs(probs, p):
    out = probs.clone()
    sorted_probs, sorted_indices = paddle.topk(out, k=out.shape[-1])
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)
    sorted_indices_to_remove = (cumulative_probs > p).astype("int64")
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    indices_to_remove = sorted_indices.masked_select(sorted_indices_to_remove == 1)
    out[indices_to_remove] = 0
    return out


# top-p + top-k + pow&ratio sampling
def sample_logits(
    logits,
    pos,
    temperature=1.0,
    top_k=None,
    top_p=None,
    min_p_pow=None,
    min_p_ratio=None,
):
    logits = logits[:, pos, :] / temperature
    probs = F.softmax(logits, axis=-1)
    if min_p_ratio is not None:
        limit = paddle.pow(paddle.max(probs), min_p_pow) * min_p_ratio
        logits = paddle.where(
            probs < limit, paddle.to_tensor(-10000.0, dtype=logits.dtype), logits
        )
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = F.softmax(logits, axis=-1)
    if top_p is not None:
        probs[0] = top_p_probs(probs[0], top_p)
    ix = paddle.multinomial(probs, num_samples=1)

    return ix[0][0].cpu()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
