########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################
import random

import numpy as np
import paddle
import paddle.nn.functional as F


def top_p_probs(probs, top_p):
    out = probs.clone()
    sorted_probs, sorted_indices = paddle.topk(out, k=out.shape[-1])
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1).numpy()
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)].cpu())
    out[probs < cutoff] = 0
    return out


def sample_logits(logits, pos, temperature=1.0, top_p=None):
    logits = logits[0][pos, :]
    probs = F.softmax(logits, axis=-1)

    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    if temperature != 1.0:
        probs = paddle.pow(probs, 1.0 / temperature)

    ix = paddle.multinomial(probs, num_samples=1)

    return ix[0].cpu()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
