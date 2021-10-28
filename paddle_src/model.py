########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import logging

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.ops import einsum

logger = logging.getLogger(__name__)


def mish(x):
    return x * F.tanh(F.softplus(x))


class RWKV_TimeMix(nn.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        self.time_ww = self.create_parameter(
            shape=[config.n_head, config.ctx_len, config.ctx_len],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.time_gamma = self.create_parameter(
            shape=[config.ctx_len, 1],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

        self.time_shift = nn.Pad2D([1, 0], data_format="NLC")

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)
        self.output = nn.Linear(config.n_attn, config.n_embd)

        # self.key.scale_init = 0
        # self.receptance.scale_init = 0
        # self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.shape

        # TT = self.ctx_len
        # w = F.pad(self.time_w, [0, 0, 0, TT])
        # w = paddle.tile(w, [TT])
        # w = w[:, :-TT].reshape(shape=[-1, TT, 2 * TT - 1])
        # w = w[:, :, TT - 1 :]
        # w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        # w = paddle.where(self.mask[:T, :T] == 0, paddle.zeros((1,)), w)

        x = paddle.concat(
            [self.time_shift(x[:, :-1, : C // 2]), x[:, :, C // 2 :]], axis=-1
        )

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = paddle.clip(k, min=-60, max=30).exp()

        sum_k = paddle.cumsum(k, axis=1)

        kv = (k * v).reshape(shape=[B, T, self.n_head, self.head_size])

        wkv = einsum("htu,buhc->bthc", self.time_ww[:,:T,:T], kv).reshape(shape=[B, T, -1])

        rwkv = F.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        return rwkv * self.time_gamma[:T, :]


class RWKV_ChannelMix(nn.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.Pad2D([1, 0], data_format="NLC")

        hidden_sz = 5 * config.n_ffn // 2
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        # self.receptance.scale_init = 0
        # self.weight.scale_init = 0

    def forward(self, x):
        C = x.shape[-1]

        x = paddle.concat(
            [self.time_shift(x[:, :-1, : C // 2]), x[:, :, C // 2 :]], axis=-1
        )
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        wkv = self.weight(mish(k) * v)
        rwkv = F.sigmoid(r) * wkv

        return rwkv


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Layer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = RWKV_TimeMix(config, layer_id)
        self.mlp = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.time_out = self.create_parameter(
            shape=[1, config.ctx_len, 1],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias_attr=False)
        self.ctx_len = config.ctx_len
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_ctx_len(self):
        return self.ctx_len

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x * self.time_out[:, :T, :]
        x = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.reshape(-1, x.shape[-1]), targets.flatten())

        return x, loss


if __name__ == "__main__":
    pass
