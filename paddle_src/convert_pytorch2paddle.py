from collections import OrderedDict


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path):
    import paddle
    import paddle.nn.functional as F
    import torch

    dont_transpose = [
        ".ln1.weight",
        ".ln2.weight",
        "ln_f.weight",
        "time_ww",
        "time_gamma",
        "time_out",
        "tok_emb.weight",
    ]
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")

    if hasattr(pytorch_state_dict, "state_dict"):
        config = pytorch_state_dict.config
        pytorch_state_dict = pytorch_state_dict.state_dict()

    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False

        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = paddle.to_tensor(v.data.numpy())

    for i in range(config.n_layer):
        prefix = f"blocks.{i}.attn."
        time_w = paddle_state_dict[prefix + "time_w"]
        time_alpha = paddle_state_dict[prefix + "time_alpha"]
        time_beta = paddle_state_dict[prefix + "time_beta"]

        TT = config.ctx_len
        T = config.ctx_len

        w = F.pad(time_w, [0, 0, 0, TT])
        w = paddle.tile(w, [TT])
        w = w[:, :-TT].reshape(shape=[-1, TT, 2 * TT - 1])
        w = w[:, :, TT - 1 :]
        w = w[:, :T, :T] * time_alpha[:, :, :T] * time_beta[:, :T, :]

        paddle_state_dict[prefix + "time_ww"] = w
        del paddle_state_dict[prefix + "time_w"]
        del paddle_state_dict[prefix + "time_alpha"]
        del paddle_state_dict[prefix + "time_beta"]

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "model/wangwen-2022-02-15.pth", "model/wangwen-2022-02-15.pdparams"
    )
