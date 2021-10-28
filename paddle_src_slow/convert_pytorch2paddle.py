from collections import OrderedDict

def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path):
    import paddle
    import torch
    skip_weight = ".attn.mask"
    dont_transpose = [
        ".ln1.weight",
        ".ln2.weight",
        "ln_f.weight",
        "time_w",
        "time_alpha",
        "time_beta",
        "time_gamma",
        "time_out",
        "tok_emb.weight",
    ]
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    if hasattr(pytorch_state_dict, "state_dict"):
        pytorch_state_dict = pytorch_state_dict.state_dict()
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if skip_weight in k:
            continue
        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True
        oldk = k

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "model/xuanhuan-2021-10-26.pth", "model_state.pdparams"
    )
