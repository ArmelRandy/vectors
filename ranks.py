from transformers import AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import json


def get_bert_layer_parameters(layer):
    attention = layer.attention
    query = attention.self.query
    key = attention.self.key
    value = attention.self.value
    output_1 = attention.output.dense
    intermediate = layer.intermediate.dense
    output_2 = layer.output.dense
    layers = [query, key, value, output_1, intermediate, output_2]
    weights = [l.weight for l in layers]
    return weights


def get_bert_delta2(model_name_or_path_1, model_name_or_path_2):
    model_1 = AutoModel.from_pretrained(model_name_or_path_1)
    try:
        model_2 = AutoModel.from_pretrained(model_name_or_path_2)
    except:
        try:
            model_2 = AutoModel.from_pretrained(model_name_or_path_2, from_flax=True)
            print("GOOD FLAX!")
        except:
            model_2 = AutoModel.from_pretrained(model_name_or_path_2, from_tf=True)
            print("GOOD TF!")

    L = []
    for layer1, layer2 in zip(model_1.encoder.layer, model_2.encoder.layer):
        weights_1 = get_bert_layer_parameters(layer1)
        weights_2 = get_bert_layer_parameters(layer2)
        diff = [w1 - w2 for (w1, w2) in zip(weights_1, weights_2)]
        for w in diff:
            S = torch.linalg.svdvals(w)
            L.append(S.tolist())
    return L


def get_gpt_layer_parameters(layer):
    c_attn = layer.attn.c_attn
    c_proj = layer.attn.c_proj
    c_fc = layer.mlp.c_fc
    mlp_c_proj = layer.mlp.c_proj
    layers = [c_attn, c_proj, c_fc, mlp_c_proj]
    weights = [l.weight for l in layers]
    return weights


def get_gpt_delta2(model_name_or_path_1, model_name_or_path_2):
    model_1 = AutoModel.from_pretrained(model_name_or_path_1)
    try:
        model_2 = AutoModel.from_pretrained(model_name_or_path_2)
    except:
        try:
            model_2 = AutoModel.from_pretrained(model_name_or_path_2, from_flax=True)
            print("GOOD FLAX!")
        except:
            model_2 = AutoModel.from_pretrained(model_name_or_path_2, from_tf=True)
            print("GOOD TF!")
    L = []
    # for layer1, layer2 in zip(model_1.transformer.h, model_2.transformer.h):
    for layer1, layer2 in zip(model_1.h, model_2.h):
        weights_1 = get_gpt_layer_parameters(layer1)
        weights_2 = get_gpt_layer_parameters(layer2)
        diff = [w1 - w2 for (w1, w2) in zip(weights_1, weights_2)]
        for w in diff:
            S = torch.linalg.svdvals(w)
            L.append(S.tolist())
    return L


def get_t5_layer_parameters(block, decoder=False):
    layer = block.layer
    first = layer[0]
    q = first.SelfAttention.q
    k = first.SelfAttention.k
    v = first.SelfAttention.v
    o = first.SelfAttention.o
    layers = [q, k, v, o]
    if decoder:
        second = layer[1]  # Cross-attention
        third = layer[2]  # Feedforward
        q_1 = second.EncDecAttention.q
        k_1 = second.EncDecAttention.k
        v_1 = second.EncDecAttention.v
        o_1 = second.EncDecAttention.o
        wi = third.DenseReluDense.wi
        wo = third.DenseReluDense.wo
        layers.extend([q_1, k_1, v_1, o_1, wi, wo])
    else:
        second = layer[1]
        wi = second.DenseReluDense.wi
        wo = second.DenseReluDense.wo
        layers.extend([wi, wo])
    weights = [l.weight for l in layers]
    return weights


def get_t5_delta2(model_name_or_path_1, model_name_or_path_2):
    model_1 = AutoModel.from_pretrained(model_name_or_path_1)
    try:
        model_2 = AutoModel.from_pretrained(model_name_or_path_2)
    except:
        try:
            model_2 = AutoModel.from_pretrained(model_name_or_path_2, from_flax=True)
            print("GOOD FLAX!")
        except:
            model_2 = AutoModel.from_pretrained(model_name_or_path_2, from_tf=True)
            print("GOOD TF!")
    n_layers = len(model_1.encoder.block)
    L = []
    for i in range(n_layers):
        layer1_enc = model_1.encoder.block[i]
        layer1_dec = model_1.decoder.block[i]
        layer2_enc = model_2.encoder.block[i]
        layer2_dec = model_2.decoder.block[i]

        weights_1 = get_t5_layer_parameters(layer1_enc) + get_t5_layer_parameters(
            layer1_dec, decoder=True
        )
        weights_2 = get_t5_layer_parameters(layer2_enc) + get_t5_layer_parameters(
            layer2_dec, decoder=True
        )
        diff = [w1 - w2 for (w1, w2) in zip(weights_1, weights_2)]
        for w in diff:
            S = torch.linalg.svdvals(w)
            L.append(S.tolist())
    return L


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        default="bert-base-uncased",
        help="The model name or path to the base model for which we analyze the fine-tuned versions.",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        default="./bert/bert.json",
        help="The path to the json file containing the name of the models of interest (i.e. the fine-tuned models).",
    )
    parser.add_argument(
        "--A",
        type=int,
        required=True,
        help="The number of transformer layers of the model : Encoder+Decoder (12 for bert-base/roberta-base/t5-base, 24 for gpt2-medium).",
    )
    parser.add_argument(
        "--B",
        type=int,
        required=True,
        help="The number of matrices we exploit : Encoder+Decoder (6 for BERT/RoBERTa, 4 for GPT-2, 24 for T5).",
    )
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="For debugging purposes, maximal number of models to consider.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output_data_path",
        default="./bert/bert_ranks.csv",
        help="The path of the file where we store the ranks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    A = args.A
    B = args.B
    limit = args.limit
    paths = []
    with open(args.input_data_path, "r") as fin:
        for line in fin:
            dico = json.loads(line)
            path = dico["path"]
            paths.append(path)

    rng.shuffle(paths)
    df = pd.DataFrame(
        columns=[f"layer_{i}_parameter_{j}" for i in range(A) for j in range(B)]
    )
    print(f"THE NUMBER OF COLUMNS IS {len(df.columns)}.")
    models = []
    df_index = 0
    for i, model_name_or_path in tqdm(enumerate(paths)):
        if limit != -1 and i >= limit:
            break
        if i == 0:
            continue
        try:
            if "bert" in args.base_model_name_or_path:
                t = get_bert_delta2(args.base_model_name_or_path, model_name_or_path)
            elif "t5" in args.base_model_name_or_path:
                t = get_t5_delta2(args.base_model_name_or_path, model_name_or_path)
            elif "gpt" in args.base_model_name_or_path:
                t = get_gpt_delta2(args.base_model_name_or_path, model_name_or_path)
            else:
                pass
            if args.verbose:
                print(f"LENNNN {len(t)}")
            df.loc[df_index] = t  # [json.dumps(element) for element in t]
            models.append(model_name_or_path)
            df_index += 1
            if args.verbose:
                print(f"Iteration {i}, LEN = {len(t)}, NEW SHAPE DF = {df.shape}")
        except:
            print("ERROR")
            pass

    df["model"] = models
    df.to_csv(args.output_data_path, index=False)
