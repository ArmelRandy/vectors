"""
We are interested into computing a vector that will gather most of the information about
a Transformer based model. In the case of the encoder based models, we look into six matrices :
- W_q
- W_k
- W_v
- W_o
- W1 (first feedforward layer)
- W2 (second feedforward layer)
We leverage the fact for a given parameter, the difference between its value in the base model and its
value in the fine-tuned model is low rank. If we have a base model $f$ and its fine-tuned version $\tilde{f}$
we are interested in studying the behaviour of $\tilde{f} - f$.


For each layer $l$, we consider each of these matrices. For a given matrix $W$ (of size NxM), we compute its
singular value decomposition (SVD), i.e.
W = U S V^T
and we consider U[:, :r] (of size Nxr) as what matters (r is a chosen rank).
The final representation of the model is thus
- Mean(
    Concat(
        [
            U[:, :r].flatten()
        ]
        for W in [W_q, W_k, W_v, W_o, W1, W2]
    )
)
The mean if computed over the number of encoder layers.
The size of the resulting vector is : 
    (
        rows(W_q)+rows(W_k)+rows(W_v)+rows(W_o)+rows(W1)+rows(W2)
    )*r
"""


from transformers import AutoModel
from huggingface_hub import ModelCard
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import json


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
        "--r", type=int, default=8, help="The rank of the low rank decomposition"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--embedding_path",
        default="./bert/bert.out",
        help="The path of the file where we store the representations.",
    )
    parser.add_argument(
        "--metadata_path",
        default="./bert/bert.csv",
        help="The path of the file where we store the metadata.",
    )
    return parser.parse_args()


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


def get_bert_delta(model_name_or_path_1, model_name_or_path_2, r=8):
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

    i = 0
    matrix_rep = None
    for layer1, layer2 in zip(model_1.encoder.layer, model_2.encoder.layer):
        weights_1 = get_bert_layer_parameters(layer1)
        weights_2 = get_bert_layer_parameters(layer2)
        diff = [w1 - w2 for (w1, w2) in zip(weights_1, weights_2)]
        L = []
        for w in diff:
            U, S, V = torch.svd_lowrank(w, q=r)
            L.append(U)
        c = torch.cat(L)
        if i == 0:
            matrix_rep = c
        else:
            matrix_rep += c
        i += 1
    matrix_rep /= i
    return matrix_rep.view(-1)


if __name__ == "__main__":
    """
    We return :
    - The list of datasets the fine-tuned models was trained on.
    - The representation vectors of the fine-tuned models.
    """
    paths = []
    args = parse_args()
    with open(args.input_data_path, "r") as fin:
        for line in fin:
            dico = json.loads(line)
            path = dico["path"]
            paths.append(path)

    embeddings = []
    metadata = []
    for i, model_name_or_path in tqdm(enumerate(paths)):
        if i == 0:
            continue
        try:
            t = get_bert_delta(
                args.base_model_name_or_path, model_name_or_path, r=args.r
            )
            t = t.detach().numpy()
            try:
                card = ModelCard.load(model_name_or_path)
                train_datasets = card.data.datasets
            except:
                train_datasets = None
            embeddings.append(t)
            metadata.append((model_name_or_path, train_datasets))
            if args.verbose:
                print("YES")
        except:
            pass

    if args.verbose:
        print(f"LEN = {len(embeddings)}")

    embeddings = np.vstack([e.reshape(1, -1) for e in embeddings])
    np.savetxt(args.embedding_path, embeddings)

    df = pd.DataFrame(
        {
            "model_name_or_path": [e[0] for e in metadata],
            "datasets": [e[1] for e in metadata],
        }
    )

    df.to_csv(args.metadata_path, index=False)
