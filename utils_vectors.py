from transformers import AutoModel
import torch


def get_parameters(model):
    params = []
    for _, param in model.named_parameters():
        if param.ndim != 2:
            continue
        params.append(param)
    return params


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

    n_layers = len(model_1.encoder.layer)
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


def get_t5_layer_parameters(block):
    layer = block.layer
    first = layer[0]
    second = layer[1]

    q = first.SelfAttention.q
    k = first.SelfAttention.k
    v = first.SelfAttention.v
    o = first.SelfAttention.o

    wi = second.DenseReluDense.wi
    wo = second.DenseReluDense.wo
    layers = [q, k, v, o, wi, wo]
    weights = [l.weight for l in layers]
    return weights


def get_t5_delta(model_name_or_path_1, model_name_or_path_2, r=8):
    model_1 = AutoModel.from_pretrained(model_name_or_path_1)
    model_2 = AutoModel.from_pretrained(model_name_or_path_2, cache_dir=None)
    n_layers = len(model_1.encoder.block)
    i = 0
    matrix_rep = None
    for block1, block2 in zip(model_1.encoder.block, model_2.encoder.block):
        weights_1 = get_t5_layer_parameters(block1)
        weights_2 = get_t5_layer_parameters(block2)
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
