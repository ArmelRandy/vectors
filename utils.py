from huggingface_hub import ModelCard, hf_hub_download, ModelFilter
from huggingface_hub import HfFileSystem, HfApi
from tqdm import tqdm
import warnings
import argparse
import json

# Attribute to ignore
ATTRIBUTES_TO_IGNORE = [
    "_name_or_path",
    "architectures",
    "use_cache",
    "vocab_size",
    "transformers_version",
    "type_vocab_size",
    "torch_dtype",
    "gradient_checkpointing",
    "task_specific_params",
]


def equality(df_1, df_2):
    for col in df_1.keys():
        if col not in ATTRIBUTES_TO_IGNORE:
            try:
                if df_1[col] != df_2[col]:
                    return False
            except:
                return False
    return True


def get_base_model(finetuned_model_path: str):
    # Fine-tuned model to pre-trained model
    hf_api = HfApi()
    fs = HfFileSystem()
    card = ModelCard.load(finetuned_model_path)
    json_files = fs.glob(f"{finetuned_model_path}/**.json", detail=False)
    if len(json_files) == 0:
        warnings.warn("This model's repository is empty!")
        return None
    else:
        path_to_config = None
        path_to_adapter_config = None
        for f in json_files:
            if f.endswith("/config.json"):
                path_to_config = f
            if f.endswith("/adapter_config.json"):
                path_to_adapter_config = f
        if path_to_adapter_config:
            with fs.open(f"hf://{path_to_adapter_config}", "r") as f:
                adapter_dict = json.load(f)
            base_model_name_or_path = adapter_dict["base_model_name_or_path"]
            # check if base_model_name_or_path can be found on the hub
            c = hf_api.list_models(
                filter=ModelFilter(model_name=base_model_name_or_path)
            )
            if len(c) > 0:
                return base_model_name_or_path
        else:
            if not path_to_config:
                warnings.warn(
                    "This model's repository does not contain any configuration file. We can not find its base model."
                )
                return None
            else:
                pass

        with fs.open(f"hf://{path_to_config}", "r") as f:
            config_dict = json.load(f)
        try:
            _name_or_path = config_dict["_name_or_path"]
        except:
            _name_or_path = None

        if _name_or_path:
            # check if _name_or_path can be found on the hub
            c = hf_api.list_models(filter=ModelFilter(model_name=_name_or_path))
            if len(c) > 0:
                return _name_or_path

        model_type = config_dict["model_type"]
        candidate_models = hf_api.list_models(filter=ModelFilter(model_name=model_type))
        candidate_models = sorted(candidate_models, key=lambda x: x.id)
        # print(f"Here are the candidate models : {candidate_models}")
        # pre-trained models do not have the attribute "_name_or_path" in config.json
        # with some exceptions like starcoder
        print(f"There are {len(candidate_models)} candidate models.")


def get_finetuned_models(pretrained_model_path: str, verbose=False):
    # Pre-trained model to fine-tuned model(s)
    # It worth noting that the pre-trained model does not use peft

    hf_api = HfApi()
    fs = HfFileSystem()

    card = ModelCard.load(pretrained_model_path)
    json_files = fs.glob(f"{pretrained_model_path}/**.json", detail=False)
    if len(json_files) == 0:
        warnings.warn("This model's repository is empty!")
        return None
    else:
        path_to_config = None
        for f in json_files:
            if f.endswith("/config.json"):
                path_to_config = f
                break
        if not path_to_config:
            warnings.warn(
                "This model's repository does not contain any configuration file. We can not find its base model."
            )
            return None
        else:
            pass
        print(f"path_to_config : {path_to_config}")
        with fs.open(f"hf://{path_to_config}", "r") as f:
            config_dict = json.load(f)
        try:
            _name_or_path = config_dict.loc["_name_or_path"]
        except:
            _name_or_path = None

        model_type = config_dict["model_type"]
        models = hf_api.list_models(
            filter=ModelFilter(model_name=pretrained_model_path.split("/")[-1])
        )
        candidate_models = []
        for model in tqdm(models):
            model_name_or_path = model.id
            try:
                with fs.open(f"hf://{model_name_or_path}/config.json", "r") as f:
                    candidate_config_dict = json.load(f)
                print("It worked!")
            except:
                print("NO")
                continue
            if equality(config_dict, candidate_config_dict):
                if verbose:
                    print(
                        f"ACCEPTED {len(candidate_models)}! : model_name_or_path = {model_name_or_path}"
                    )
                candidate_models.append(model_name_or_path)
        # print(f"Here are the candidate models : {candidate_models}")
        print(f"There are {len(candidate_models)} candidate models.")
        return candidate_models


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
        "--output_data_path",
        default="./bert/bert.out",
        help="The path of the json file where we store the fine-tuned models path.",
    )
    return parser.parse_args()


def main(args):
    candidates = get_finetuned_models(args.base_model_name_or_path)
    with open(args.output_data_path, "a") as fout:
        for model_name_or_path in candidates:
            fout.write(json.dumps({"path": model_name_or_path}) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
