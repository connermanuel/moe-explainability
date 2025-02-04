# %%
import typing as tp

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, SwitchTransformersEncoderModel


# %% Load the model and dataset
def load_model_and_tokenizer(model_name, device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SwitchTransformersEncoderModel.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


model_name = "google/switch-base-8"
model, tokenizer = load_model_and_tokenizer(model_name)
data = load_dataset("universal_dependencies", "en_ewt")


# %% Get token probs for all tokens in all layers
def map_input_ids_to_tokens(
    decoded_tokens: list[str], sent_tokens: list[str]
) -> list[int]:
    """Example:
    decoded_tokens = ['Al', '-', 'Z', 'a', 'man']
    sent_tokens = ['Al', '-', 'Zaman']
    output = [0, 1, 2, 2, 2]
    """
    mapping = []
    token_idx = 0
    token = sent_tokens[token_idx]
    token_accum = ""

    for input_id in decoded_tokens:
        token_accum += input_id
        mapping.append(token_idx)
        if token_accum == token:
            token_idx += 1
            if token_idx < len(sent_tokens):
                token = sent_tokens[token_idx]
            token_accum = ""

    return mapping


def iter_sentence(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    datum: dict[str],
    device: str = "cuda",
) -> tp.Iterator[dict[str, str]]:
    """Iterate through the tokens of an entry from UD and return the expert number for each token in each layer."""
    # Tokenize the input sentences
    input_ids = tokenizer(
        datum["text"], return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)

    # Call the encoder model on the entire batch
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, output_hidden_states=True, output_router_logits=True
        )
        router_probs = outputs.router_probs

    # Combine the two lists and iterate every element from each list
    tokenizer_tokens = [tokenizer.decode(input_id) for input_id in input_ids[0]]
    mapping = map_input_ids_to_tokens(tokenizer_tokens, datum["tokens"])
    assert len(mapping) == len(tokenizer_tokens)

    for decoded_token_id, sent_token_id in enumerate(mapping):
        input_id = input_ids[0, decoded_token_id].item()
        if input_id == tokenizer.eos_token_id:
            sent_token_id = -1
            sent_token = "<eos>"
            upos = -1
            xpos = None
        else:
            sent_token = datum["tokens"][sent_token_id]
            xpos = datum["xpos"][sent_token_id]
            upos = datum["upos"][sent_token_id]

        row = {
            "decoded_token_position": decoded_token_id,
            "sent_token_position": sent_token_id,
            "input_id": input_id,
            "decoded_token": tokenizer_tokens[decoded_token_id],
            "sent_token": sent_token,
            "xpos": xpos,
            "upos": upos,
        }
        for layer_num in range(len(router_probs)):
            if layer_num % 2 == 1:
                row[f"expert_id_{layer_num}"] = router_probs[layer_num][1][
                    0, decoded_token_id
                ].item()
        yield row


data_subset = data["train"]
rows = []
for sent_id, datum in tqdm(enumerate(data_subset), total=len(data_subset)):
    for entry in iter_sentence(model, tokenizer, datum):
        entry = {"sentence_id": sent_id, **entry}
        rows.append(entry)


df = pd.DataFrame(rows)
df.to_parquet("ud_train_routes.parquet.gzip", compression="gzip")
# %%
df = pd.read_parquet("ud_routes_train.parquet.gzip")


def int_to_binary_vector(n, length=3):
    return np.array([int(x) for x in f"{n:0{length}b}"])


def create_route_vector(row):
    binary_vectors = []
    for i in range(1, 12, 2):
        expert_id = row[f"expert_id_{i}"]
        binary_vectors.append(int_to_binary_vector(expert_id))
    return np.concatenate(binary_vectors)


df["route_vector"] = df.apply(create_route_vector, axis=1)
df.to_parquet("ud_train_routes.parquet.gzip", compression="gzip")

# %%
