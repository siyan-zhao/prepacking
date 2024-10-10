from typing import List, Union
import torch
from torch import Tensor
from ortools.linear_solver import pywraplp
import binpacking
from transformers import AutoTokenizer
from model import CustomCausalLlamaModel, CustomCausalMistralModel

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LLAMA_ATTENTION_CLASSES,
    apply_rotary_pos_emb,
    logger
)

from model import LlamaFlexAttention


# As implemented here:
# https://github.com/pytorch/pytorch/issues/10536#issuecomment-1320935162
def left_pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> Tensor:

    sequences = tuple(map(lambda s: s.flip(0), sequences))
    padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    _seq_dim = padded_sequence.dim()
    padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    return padded_sequence


def greedy_packing(length_dict, max_bin_size):
    return binpacking.to_constant_volume(length_dict, max_bin_size)


# https://developers.google.com/optimization/pack/bin_packing
def integer_program_packing(length_dict, max_bin_size):
    data = {}
    data["items"] = list(length_dict.keys())
    data["weights"] = list(length_dict.values())
    data["bins"] = data["items"]
    data["bin_capacity"] = max_bin_size

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return
    x = {}
    for i in data["items"]:
        for j in data["bins"]:
            x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))
    y = {}
    for j in data["bins"]:
        y[j] = solver.IntVar(0, 1, "y[%i]" % j)

    for i in data["items"]:
        solver.Add(sum(x[i, j] for j in data["bins"]) == 1)

    for j in data["bins"]:
        solver.Add(sum(x[(i, j)] * data["weights"][i] for i in data["items"]) <= y[j] * data["bin_capacity"])

    solver.Minimize(solver.Sum([y[j] for j in data["bins"]]))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        result = []
        for j in data["bins"]:
            if y[j].solution_value() == 1:
                bin_dict = {}
                for i in data["items"]:
                    if x[i, j].solution_value() > 0:
                        bin_dict[i] = data["weights"][i]
                result.append(bin_dict)
    else:
        raise ("The problem does not have an optimal solution.")

    return result


def load_model_and_tokenizer(
    base_model: str = "llama1b",
    loadbit: int = 8,
    attn_implementation: str = "flex",
):
    # Load tokenizer and model
    if base_model == "llama1b":
        path = "princeton-nlp/Sheared-LLaMA-1.3B"
    elif base_model == "llama2":
        path = "/path/to/llama2"

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = "[PAD]"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_in_8bit = loadbit == 8
    load_in_4bit = loadbit == 4
    if "llama" in base_model:
        # if attn_implementation == "flex":
        #     # LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlexAttention
        #     LLAMA_ATTENTION_CLASSES["sdpa"] = LlamaFlexAttention
        # else:
        #     # LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2
        #     LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaSdpaAttention


        # Always use flex attention:
        LLAMA_ATTENTION_CLASSES["sdpa"] = LlamaFlexAttention
        
        attn_implementation = "sdpa"

        model = CustomCausalLlamaModel.from_pretrained(
            path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, attn_implementation=attn_implementation
        )
    elif "mistral" in base_model:
        model = CustomCausalMistralModel.from_pretrained(
            path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, attn_implementation=attn_implementation
        )
    model.eval()
    if loadbit != 8 and loadbit != 4:
        model.to(device)

    return model, tokenizer
