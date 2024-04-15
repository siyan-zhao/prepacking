import os
from typing import List
import torch
import fire
import random
import time
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
from dataset_utils import (
    PackedDataset,
    sample_batches,
    sample_batches_by_length,
    sample_packed_dataset,
    unpack_kv,
    load_and_evaluate_dataset,
)
from processor import PrePackProcessor
from utils import integer_program_packing, load_model_and_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def prefill_packed_sentence_output(sentences, model, tokenizer, device, processor):
    new_tokens, new_positions, new_mask, restart_dict, original_ids = processor.batch_process(sentences)
    with torch.no_grad():
        packed_outputs = model(
            input_ids=new_tokens,
            attention_mask=new_mask,
            position_ids=new_positions,
            return_dict=True,
        )
    return packed_outputs


def TTFT_packed_sentence_output(sentences, model, tokenizer, device, processor):
    new_tokens, new_positions, new_mask, restart_dict, original_ids = processor.batch_process(sentences)
    with torch.no_grad():
        packed_outputs = model(
            input_ids=new_tokens,
            attention_mask=new_mask,
            position_ids=new_positions,
            return_dict=True,
        )
        cache, final_tokens, attention_mask = unpack_kv(
            packed_outputs["past_key_values"], restart_dict, original_ids, device
        )
        _ = model.generate(
            input_ids=final_tokens,
            attention_mask=attention_mask,
            max_new_tokens=1,
            use_cache=True,
            do_sample=False,
            past_key_values=cache,
        )
    return


def TTFT_packed_dataset_output(batch, model, tokenizer=None, model_device=None, optimized_processor=None):
    new_tokens, new_positions, new_mask, restart_dict, original_ids = batch
    with torch.no_grad():
        packed_outputs = model(
            input_ids=new_tokens,
            attention_mask=new_mask,
            position_ids=new_positions,
            return_dict=True,
        )
        cache, final_tokens, attention_mask = unpack_kv(
            packed_outputs["past_key_values"], restart_dict, original_ids, model_device
        )
        _ = model.generate(
            input_ids=final_tokens,
            attention_mask=attention_mask,
            max_new_tokens=1,
            use_cache=True,
            do_sample=False,
            past_key_values=cache,
        )
    return


def prefill_packed_dataset_output(batch, model, tokenizer=None, device=None, processor=None):

    new_tokens, new_positions, new_mask, restart_dict, original_ids = batch
    with torch.no_grad():
        packed_outputs = model(
            input_ids=new_tokens,
            attention_mask=new_mask,
            position_ids=new_positions,
        )
    return packed_outputs


def prefill_batch_sentence_output(sentences, model, tokenizer, device, processor=None):
    batch_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        batch_sentences_outputs = model(
            batch_sentences["input_ids"].to(device),
            attention_mask=batch_sentences["attention_mask"].to(device),
        )
    return batch_sentences_outputs


def TTFT_batch_sentence_output(sentences, model, tokenizer, device, processor=None):
    batch_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        _ = model.generate(
            **batch_sentences,
            max_new_tokens=1,
            use_cache=True,
            do_sample=False,
        )
    return


def measure_inference_time(
    method,
    texts,
    batch_size,
    num_runs,
    total_batches,
    model,
    tokenizer,
    model_device,
    metric="TTFT",
    binpack_algo="greedy",
):

    if metric == "TTFT":
        method_functions = {
            "prepack": TTFT_packed_sentence_output,
            "full-batching": TTFT_batch_sentence_output,
            "length-ordered": TTFT_batch_sentence_output,
            "prepack_dataset": TTFT_packed_dataset_output,
        }
    elif metric == "prefill":
        method_functions = {
            "prepack": prefill_packed_sentence_output,
            "full-batching": prefill_batch_sentence_output,
            "length-ordered": prefill_batch_sentence_output,
            "prepack_dataset": prefill_packed_dataset_output,
        }
    desc = method
    method_function = method_functions.get(method)
    packing_fn = None if binpack_algo == "greedy" else integer_program_packing
    optimized_processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    total_request_times = []
    for _ in range(num_runs):
        if method == "length-ordered":
            batches_generator = sample_batches_by_length(texts, batch_size)
        elif method == "prepack_dataset":
            new_tokens, new_positions, new_mask, restart_indices, original_ids = (
                optimized_processor.batch_process(texts)
            )
            dataset = PackedDataset(
                new_tokens, new_positions, new_mask, restart_indices, original_ids, batch_size=batch_size
            )

            batches_generator = sample_packed_dataset(dataset)
            del new_tokens, new_positions, new_mask, restart_indices, original_ids
        else:
            batches_generator = sample_batches(texts, batch_size)
        start_time = time.time()
        for batch in tqdm(batches_generator, total=total_batches, desc=desc):
            _ = method_function(batch, model, tokenizer, model_device, optimized_processor)
        elapsed = time.time() - start_time
        total_request_times.append(elapsed)

    per_request_time = np.mean(total_request_times) / (len(texts) * num_runs)
    per_request_time_std = np.std(total_request_times) / (len(texts) * num_runs)
    return per_request_time, per_request_time_std


def main(
    methods: List[str] = ["prepack_dataset", "prepack", "full-batching", "length-ordered"],
    metric: str = "prefill",
    dataset: str = "mmlu",
    model_name: str = "llama1b",
    loadbit: int = 8,
    num_runs: int = 5,
    batch_size: int = 32,
    binpack_algo: str = "greedy",
):

    torch.set_num_threads(5)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if binpack_algo != "greedy":
        binpack_algo = "ip"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model=model_name, loadbit=loadbit)

    # Load and prepare the dataset
    texts = load_and_evaluate_dataset(dataset, tokenizer)

    total_batches = len(texts) // batch_size
    if len(texts) % batch_size != 0:
        total_batches += 1
    table = PrettyTable()

    table.field_names = [
        "Method",
        f"Avg Prefill Time per request (s). bs={batch_size},"
        f"Bits: {loadbit}, {dataset}, {model_name},"
        f"metric: {metric},"
        f"binpack_algo: {binpack_algo}",
        f"std dev over {num_runs} runs",
    ]
    results = {}
    for method in methods:
        avg_time, std = measure_inference_time(
            method,
            texts,
            batch_size,
            num_runs,
            total_batches,
            model,
            tokenizer,
            model.device,
            metric=metric,
            binpack_algo=binpack_algo,
        )
        table.add_row([method, f"{avg_time:.5f}", f"{std:.5f}"])
        results[method] = {
            "Avg Prefill Time per request (s)": avg_time,
            "Std Dev": std,
        }
        print(table)


if __name__ == "__main__":
    fire.Fire(main)
