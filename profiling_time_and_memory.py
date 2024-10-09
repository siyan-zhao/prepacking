import os
import torch
import threading
import time
import GPUtil
import fire
import random
from tqdm import tqdm
import numpy as np
from typing import List
from prettytable import PrettyTable
from dataset_utils import (
    load_and_evaluate_dataset,
    sample_batches,
    sample_batches_by_length,
    unpack_kv,
)
from processor import PrePackProcessor
from utils import integer_program_packing, load_model_and_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def monitor_gpu_utilization(stop_event, utilization_stats, device_id=0, interval=0.1):
    max_utilization, total_utilization, count = 0, 0, 0
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    visible_device_ids = list(map(int, cuda_visible_devices.split(",")))
    while not stop_event.is_set():
        gpus = GPUtil.getGPUs()
        current_utilization = gpus[visible_device_ids[device_id]].load
        max_utilization = max(max_utilization, current_utilization)
        total_utilization += current_utilization
        count += 1
        time.sleep(interval)
    utilization_stats["max_util"] = max_utilization * 100  # Convert to percentage
    utilization_stats["mean_util"] = (total_utilization / count) * 100 if count > 0 else 0


def prefill_with_prepacking(sentences, model, tokenizer, device, processor):
    new_tokens, new_positions, new_mask, restart_dict, original_ids = processor.batch_process(sentences)
    with torch.no_grad():
        output = model(
            input_ids=new_tokens,
            attention_mask=new_mask,
            position_ids=new_positions,
            return_dict=True,
        )
    return output


def TTFT_with_prepacking(sentences, model, tokenizer, device, processor):
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


def prefill_with_baseline(sentences, model, tokenizer, device, processor=None):
    batch_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        batch_sentences_outputs = model(
            batch_sentences["input_ids"].to(device),
            attention_mask=batch_sentences["attention_mask"].to(device),
            return_dict=True,
        )
    return batch_sentences_outputs


def TTFT_with_baseline(sentences, model, tokenizer, device, processor=None):
    batch_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        _ = model.generate(
            **batch_sentences,
            max_new_tokens=1,
            use_cache=True,
            do_sample=False,
        )
    return


def get_average_gpu_utilization():
    # get current device id, assuming only 1 GPU is used
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    visible_device_ids = list(map(int, cuda_visible_devices.split(",")))
    gpus = GPUtil.getGPUs()
    return gpus[visible_device_ids[0]].load


def measure_inference_resources(
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
    scenario_times = []

    if metric == "TTFT":
        method_functions = {
            "prepacking": TTFT_with_prepacking,
            "full-batching": TTFT_with_baseline,
            "length-ordered": TTFT_with_baseline,
        }
    elif metric == "prefill":
        method_functions = {
            "prepacking": prefill_with_prepacking,
            "full-batching": prefill_with_baseline,
            "length-ordered": prefill_with_baseline,
        }
    method_function = method_functions.get(method)
    packing_fn = None if binpack_algo == "greedy" else integer_program_packing
    optimized_processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    method_function = method_functions.get(method)

    for _ in range(num_runs):
        batches_generator = (
            sample_batches(texts, batch_size)
            if method != "length-ordered"
            else sample_batches_by_length(texts, batch_size)
        )

        max_gpu_utilization = []
        max_gpu_memory = []
        batch_gpu_memories = []
        batch_gpu_utilizations = []
        mean_gpu_utilizations = []
        for batch in tqdm(batches_generator, total=total_batches, desc=method):
            utilization_stats = {}
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_gpu_utilization, args=(stop_event, utilization_stats), daemon=True
            )
            monitor_thread.start()

            torch.cuda.reset_peak_memory_stats(model_device)  # Reset memory stats at the start
            torch.cuda.empty_cache()

            start_time = time.time()
            print("TESTING")
            print("NEW ITERATION!!!")
            print()
            _ = method_function(batch, model, tokenizer, model_device, optimized_processor)
            elapsed = time.time() - start_time

            scenario_times.append(elapsed)
            stop_event.set()
            monitor_thread.join()
            peak_memory = torch.cuda.max_memory_allocated(model_device) / (1024**2)
            batch_gpu_memories.append(peak_memory)

            max_util = utilization_stats.get("max_util", 0)
            mean_util = utilization_stats.get("mean_util", 0)  # Get mean utilization
            batch_gpu_utilizations.append(max_util)
            mean_gpu_utilizations.append(mean_util)

        max_gpu_memory.append(max(batch_gpu_memories))
        max_gpu_utilization.append(max(batch_gpu_utilizations))
    avg_scenario_time = np.mean(scenario_times)
    avg_gpu_utilization = np.mean(max_gpu_utilization)
    avg_gpu_memory = np.mean(max_gpu_memory)
    avg_mean_gpu_utilization = np.mean(mean_gpu_utilizations)
    std_dev_time = np.std(scenario_times)
    std_gpu_utilization = np.std(max_gpu_utilization)
    std_gpu_memory = np.std(max_gpu_memory)  # = 0
    std_mean_gpu_utilization = np.std(mean_gpu_utilizations)
    return (
        avg_scenario_time,
        avg_gpu_utilization,
        avg_gpu_memory,
        avg_mean_gpu_utilization,
        std_dev_time,
        std_gpu_utilization,
        std_mean_gpu_utilization,
    )


def main(
    methods: List[str] = [
        "prepacking",
        # "full-batching",
        # "length-ordered",
    ],
    metric: str = "TTFT",
    dataset: str = "mmlu",
    model_name: str = "llama1b",
    loadbit: int = 4,
    num_runs: int = 5,
    batch_size: int = 64,
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
        f"Avg {metric} Time /batch (s)",
        "Max GPU Utilization (%)",
        "Max GPU Memory (MB)",
        "Mean GPU Utilization (%)",
        "Std Dev Time (s)",
        "Std Dev Max GPU Util (%)",
        "Std Dev Mean GPU Util (%)",
    ]

    for method in methods:
        try:
            (
                avg_scenario_time,
                avg_gpu_utilization,
                avg_gpu_memory,
                avg_mean_gpu_utilization,
                std_dev_time,
                std_gpu_util,
                std_mean_gpu_util,
            ) = measure_inference_resources(
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
            table.add_row(
                [
                    method,
                    f"{avg_scenario_time:.3f}",
                    f"{avg_gpu_utilization:.3f}",
                    f"{avg_gpu_memory:.3f}",
                    f"{avg_mean_gpu_utilization:.3f}",
                    f"{std_dev_time:.3f}",
                    f"{std_gpu_util:.3f}",
                    f"{std_mean_gpu_util:.3f}",
                ]
            )
            print(table)
        except Exception as e:  # OOM error
            print(f"An error occurred while processing method {method}: {e}")
            torch.cuda.empty_cache()

        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)
