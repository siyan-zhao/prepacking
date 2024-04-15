from transformers import LlamaForCausalLM
import torch
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


torch.set_num_threads(1)


def run_batching_experiment(model, batch_size, length):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenario_times = []
    for _ in range(100):
        data = torch.randint(0, 32000, (batch_size, length)).to(device)
        with torch.no_grad():
            start_time = time.time()
            _ = model(input_ids=data)
            elapsed = time.time() - start_time
            scenario_times.append(elapsed)
    avg_scenario_time = np.mean(scenario_times)
    std_dev = np.std(scenario_times)
    return avg_scenario_time, std_dev


if __name__ == "__main__":
    model_path = "princeton-nlp/Sheared-LLaMA-1.3B"
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    LENGTHS = [50, 100, 200, 400]
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    data_to_save = {}
    for length in tqdm(LENGTHS):
        data_to_save[length] = [run_batching_experiment(model, bs, length) for bs in BATCH_SIZES]
        print(f"Length: {length}", data_to_save[length])

    with open("experiments/batch_size_data.pickle", "wb") as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plotting
    with open("experiments/batch_size_data.pickle", "rb") as handle:
        data = pickle.load(handle)

    for length in LENGTHS:
        d = data[length]
        plt.plot(BATCH_SIZES, [d[0] for d in data[length]], label=f"D={length}")
        plt.fill_between(
            BATCH_SIZES, [d[0] - d[1] for d in data[length]], [d[0] + d[1] for d in data[length]], alpha=0.2
        )
    plt.legend()
    plt.xlabel("Batch size")
    plt.xscale("log", base=2)
    plt.ylabel("TTFT (s)")
    plt.savefig("batch_size_figure.png")
