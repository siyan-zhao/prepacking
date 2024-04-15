# prepacking

# create env
```conda env create -f environment.yml```


# Profile


## Profile Prefill or Time to First Token (TTFT) Time, and compare peak GPU Memory and Utilization

```python profiling_time_and_memory.py --metric=prefill --dataset=mmlu --batch_size=64 --model_name=llama1b --num_runs=5```

## Compare Per Prompt Inference Prefill Time Including Dataset Prepacking

```python profiling_dataset_level_prepacking.py  --metric=prefill --model_name=llama1b --batch_size=32 --loadbit=8 --dataset=mmlu```

## sanity check generation tokens.

```python generation_example_sanitycheck.py```

