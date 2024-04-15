import torch
from processor import PrePackProcessor
from model import CustomCausalLlamaModel
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from dataset_utils import unpack_kv
from transformers import BitsAndBytesConfig

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    model_path = "princeton-nlp/Sheared-LLaMA-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "[PAD]"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_model = CustomCausalLlamaModel.from_pretrained(model_path, device_map="auto")
    custom_model.eval()

    processor = PrePackProcessor(tokenizer)

    sentences = [
        "Rescuers are searching for multiple people in the water after Baltimore bridge collapse, report says",
        "Major bridge in Maryland collapses after being hit by a ship",
        "The capital of Germany is",
        "The capital of Spain is",
        "The capital of Greece is",
        "Today I'm going to the",
        "Baltimore Police Department told NBC",
        "My",
        "It",
    ]

    new_tokens, new_positions, new_mask, restart_dict, original_ids = processor.batch_process(sentences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        packed_outputs = custom_model(
            input_ids=new_tokens.to(device),
            attention_mask=new_mask.to(device),
            position_ids=new_positions.to(device),
            return_dict=True,
            output_hidden_states=True,
        )

    cache, final_tokens, attention_mask = unpack_kv(
        packed_outputs["past_key_values"], restart_dict, original_ids, device
    )

    print("Generating Prepacked")
    print("---------------------------")

    prepack_generated_output = custom_model.generate(
        input_ids=final_tokens.to(device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=20,
        use_cache=True,
        do_sample=False,
        past_key_values=cache,
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    print("Generating Default")
    print("---------------------------")

    with torch.no_grad():
        normal_tokens_id = tokenizer(sentences, return_tensors="pt", padding=True, truncation=False).to(
            device
        )
        normal_outputs = custom_model(**normal_tokens_id, return_dict=True, output_hidden_states=True)

    default_generated_output = custom_model.generate(
        **normal_tokens_id,
        max_new_tokens=20,
        use_cache=True,
        do_sample=False,
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True
    )

    attention_mask = normal_tokens_id["attention_mask"]
    idx = 0

    print("Asserting Same Tokens")
    print("---------------------------")

    # Check tokens
    for i, (prepack_token, default_token) in enumerate(
        zip(prepack_generated_output.sequences, default_generated_output.sequences)
    ):

        prepack = tokenizer.decode(prepack_token[1:])
        default = tokenizer.decode(default_token[attention_mask.shape[-1] :])
        print("-" * 15, "comparing", "-" * 15)
        print("Prepacked", i, ":", prepack)
        print("Default", i, ":", default)

        assert prepack == default
