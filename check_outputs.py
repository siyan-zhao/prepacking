import torch
from processor import PrePackProcessor
from model import CustomCausalLlamaModel
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from dataset_utils import unpack_kv
from transformers import BitsAndBytesConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LLAMA_ATTENTION_CLASSES,
    apply_rotary_pos_emb,
    logger,
)
from model import LlamaFlexAttention

SEED = 42
set_seed(SEED)
model_path = "princeton-nlp/Sheared-LLaMA-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLAMA_ATTENTION_CLASSES["sdpa"] = LlamaFlexAttention

attn_implementation = "sdpa"
custom_model = CustomCausalLlamaModel.from_pretrained(model_path)
custom_model.to(device)
custom_model.eval()


processor = PrePackProcessor(tokenizer)

# Change to any prompts
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

packed_tokens, restart_positions, document_ids, restart_dict, original_ids = processor.batch_process(
    sentences
)


with torch.no_grad():
    packed_outputs = custom_model(
        input_ids=packed_tokens.to(device),
        # attention_mask=independent_mask.to(device),
        position_ids=restart_positions.to(device),
        return_dict=True,
        output_hidden_states=True,
        document_ids=document_ids.to(device),
    )

cache, final_tokens, attention_mask = unpack_kv(
    packed_outputs["past_key_values"], restart_dict, original_ids, device
)

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


# with torch.no_grad():
normal_tokens_id = tokenizer(sentences, return_tensors="pt", padding=True, truncation=False).to(device)
    # normal_outputs = custom_model(**normal_tokens_id, return_dict=True, output_hidden_states=True)

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


print("Asserting Same Tokens")

print("Prepacked Tokens:", prepack_generated_output.sequences)
print("Default Tokens:", default_generated_output.sequences)

# for default_token in default_generated_output.sequences:
#     print(tokenizer.decode(default_token, skip_special_tokens=True))

# Check tokens
# Note that it is possible to have different generations due to numerical instability
for i, (prepack_token, default_token) in enumerate(
    zip(prepack_generated_output.sequences, default_generated_output.sequences)
):

    prepack = tokenizer.decode(prepack_token[1:])
    default = tokenizer.decode(default_token[attention_mask.shape[-1] :])
    print("-" * 15, "comparing", "-" * 15)
    print("Prepacked", i, ":", prepack)
    print("Default", i, ":", default)

    # assert prepack == default