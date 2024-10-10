
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast

from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from typing import Optional, Tuple, Union, List
from itertools import chain
import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask
)
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature

import attn_gym.masks as flex_masks

import attn_gym as ag

import torch.nn.functional as F
# from attn_gym.masks import causal_mask
from attn_gym.masks.document_mask import generate_doc_mask_mod

import datasets

from transformers.cache_utils import Cache, StaticCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LLAMA_ATTENTION_CLASSES,
    apply_rotary_pos_emb,
    logger
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from functools import partial

logger = logging.get_logger(__name__)


# attn_mask = torch.ones((4,1,2048,2048), dtype=torch.bool, device='cuda').tril()

# def causal(b, h, q_idx, kv_idx):
#     h_ = h.new_zeros(h.shape)
#     # print(b)  # uncomment this line to make the code work
#     return attn_mask[b][h_][q_idx][kv_idx]

# global flex_attention

# def recompile():

#     flex_attention = torch.compile(flex_attention)

# def causal(b, h, q_idx, kv_idx):
#     return q_idx >= kv_idx

# def causal_document_masking(b, h, q_idx, kv_idx, document_ids):
#     return (q_idx >= kv_idx) and document_ids[b][q_idx] == document_ids[b][kv_idx] 

# def flexible_mask(b, h, q_idx, kv_idx, mask):
#     return mask[b][0][q_idx][kv_idx] # assuming all heads have same mask


import torch


# # def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
# def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
#     """Generates mask mods that apply to inputs to flex attention in the sequence stacked
#     format.

#     Args:
#         mask_mod: The mask mod to apply to the documents
#         offsets: This tensor should be of shape(num_documents + 1)
#             this should contain the cumulative counts of document tokens.
#             e.g. if you have 3 documents of length 2, 4, 3 then
#             offsets = [0, 2, 6, 9]

#     Note:
#         What is the sequence stacked format? When assembling batches of inputs, we
#         take multiple sequences and stack them together to form 1 large sequence. We then
#         use masking to ensure that the attention scores are only applied to tokens within
#         the same document.
#     """
#     document_id = _offsets_to_doc_ids_tensor(offsets)
    

#     def doc_mask_mod(b, h, q_idx, kv_idx):
#         same_doc = document_id[q_idx] == document_id[kv_idx]
#         q_logical = q_idx - offsets[document_id[q_idx]]
#         kv_logical = kv_idx - offsets[document_id[kv_idx]]
#         inner_mask = mask_mod(b, h, q_logical, kv_logical)
#         return same_doc & inner_mask

#     return doc_mask_mod

def mask_to_doc_ids(mask):
    """
    Assigns document IDs to a causal block diagonal mask.

    Args:
    mask: A causal block diagonal mask of shape (batch, n, n).

    Returns:
    A tensor of document IDs of shape (batch, n).
    """

    # batch_size, n = mask.shape[:2]

    batch_size = mask.shape[0]
    n = mask.shape[1]

    # print("batch size", batch_size, n)

    # Sum along the last dimension
    sums = mask.sum(dim=-1)  # Shape (batch, n)

    # print("SUMS", sums.shape, sums)
    # print(sums[0])
    # print(sums[1])
    
    # Create a tensor to store document IDs
    doc_ids = torch.zeros((batch_size, n), dtype=torch.long, device=mask.device)

    # Assign document IDs
    for batch_idx in range(batch_size):
        prev_idx = 0
        doc_id = 0
        restart_indices = torch.where(sums[batch_idx] == 1)[0]  # Indices where a new block starts
        # print("RESTART INDICES", restart_indices.shape, restart_indices)
        for idx in restart_indices:
            doc_ids[batch_idx, prev_idx:idx] = doc_id
            prev_idx = idx
            doc_id += 1

        doc_ids[batch_idx, prev_idx:] = doc_id  # Assign the last document ID

    return doc_ids



# Code snippet from https://gist.github.com/why-in-Shanghaitech/8b8205f98568c6741a2e38dfcdb9d362
class LlamaFlexAttention(LlamaAttention):

    

    """
    Llama flex attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flex attention and deal with padding tokens in case the input contains any of them.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        flex_mask=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        global flex_attention 
        # if isinstance(past_key_value, StaticCache):
        #     raise ValueError(
        #         "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
        #         "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        #     )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        print("APPLYING FLEX ATTENTION")
        print("Query states shape: ", query_states.shape)
        print("Key states shape: ", key_states.shape)

        q_len = query_states.shape[-2]
        kv_len = key_states.shape[-2]

        # print("Q_LEN", q_len)

        if flex_mask is None:

            print("USING SPDA")

            sdpa_mask = attention_mask
            if attention_mask is not None and cache_position is not None:
                sdpa_mask = sdpa_mask[:, :, cache_position, :key_states.shape[-2]]

            attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            dropout_p=self.attention_dropout if self.training else 0.0)
        else:
            
            

            # print("ATTENTION")
            # print(attention_mask.shape, attention_mask)

            # # Apply the masks to replace the values

            # new_mask = attention_mask.clone()
            # new_mask[attention_mask == attention_mask.min()] = 0
            # new_mask[attention_mask == 0] = 1
            # new_mask = new_mask[:, :, :kv_len, :kv_len].bool()
            # new_mask = new_mask.bool()


            # document_ids = mask_to_doc_ids(new_mask.squeeze())

            # print("DOCUMENT IDS", document_ids)


            # # document_ids = torch.arange(0, kv_len, device=new_mask.device).unsqueeze(0).expand(bsz, -1)
            
            # # print("NEW ATTENTION")
            # # print(new_mask.shape, new_mask) 

            # def causal_document_masking(b, h, q_idx, kv_idx):
            #     return (q_idx >= kv_idx) and document_ids[b][q_idx] == document_ids[b][kv_idx] 
            
            # block_mask = create_block_mask(partial(causal_document_masking, document_ids=document_ids), B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len)

            # block_mask = create_block_mask(causal_document_masking, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len)

            # block_mask = create_block_mask(partial(flexible_mask, mask=new_mask), B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len)



            print("USING FLEX ATTENTION")
        
            attn_output = flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=flex_mask,
                enable_gqa=True,
            )
        # import pdb; pdb.set_trace()

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# Custom model to allow passing of 2D attention masks 
class CustomLlamaModel(transformers.LlamaModel):
    def __init__(self, config):
        super().__init__(config)

    # CUSTOM FORWARD FUNCTION
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # global flex_attention

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None


        # Create Flex Mask:

        # flex_attention = torch.compile(flex_attention)

        # print("ATTENTION")
        # print(attention_mask.shape, attention_mask)

        # Apply the masks to replace the values

        # new_mask = attention_mask.clone()
        # new_mask[attention_mask == attention_mask.min()] = 0
        # new_mask[attention_mask == 0] = 1
        # new_mask = new_mask[:, :, :kv_len, :kv_len].bool()

        # print("NEW MASK")
        # print(new_mask.shape, new_mask)

        document_ids = mask_to_doc_ids(attention_mask)

        print("DOCUMENT IDS", document_ids.shape, document_ids)


        # document_ids = torch.arange(0, kv_len, device=new_mask.device).unsqueeze(0).expand(bsz, -1)
            
            # print("NEW ATTENTION")
            # print(new_mask.shape, new_mask) 

        # def causal_document_masking(b, h, q_idx, kv_idx):
        #     return (q_idx >= kv_idx) and document_ids[b][q_idx] == document_ids[b][kv_idx] 

        bsz, q_len, _ = hidden_states.size()

        # print("Q_LEN", q_len)
        # from attn_gym.masks.document_mask import length_to_offsets

        # # lengths = [100,100,100,100]

        # lengths = [q_len // 5] * 5
        # lengths[-1] += q_len % 5

        

        # offsets = length_to_offsets(lengths, device=hidden_states.device)

        # causal_document_masking = generate_doc_mask_mod(ag.masks.causal_mask, offsets)
        
        
        # def make_tensor():
        #     return torch.ones(1, 1, q_len, 8, device=hidden_states.device)

        # query, key = make_tensor(), make_tensor()
        # # document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)

        # ag.utils.visualize_attention_scores(
        #     query,
        #     key,
        #     mask_mod=causal_document_masking,
        #     device=hidden_states.device,
        #     name="document_causal_mask",
        # )

        document_id = mask_to_doc_ids(attention_mask.squeeze())

        print("DOCUMENT IDS", document_id.shape)
        print(document_id)


        print("DOCUMENT ID LENGTH", len(document_id))
        print("Q_LEN", q_len)

        def round_up_to_multiple(x, multiple):
            return (x + multiple - 1) // multiple * multiple
        
        desired_length = round_up_to_multiple(q_len, 128)

        print("DESIRED", desired_length)

        document_id = F.pad(input=document_id, pad=(0, desired_length - q_len), mode='constant', value=-1)

        print("DOCUMENT ID LENGTH", document_id.shape)
        print("NEW", document_id)

        def document_causal_mask(b, h, q_idx, kv_idx):

            # print("Q_IDX", q_idx)
            # print("KV_IDX", kv_idx)
            # print("DOCUMENT ID", document_id)
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[b][q_idx] == document_id[b][kv_idx]
            return causal_mask & document_mask


        if q_len == 1: # not prefilling
            flex_mask = None
        else:

            global flex_attention
            flex_attention = torch.compile(flex_attention)
            flex_mask = create_block_mask(document_causal_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    flex_mask=flex_mask
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full(
                (2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]),
                fill_value=1,
            )
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        # We use the current dtype to avoid any overflows
        min_dtype = torch.finfo(dtype).min
        causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.to(dtype=dtype, device=device)
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length, :mask_length].eq(0.0) * attention_mask[
                :, None, :, :
            ].eq(0.0)
            causal_mask[..., :mask_length, :mask_length] = causal_mask[
                ..., :mask_length, :mask_length
            ].masked_fill(padding_mask, min_dtype)

        if self.config._attn_implementation == "sdpa" and attention_mask is not None:
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)).to(
                    dtype
                )

        return causal_mask


class CustomCausalLlamaModel(transformers.LlamaForCausalLM, GenerationMixin):
    
    def __init__(self, config):

        print(config)

        super().__init__(config)

        self.model = CustomLlamaModel(config)





class CustomMistralModel(transformers.MistralModel):
    def __init__(self, config):
        super().__init__(config)

    # @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            if attention_mask.shape[1] == 1:  # for use cache
                attention_mask = attention_mask.view(batch_size, -1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.view(batch_size, 1, seq_length, seq_length)
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class CustomCausalMistralModel(transformers.MistralForCausalLM, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMistralModel(config)
