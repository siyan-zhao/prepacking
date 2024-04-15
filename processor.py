from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
from utils import greedy_packing


class PrePackProcessor:
    def __init__(self, tokenizer, packing_fn=None):
        self.tokenizer = tokenizer
        self.pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if packing_fn:
            self.packing_fn = packing_fn
        else:
            self.packing_fn = greedy_packing

    def process(self, length_dict, packing_dict, token_dict):
        new_positions = []
        new_tokens = []
        restart_dict = {0: -1}  # maps restart index -> token_dict index, -1 is a placeholder
        restart_index = 0

        for key in packing_dict:
            new_tokens += token_dict[key][:-1]  # omit final token for generation
            restart_index += length_dict[key] - 1
            new_positions += list(range(length_dict[key] - 1))
            restart_dict[restart_index] = key

        restart_indices = list(restart_dict.keys())
        size = len(new_tokens)
        new_mask = torch.zeros(size, size, device=self.device)

        for i in range(len(restart_indices) - 1):
            start = restart_indices[i]
            end = restart_indices[i + 1]
            new_mask[start:end, start:end] = torch.tril(torch.ones((end - start, end - start)))

        new_tokens = torch.tensor(new_tokens, device=self.device)
        new_positions = torch.tensor(new_positions, device=self.device)
        new_mask = new_mask.clone().detach()

        return new_tokens, new_positions, new_mask, restart_dict

    def batch_process(self, sentences):

        original_ids = self.tokenizer(sentences).input_ids
        token_dict = dict(enumerate(original_ids))
        length_dict = [len(toks) for toks in original_ids]
        length_dict = {index: len(toks) for index, toks in enumerate(original_ids)}

        max_bin_size = max(length_dict.values())
        packing_lst = self.packing_fn(length_dict, max_bin_size)

        batch_new_tokens = []
        batch_new_positions = []
        batch_new_mask = []
        batch_restart_indices = []
        for packing_dict in packing_lst:
            new_tokens, new_positions, new_mask, restart_indices = self.process(
                length_dict, packing_dict, token_dict
            )
            batch_new_tokens.append(new_tokens)
            batch_new_positions.append(new_positions)
            batch_new_mask.append(new_mask)
            batch_restart_indices.append(restart_indices)

        batch_new_tokens = pad_sequence(batch_new_tokens, batch_first=True, padding_value=self.pad_token)
        batch_new_positions = pad_sequence(batch_new_positions, batch_first=True, padding_value=1)

        max_size = max(tensor.shape[1:] for tensor in batch_new_mask)[0]
        padded_masks = [
            F.pad(tensor, (0, max_size - tensor.size(0), 0, max_size - tensor.size(1)))
            for tensor in batch_new_mask
        ]
        batch_new_mask = torch.stack(padded_masks)

        return batch_new_tokens, batch_new_positions, batch_new_mask, batch_restart_indices, original_ids
