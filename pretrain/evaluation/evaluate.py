"""GLM zero-shot evaluation."""

import os
import glob
import time
import itertools
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from megatron import get_args, get_tokenizer
from megatron import print_rank_0, print_rank_last
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module


from datasets import build_dataset

from pretrain_glm import model_provider as glm_model_provider
from evaluation.utils import build_data_loader, cond_log_prob, generate_text

def forward_step(model, tokens, position_ids, attention_mask, tokentype_ids,
                 layer_past=None, get_key_value=None, get_embs = False,
                 forward_method_parallel_output=None):
    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor = model(tokens, position_ids, attention_mask,
                          tokentype_ids=tokentype_ids,
                          layer_past=layer_past,
                          get_key_value=get_key_value,
                          get_embs = get_embs,
                          forward_method_parallel_output=forward_method_parallel_output)

    # if get_key_value:
    #     output_tensor, layer_past = output_tensor

    if get_embs:
        logits, output_tensor = output_tensor
    send_forward(output_tensor)
    # breakpoint()
    args.seq_length = orig_seq_length
    # if get_key_value:
    #     return output_tensor, layer_past
    return output_tensor


def get_masks_and_position_ids(tokens, max_len = None):
    context_length = tokens.shape[1]
    attention_mask = torch.ones((1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    
    position_ids = torch.arange(context_length, dtype=torch.long, device=tokens.device)
    block_position_ids = torch.zeros(context_length, dtype=torch.long, device=tokens.device)
    # block_position_ids = torch.concat(
    #     (
    #         torch.zeros(context_length-1, dtype=torch.long, device=tokens.device),
    #         torch.arange(1, 2, dtype=torch.long, device=tokens.device),
    #     ),
    #     dim = -1
    # )
    position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    
    position_ids = position_ids.unsqueeze(0)
    # breakpoint()
    return tokens, attention_mask, position_ids


def get_embeds(args, data, model, tokenizer):
    tokens = tokenizer.tokenize(data)
    # breakpoint()
    # seqs = torch.cuda.LongTensor(
    #     [tokens + [tokenizer.get_command("eos")]],
    #     # [prefix + tokens + [tokenizer.get_command("eos")]],
    #     # device=args.device,
    # )
    seqs = torch.cuda.LongTensor(
        [[13, 10,  1,  2,  7,  9,  6, 13, 15,  1, 10,  2,  1, 19,  1,  6, 12, 14,
            1,  2,  6, 14,  8,  1, 14,  2, 16,  7,  7, 10,  1,  5, 17, 17,  4,  6,
            19,  1, 18, 18,  7,  3,  1,  8,  1,  2,  8,  2, 13,  5, 10, 10,  1, 13,
            2,  1,  1,  2,  6,  7,  1,  5,  5,  2,  7,  1,  1,  5,  2,  4,  7,  7,
            1, 15, 13, 16,  1, 16,  7,  6, 12, 15,  7,  6, 10, 10, 11,  5,  2, 18,
            1, 12, 10,  1,  5,  6,  2, 13,  4,  6,  7,  1,  1, 13,  2, 11,  1,  9,
            10, 13, 11,  1,  6,  1,  7, 10, 12,  2, 17,  1,  6,  4,  1, 16,  2,  8,
            3,  1,  7,  4,  5,  6,  1,  4,  3,  1,  8, 17,  5, 10,  9,  5,  1,  7,
            13,  3,  4,  4,  7,  4,  9,  3, 12,  3, 14, 12,  6,  7,  1,  4, 11,  1,
            3,  6,  6,  2,  4, 16, 19,  1,  6,  8, 16,  1,  6, 18,  3,  7, 11, 19,
            1,  1, 14,  3,  4,  5,  9, 10,  4,  1, 15, 11,  5, 13,  7,  2, 13, 13,
            17,  8,  7, 13,  8, 15, 19, 18,  7,  9, 12, 18, 16,  2,  4,  1,  2,  3,
            9, 10,  5,  6, 12,  1,  5, 11, 18,  4,  1,  7, 18,  2, 15,  2,  8, 18,
            1,  1, 14, 18,  3,  2, 10,  1,  7,  4,  4, 13, 17,  1,  1,  5, 10,  1,
            5, 10,  1, 34]]
    )

    
    batch_size, context_length = seqs.shape
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs, max_len=None)
    tokens = seqs[..., :context_length]
    
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    if dist.get_rank() == 0:
        print(tokens)
    model.eval()
    with torch.no_grad():
        output = forward_step(model, tokens,
                                position_ids,
                                attention_mask,
                                tokentype_ids=False,
                                get_key_value=False,
                                get_embs=True,
                                forward_method_parallel_output=False)
    # breakpoint()
    if mpu.is_pipeline_last_stage():
        assert output is not None
        output = output[:, :context_length - 1, :]
    # breakpoint()
    return output

def main():
    """Main program."""
    args = get_args()
    tokenizer = get_tokenizer()

    assert args.micro_batch_size == 1

    model = get_model(glm_model_provider)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    
    start = time.time()
    raw_text = ["DKATIPSESPFAAAEVADGAIVVDIAKMKYETPELHVKVGDTVTWINREAMPHNVHFVAGVLGEAALKGPMMKKEQAYSLTFTEAGTYDYHCTPHPFMRGKVVVE",
                "MTTTVATDYDNIEIQQQYSDVNNRWDVDDWDNENSSARLFERSRIKALADEREAVQKKTF",
                "GNEYLFQAKDDEEMNTWIQAISSAISSDKHDTSASTQSTPASSRAQTLPTSVVTITSESS",
                "MPPYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGRTLGLYGKDQQEAALVDMVNDGVEDLRCKYISLIYTNYEAGKDDYVKALPGQLKPFETLLSQNQGGKTFIVGDQISFADYNLLDLLLIHEVLAPGCLDAFPLLSAYVGRLSARPKLKAFLASPEYVNLPINGNGKQMPPYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLY"]
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Process: {len(raw_text)} items.")
    for i, data in tqdm(enumerate(raw_text)):
        if i > 0:
            break
        hidden_states = get_embeds(args, data, model, tokenizer).cpu().numpy().astype("float32")
        # if dist.get_rank() == 0:
        #     breakpoint()
        dist.barrier()
        # assert hidden_states.shape[1] == len(data)    
        if dist.get_rank() == 0:
            # breakpoint()
            print(f"idx:{i}, embs: {hidden_states[0][10][:10]}, size: {hidden_states.shape}")
            breakpoint()
            # id2embs[i] = hidden_states[:-1] # get rid of  <EOS>


    dist.barrier()
    print_rank_0(f"done :-), total time: {time.time() - start}")
