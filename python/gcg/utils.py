# NOTE. Code adapted from the following (by @andyzoujm [Andy Zou] and @zifanw505 [Zifan Wang]):
# https://github.com/llm-attacks/llm-attacks

import gc
import re
import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPT2LMHeadModel, GPTJForCausalLM,
                          StableLmForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM)


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad


def log_prob_loss(logits, labels, temp=1):
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    if torch.isnan(logits).any():
        assert False
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits / temp
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def log_perplexity(logits, suffixes, return_all=False):
    shift_suffixes = suffixes[:, 1:]
    shift_logits = logits[:, :shift_suffixes.shape[1], :]
    log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
    stacked_perplexities = torch.stack([log_probs[i, torch.arange(shift_suffixes.shape[1]), shift_suffixes[i]].mean() for i in range(log_probs.shape[0])])
    if return_all:
        return -stacked_perplexities
    else:
        return -stacked_perplexities.mean()


def arca_suffix(model, tokenizer, input_ids, input_slice, target_slice, loss_slice, not_allowed_tokens=None, k=64):
    device = model.device
    embed_weights = get_embedding_matrix(model)

    input_size = len(input_ids)
    vocab_size = embed_weights.shape[0]
    embedding_dim = embed_weights.shape[1]
    targets = input_ids[target_slice]

    curr_toks = input_ids.clone()
    stacked_curr_toks = np.tile(curr_toks.detach().cpu(), (k, 1))
    curr_toks_tensor = torch.Tensor(stacked_curr_toks).long().to(device)
    full_embeds = torch.zeros(k, input_size, embedding_dim, dtype=embed_weights.dtype).to(device)

    # Initialize full embeddings
    for i in range(input_size):
        full_embeds[:, i] = embed_weights[curr_toks[i]].unsqueeze(0).repeat(k, 1)

    # Iterate
    for tok_id in range(input_slice.start, input_slice.stop):
        new_indices = np.random.choice(vocab_size, size=k, replace=True)
        full_embeds[:, tok_id, :] = embed_weights[new_indices, :]

        curr_toks_tensor[:, tok_id] = torch.Tensor(new_indices).long().to(device)

        full_embeds = full_embeds.detach()

        if full_embeds.requires_grad:
            full_embeds.grad.zero_()

        full_embeds.requires_grad = True
        full_embeds.retain_grad()

        logits = model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        loss = nn.CrossEntropyLoss()(logits[:,loss_slice,:].reshape(-1, logits.shape[-1]), targets.repeat(k, 1).view(-1))

        loss.backward(retain_graph=True)

        grad = full_embeds.grad
        scores = -torch.matmul(embed_weights, grad[:,tok_id,:].mean(dim=0))

        best_scores_idxs = scores.argsort(descending=True)
        if not_allowed_tokens is not None:
            best_scores_idxs = filter_forbidden_toks(best_scores_idxs, not_allowed_tokens)

        full_embeds = full_embeds.detach()

        with torch.no_grad():
            full_embeds[:, tok_id, :] = embed_weights[best_scores_idxs[:k], :]

            logits = model(inputs_embeds=full_embeds).logits
            log_probs = nn.functional.log_softmax(logits[:,loss_slice,:], dim=2)

            batch_log_probs = torch.stack([log_probs[i, :, curr_toks_tensor[i, loss_slice]].sum() for i in range(k)])
            best_batch_idx = batch_log_probs.argmax() # TODO: what about flipped?
            best_idx = best_scores_idxs[best_batch_idx]

            curr_toks[tok_id] = best_idx.item()
            full_embeds[:, tok_id, :] = embed_weights[best_idx].unsqueeze(0).repeat(k, 1)

    adv_suffix_tokens = curr_toks[input_slice]
    return tokenizer.decode(adv_suffix_tokens)


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits


def filter_forbidden_toks(toks_tensor, forbidden_toks):
    if len(forbidden_toks) == 0:
        return toks_tensor

    # Toks tensor has all tokens included
    mask = np.zeros(toks_tensor.shape[0])

    # Should try to get the indices where bad things happen...
    mask[forbidden_toks] = 1
    if isinstance(toks_tensor, torch.Tensor):
        elements_ok = np.where(mask[toks_tensor.detach().cpu().numpy()] == 0)[0]
    else:
        elements_ok = np.where(mask[toks_tensor] == 0)[0]

    toks_tensor = toks_tensor[elements_ok]
    return toks_tensor


def standardize(X, dim=0):
    means = torch.mean(X, dim=dim)
    stds = torch.std(X, dim=dim)
    return (X - means) / stds


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None, flipped=False, uniform=True, m_tokens=1):
    if not_allowed_tokens is not None:
        if flipped:
            grad[:, not_allowed_tokens.to(grad.device)] = -np.infty
        else:
            grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    if flipped:
        top_grads, top_indices = (grad).topk(topk, dim=1)
    else:
        top_grads, top_indices = (-grad).topk(topk, dim=1)
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)

    sub_topk = 0
    sub_batch_size = 0
    if m_tokens > 1:
        sub_topk = topk // 2
        sub_batch_size = batch_size // 2
        original_control_toks = control_toks.repeat(sub_batch_size, 1)
        new_control_toks_multiple = original_control_toks.clone()
        for m in range(0, m_tokens):  # Replace 2 to m tokens
            # Generate random token positions within suffix_size
            new_token_pos = torch.randint(0, len(control_toks), (sub_batch_size, m+1), device=grad.device)

            # Select topk indices for each position
            expanded_pos = new_token_pos[:, m].unsqueeze(1).expand(-1, sub_topk)
            batch_indices = torch.arange(sub_batch_size, device=grad.device).unsqueeze(1).expand(-1, sub_topk)
            top_indices_at_pos = top_indices.gather(0, expanded_pos)

            # Randomly select within the sub_topk indices
            sampled_indices = torch.randint(0, sub_topk, (sub_batch_size, 1), device=grad.device)
            new_token_val = torch.gather(top_indices_at_pos, 1, sampled_indices)

            # Scatter the new token values into new_control_toks_multiple at the new_token_pos locations
            new_control_toks_multiple.scatter_(1, new_token_pos[:, m].unsqueeze(1), new_token_val)
    else:
        original_control_toks = control_toks.repeat(batch_size, 1)

    sub_topk = topk - sub_topk
    sub_batch_size = batch_size - sub_batch_size
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / sub_batch_size,
        device=grad.device
    ).type(torch.int64)

    if uniform:
        sampled_indices = torch.randint(0, sub_topk, (sub_batch_size, 1), device=grad.device)
    else:
        w = torch.mean(top_grads, dim=0)
        w = standardize(w)
        w = nn.functional.softmax(w)
        sampled_indices = w.multinomial(sub_batch_size, replacement=True).reshape(sub_batch_size, 1)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        sampled_indices
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    if m_tokens > 1:
        new_control_toks = torch.cat((new_control_toks, new_control_toks_multiple), dim=0)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None, clean_up_tokenization_spaces=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice, control_slice, include_perp=False, lambda_perp=0.1, flipped_perp=False):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    losses = loss.mean(dim=-1)
    if include_perp:
        log_ppls = lambda_perp*log_perplexity(logits, ids[:,control_slice], return_all=True)
        if flipped_perp:
            losses = losses - log_ppls
        else:
            losses = losses + log_ppls
    return losses


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', device_map='auto', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
            **kwargs
        ).eval()
        # ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    use_fast = 'stablelm' in tokenizer_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=use_fast
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    # if gen_config.max_new_tokens > 50:
    #     print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids[assistant_role_slice.stop:]

def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    gen_str = tokenizer.decode(generate(model,
                                        tokenizer,
                                        input_ids,
                                        assistant_role_slice,
                                        gen_config=gen_config)).strip()
    jailbroken = not any([prefix.lower() in gen_str.lower() for prefix in test_prefixes])
    return jailbroken

def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel) or 'BTLMLMHeadModel' in str(type(model)):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM) or isinstance(model, StableLmForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel) or 'BTLMLMHeadModel' in str(type(model)):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM) or isinstance(model, StableLmForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel) or 'BTLMLMHeadModel' in str(type(model)):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM) or isinstance(model, StableLmForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_nonascii_toks(tokenizer, device='cpu', only_word_chars=False, placeholder_token="!"):
    def is_word_char(s):
        if only_word_chars:
            regex = re.compile(rf"^[a-zA-Z0-9_{placeholder_token}\s\.!?,'\"\-\+\*%&@<>]*$")
            return regex.match(s)
        else:
            return True

    def is_ascii(s):
        return s.isascii() and s.isprintable() and is_word_char(s)

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)
