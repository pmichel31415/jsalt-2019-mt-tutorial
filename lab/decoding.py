import torch as th


def sample(model, src_tokens, temperature=1.0, max_len=200, device=None):
    # Either decode on the model's device or specified device
    # (in which case move the model accordingly)
    if device is None:
        device = list(model.parameters())[0].device
    else:
        model = model.to(device)
    # Go into eval mode (e.g. disable dropout)
    model.eval()
    # Encode source sentece
    src_tensor = th.LongTensor(src_tokens).to(device).view(-1, 1)
    encodings = model.encode(src_tensor)
    # Initialize decoder state
    state = model.initial_state()
    # Start decoding
    out_tokens = [model.vocab["<sos>"]]
    eos_token = model.vocab["<eos>"]
    while out_tokens[-1] != eos_token and len(out_tokens) <= max_len:
        current_token = th.LongTensor([out_tokens[-1]]).view(1, 1).to(device)
        # One step of the decoder
        log_p, state = model.decode_step(current_token, encodings, state)
        # Probabilities
        probs = th.exp(log_p / temperature).view(-1)
        # Sample
        next_token = th.multinomial(probs.view(-1), 1).item()
        # Add to the generated sentence
        out_tokens.append(next_token)
    # Return generated token (idxs) without <sos> and <eos>
    out_tokens = out_tokens[1:]
    if out_tokens[-1] == eos_token:
        out_tokens = out_tokens[:-1]
    return out_tokens


def greedy(model, src_tokens, max_len=200, device=None):
    # Either decode on the model's device or specified device
    # (in which case move the model accordingly)
    if device is None:
        device = list(model.parameters())[0].device
    else:
        model = model.to(device)

    # TODO 3: implement greedy decoding
    #
    # (hint: the implementation is very similar to sampling)

    raise NotImplementedError("TODO 3")


def beam_search(
    model,
    src_tokens,
    beam_size=1,
    len_penalty=0.0,
    max_len=200,
    device=None
):
    # Either decode on the model's device or specified device
    # (in which case move the model accordingly)
    if device is None:
        device = list(model.parameters())[0].device
    else:
        model = model.to(device)
    # TODO 4: implement beam search

    # Hints:
    # - For each beam you need to keep track of at least:
    #   1. The previously generated tokens
    #   2. The decoder state
    #   3. The score (log probability of the generated tokens)
    # - Be careful of how many decoding step you need to perform at each step
    # - Think carefuly of the stopping criterion (there are 2)
    # - As a sanity check you can check that setting beam_szie to 1 returns
    #   the same result as greedy decoding
    raise NotImplementedError("TODO 4")
