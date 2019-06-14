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
    # Go into eval mode (e.g. disable dropout)
    model.eval()
    # Encode source sentece
    src_tensor = th.LongTensor(src_tokens).to(device).view(-1, 1)
    encodings = model.encode(src_tensor)
    # Initialize beams
    beams = [{
        # Tokens generated in this beam
        "tokens": [model.vocab["<sos>"]],
        # Internal decoder state
        "state": model.initial_state(),
        # log probabilityof the sequence
        "log_p": 0,
        # Whether this beam is dead
        "is_over": False,
    }]
    # Start decoding
    eos_token = model.vocab["<eos>"]
    while not beams[-1]["is_over"]:
        beam_candidates = []
        for beam in beams:
            # Ignore dead beams
            if beam["is_over"]:
                continue
            # Input last procuced token
            current_token = th.LongTensor([beam["tokens"][-1]])
            current_token = current_token.view(1, 1).to(device)
            # One step of the decoder
            log_p, new_state = model.decode_step(
                current_token,
                encodings,
                beam["state"]
            )
            # Get topk tokens
            log_p_tokens, top_tokens = log_p.view(-1).topk(beam_size)
            # Append to candidates
            for token, log_p_token in zip(top_tokens, log_p_tokens):
                # Update tokens, state and log_p
                beam_candidate = {
                    "tokens": beam["tokens"] + [token.item()],
                    "state": [h.detach() for h in new_state],
                    "log_p": beam["log_p"] + log_p_token.item(),
                    "is_over": False,
                }
                # check whether this beam is over
                if beam_candidate["tokens"][-1] == eos_token:
                    beam_candidate["is_over"] = True
                # Save candidate
                beam_candidates.append(beam_candidate)
        # Now rerank and keep top beams
        beams = sorted(
            beam_candidates,
            key=lambda beam: beam["log_p"] /  # log probability
            (len(beam["tokens"]))**len_penalty,  # Length penalty
        )[-beam_size:]  # top k
    # Return generated token (idxs) without <sos> and <eos>
    out_tokens = beams[-1]["tokens"][1:]
    if out_tokens[-1] == eos_token:
        out_tokens = out_tokens[:-1]
    return out_tokens
