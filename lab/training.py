import os.path
import argparse
from math import sqrt
import torch as th
from data import MTDataset, MTDataLoader, Vocab
from transformer import Transformer
from tqdm import tqdm


def load_data(cached_data="data/cached.pt", overwrite=False):
    if not os.path.isfile(cached_data) or overwrite:
        vocab = Vocab.from_data_files("data/train.bpe.fr", "data/train.bpe.en")
        train = MTDataset(vocab, "data/train.bpe",
                          src_lang="fr", tgt_lang="en")
        valid = MTDataset(vocab, "data/valid.bpe",
                          src_lang="fr", tgt_lang="en")
        th.save([vocab, train, valid], cached_data)
    # Load cached dataset
    return th.load(cached_data)


def get_args():
    parser = argparse.ArgumentParser("Train an MT model")
    # General params
    parser.add_argument("--model-file", type=str, default="model.pt")
    parser.add_argument("--overwrite-model", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    # Model parameters
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Optimization parameters
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tokens-per-batch", type=int, default=2000)
    parser.add_argument("--samples-per-batch", type=int, default=200)
    return parser.parse_args()


def move_to_device(tensors, device):
    return [tensor.to(device) for tensor in tensors]


def inverse_sqrt_schedule(warmup, lr0):
    step = 0
    # Trick for allowing warmup of 0
    warmup = max(warmup, 0.01)
    while True:
        scale = min(1/sqrt(step+1e-20), step/sqrt(warmup**3))
        step += 1
        yield lr0 * scale


def train_epoch(model, optim, dataloader, lr_schedule=None):
    # Model device
    device = list(model.parameters())[0].device
    # track stats
    stats = {"loss": 10000}
    # Iterate over batches
    itr = tqdm(dataloader)
    for batch in itr:
        itr.total = len(dataloader)
        # Cast input to device
        batch = move_to_device(batch, device)
        # Various inputs
        src_tokens, src_mask, tgt_tokens, tgt_mask = batch
        # Get log probs
        log_p = model(src_tokens, tgt_tokens[:-1], src_mask, tgt_mask[:-1])
        # Loss (this selects log_p[i, b, tgt_tokens[i+1, b]])
        nll = - log_p.gather(-1, tgt_tokens[1:].unsqueeze(-1)).squeeze(-1)
        # Label smoothing
        label_smoothing = - log_p.mean(dim=-1)
        # Final loss at each step
        loss = 0.9 * nll + 0.1 * label_smoothing
        # Reduce (with masking)
        loss_mask = tgt_mask[1:].eq(0).float()
        reduced_loss = (loss * loss_mask).sum() / loss_mask.sum()
        # Backprop
        reduced_loss.backward()
        # Adjust learning rate with schedule
        if lr_schedule is not None:
            learning_rate = next(lr_schedule)
            for param_group in optim.param_groups:
                param_group['lr'] = learning_rate
        # Optimizer step
        optim.step()
        # Update stats
        itr.set_postfix(loss=reduced_loss.item())


def evaluate_ppl(model, dataloader):
    # Model device
    device = list(model.parameters())[0].device
    # total tokens
    tot_tokens = ppl = 0
    # Iterate over batches
    for batch in tqdm(dataloader):
        # Cast input to device
        batch = move_to_device(batch, device)
        # Various inputs
        src_tokens, src_mask, tgt_tokens, tgt_mask = batch
        with th.no_grad():
            # Get log probs
            log_p = model(src_tokens, tgt_tokens, src_mask, tgt_mask)
            # Loss (this selects log_p[i, b, tgt_tokens[i+1, b]] for each batch b, position i)
            nll = - log_p[:-1].gather(-1, tgt_tokens[1:].unsqueeze(-1))
            # Perplexity
            loss_mask = tgt_mask[1:].eq(0).float()
            ppl += th.exp(nll.squeeze(-1) * loss_mask).sum().item()
            # Denominator
            tot_tokens += loss_mask.sum().item()
    return ppl/tot_tokens


def main():
    # Command line arguments
    args = get_args()
    # data
    vocab, train_data, valid_data = load_data()
    # Model
    model = Transformer(
        args.n_layers,
        args.embed_dim,
        args.hidden_dim,
        args.n_heads,
        vocab,
        args.dropout
    )
    if args.cuda:
        model = model.cuda()
    # Load existing model
    if os.path.isfile(args.model_file) and not args.overwrite_model:
        model.load_state_dict(th.load(args.model_file))
    # Optimizer
    optim = th.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # Learning rate schedule
    lr_schedule = inverse_sqrt_schedule(4000, args.lr)
    # Dataloader
    train_loader = MTDataLoader(
        train_data,
        max_bsz=args.samples_per_batch,
        max_tokens=args.tokens_per_batch,
        shuffle=True
    )
    valid_loader = MTDataLoader(
        valid_data,
        max_bsz=args.samples_per_batch,
        max_tokens=args.tokens_per_batch,
        shuffle=False
    )
    # Train epochs
    best_ppl = 10000
    for epoch in range(1, args.n_epochs+1):
        print(f"----- Epoch {epoch} -----")
        # Train for one epoch
        model.train()
        train_epoch(model, optim, train_loader, lr_schedule)
        # Check dev ppl
        model.eval()
        valid_ppl = evaluate_ppl(model, valid_loader)
        print(f"Validation perplexity: {valid_ppl:.2f}")
        # Early stopping maybe
        if valid_ppl < best_ppl:
            print(f"Saving new best model (epoch {epoch} ppl {valid_ppl})")
            th.save(model.state_dict(), args.model_file)


if __name__ == "__main__":
    main()
