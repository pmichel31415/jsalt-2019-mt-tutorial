import sys
import argparse
import torch as th
from transformer import Transformer
from decoding import sample
from training import load_data


def get_args():
    parser = argparse.ArgumentParser("Translate with an MT model")
    # General params
    parser.add_argument("--model-file", type=str,
                        default="model.pt", required=True)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--cuda", action="store_true")
    # Model parameters
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Translation parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beam-size", type=float, default=1.0)
    return parser.parse_args()


def move_to_device(tensors, device):
    return [tensor.to(device) for tensor in tensors]


def translate_sentence(model, sentence, beam_size=1, temperature=1.0):
    # Convert string to indices
    src_tokens = [model.vocab[word] for word in sentence]
    # Decode
    with th.no_grad():
        out_tokens = sample(model, src_tokens, temperature)
    # Convert back to strings
    return [model.vocab[tok] for tok in out_tokens]


def main():
    # Command line arguments
    args = get_args()
    # data
    vocab, _, _ = load_data()
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
    model.load_state_dict(th.load(args.model_file))
    # Read from file/stdin
    if args.input_file is not None:
        input_stream = open(args.input_file, "r", encoding="utf-8")
    else:
        input_stream = sys.stdin
    # Write to file/stdout
    if args.output_file is not None:
        output_stream = open(args.output_file, "w", encoding="utf-8")
    else:
        output_stream = sys.stdout
    # Translate
    try:
        for line in input_stream:
            in_words = input_stream.strip().split()
            out_words = translate_sentence(model, in_words)
            print(out_words, file=output_stream)
    except KeyboardInterrupt:
        pass
    finally:
        input_stream.close()
        output_stream.close()


if __name__ == "__main__":
    main()
