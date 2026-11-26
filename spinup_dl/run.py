import ast
import sys

from spinup_dl.utils.run_utils import ExperimentGrid

USAGE = """Usage: python -m spinup_dl.run MODEL [--exp_name NAME] [--data_dir DIR] [--datestamp] [--ARG VALUE [VALUE ...]] ...

MODEL is one of: mlp, cnn, rnn, transformer, vae, diffusion
Give an argument several values to run a grid of experiments over them, for example:
    python -m spinup_dl.run mlp --hidden_sizes [128] [256,256] --lr 1e-3 3e-4 --seed 0 1 2
Run 'python -m spinup_dl.run MODEL --help' to list the arguments of a model."""


def parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_args(argv):
    args, key = {}, None
    for token in argv:
        if token.startswith("--"):
            key = token[2:]
            args[key] = []
        elif key is None:
            raise SystemExit(f"Unexpected value '{token}': put arguments after a --name.\n\n{USAGE}")
        else:
            args[key].append(parse_value(token))
    return args


def main(argv=None):
    import inspect

    from spinup_dl.models import TRAINERS

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    model, rest = argv[0], argv[1:]
    if model not in TRAINERS:
        raise SystemExit(f"Unknown model '{model}'.\n\n{USAGE}")
    trainer = TRAINERS[model]
    params = inspect.signature(trainer).parameters
    if rest and rest[0] in ("-h", "--help"):
        print(f"Arguments of {model} and their defaults:")
        for name, p in params.items():
            if name != "logger_kwargs":
                print(f"  --{name} {p.default!r}")
        return
    args = parse_args(rest)
    exp_name = args.pop("exp_name", [f"{model}"])[0]
    data_dir = args.pop("data_dir", [None])[0]
    datestamp = "datestamp" in args
    args.pop("datestamp", None)
    unknown = [k for k in args if k not in params]
    if unknown:
        raise SystemExit(f"Unknown argument(s) for {model}: {', '.join(unknown)}. Run 'python -m spinup_dl.run {model} --help'.")
    grid = ExperimentGrid(exp_name)
    for key, values in args.items():
        if not values:
            raise SystemExit(f"--{key} needs at least one value")
        if isinstance(params[key].default, tuple):
            values = [tuple(v) if isinstance(v, list) else v for v in values]
        grid.add(key, values)
    if "seed" not in args:
        grid.add("seed", [0])
    grid.run(trainer, data_dir=data_dir, datestamp=datestamp)


if __name__ == "__main__":
    main()
