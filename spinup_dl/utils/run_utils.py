import itertools
import json
import pathlib

from spinup_dl.utils.logx import setup_logger_kwargs


def _short(value):
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value).replace("/", "-").replace(" ", "")


def call_experiment(exp_name, thunk, seed=0, data_dir=None, datestamp=False, **kwargs):
    if isinstance(thunk, str):
        from spinup_dl.models import TRAINERS

        thunk = TRAINERS[thunk]
    kwargs["seed"] = seed
    kwargs["logger_kwargs"] = setup_logger_kwargs(exp_name, seed, data_dir, datestamp)
    print(f"Running experiment {exp_name} with seed {seed}", flush=True)
    return thunk(**kwargs)


class ExperimentGrid:
    def __init__(self, name):
        self.name = name
        self.keys, self.values, self.in_names = [], [], []

    def add(self, key, values, in_name=False):
        if not isinstance(values, list):
            values = [values]
        self.keys.append(key)
        self.values.append(values)
        self.in_names.append(in_name)

    def variants(self):
        for combo in itertools.product(*self.values):
            yield dict(zip(self.keys, combo))

    def variant_name(self, variant):
        parts = [self.name]
        for key, in_name, values in zip(self.keys, self.in_names, self.values):
            if key == "seed":
                continue
            if in_name or len(values) > 1:
                parts.append(f"{key}{_short(variant[key])}")
        return "_".join(parts)

    def print(self):
        print("=" * 60)
        print(f"ExperimentGrid {self.name} runs over:")
        for key, values in zip(self.keys, self.values):
            print(f"  {key}: {json.dumps(values)}")
        print(f"Variants, counting seeds: {len(list(self.variants()))}")
        print("=" * 60, flush=True)

    def run(self, thunk, data_dir=None, datestamp=False):
        self.print()
        results = []
        for variant in self.variants():
            variant = dict(variant)
            seed = variant.pop("seed", 0)
            results.append(call_experiment(self.variant_name({**variant, "seed": seed}), thunk, seed=seed, data_dir=data_dir, datestamp=datestamp, **variant))
        return results
