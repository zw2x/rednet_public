import os
import json
import time
import click
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

from atomtools.folding import AlphaFold3Worker
from atomtools.folding.alphafold3.tpl_utils import parse_mmcif_template_inputs
from atomtools.folding.rosetta import RosettaWorker

# from atomtools.folding.alphafold3.out_utils import parse_confidences


@click.group()
def cli_main():
    """Main entry point for the CLI application.
    1. Run config_fold/config_sel to generate input configurations.
    2. Run run_fold to execute the folding pipeline.
    3. Run run_relax to perform Rosetta relaxation and scoring.
    4. Run qa to evaluate model quality.
    5. Run check_hdimer/sel to check prediction results.
    """
    pass


@dataclass
class InputConfig:
    name: str
    seqs: dict[str, str]
    save_dir: str
    num_seeds: int
    use_template: bool = False
    tpl_cfg: dict | None = None


def process_input_json(in_file, save_dir, tpl_cif_file=None, sel_key="dsn_seq"):
    inputs = json.load(open(in_file, "r"))
    seqs = {}
    tpl_cfg = {}
    for c, chain in inputs["chains"].items():
        seqs[c] = chain[sel_key]
        if not chain["is_dsn"]:
            assert tpl_cif_file is not None, "Template CIF file must be provided for non-DSN chains."
            tpl_c = c
            tpl_cfg[c] = {
                "cif_file": Path(tpl_cif_file),
                "chain_id": tpl_c,
            }

    input_cfg = InputConfig(
        name=inputs["sample_id"],
        seqs=seqs,
        num_seeds=inputs.get("num_seeds", 1),
        save_dir=save_dir,
        use_template=len(tpl_cfg) > 0,
        tpl_cfg=tpl_cfg,
    )

    return input_cfg


@cli_main.command("config_fold")
@click.argument("in_dir", type=click.Path(exists=True))
@click.argument("tpl_dir", type=click.Path(exists=True))
def config_folding_pipeline(in_dir, tpl_dir):
    in_dir = Path(in_dir)
    tpl_dir = Path(tpl_dir)
    ignored = []
    for in_file in in_dir.glob("*.json"):
        _save_dir = in_dir / in_file.stem
        _save_dir.mkdir(exist_ok=True)
        tpl_cif_file = tpl_dir / f"{in_file.stem.split('_')[0]}.cif"
        assert tpl_cif_file.exists(), f"Template CIF file {tpl_cif_file} not found."
        input_cfg = process_input_json(in_file, _save_dir, tpl_cif_file)
        format_inputs(input_cfg)


@cli_main.command("config_sel")
@click.argument("in_dir", type=click.Path(exists=True))
@click.argument("tpl_dir", type=click.Path(exists=True))
def config_sel_folding_pipeline(in_dir, tpl_dir):
    in_dir = Path(in_dir)
    tpl_dir = Path(tpl_dir)
    for in_file in in_dir.glob("*.json"):
        _save_dir = in_dir / in_file.stem
        # _save_dir.mkdir(exist_ok=True)
        tpl_cif_file = tpl_dir / f"{in_file.stem.split('+')[0]}.cif"
        assert tpl_cif_file.exists(), f"Template CIF file {tpl_cif_file} not found."
        off_tpl_cif_file = tpl_dir / f"{in_file.stem.split('+')[1]}.cif"
        assert off_tpl_cif_file.exists(), f"Template CIF file {off_tpl_cif_file} not found."
        input_cfg, off_input_cfg = process_sel_input_json(in_file, _save_dir, tpl_cif_file, off_tpl_cif_file)
        format_inputs(input_cfg)
        format_inputs(off_input_cfg)


def process_sel_input_json(in_file, save_dir, tpl_cif_file, off_tpl_cif_file):
    inputs = json.load(open(in_file, "r"))
    seqs = {}
    tpl_cfg = {}
    for c, chain in inputs["chains"].items():
        seqs[c] = chain["dsn_seq"]
        if not chain["is_dsn"]:
            assert tpl_cif_file is not None, "Template CIF file must be provided for non-DSN chains."
            tpl_c = c
            tpl_cfg[c] = {
                "cif_file": Path(tpl_cif_file),
                "chain_id": tpl_c,
            }

    _save_dir = save_dir / "on_tgt"
    input_cfg = InputConfig(
        name=inputs["sample_id"],
        seqs=seqs,
        num_seeds=inputs.get("num_seeds", 1),
        save_dir=_save_dir,
        use_template=len(tpl_cfg) > 0,
        tpl_cfg=tpl_cfg,
    )

    off_tgt_seqs, off_tgt_tpl_cfg = {}, {}
    for c, chain in inputs["offtgt_chains"].items():
        off_tgt_seqs[c] = chain["dsn_seq"]
        if not chain["is_dsn"]:
            assert off_tpl_cif_file is not None, "Template CIF file must be provided for non-DSN chains."
            tpl_c = c
            off_tgt_tpl_cfg[c] = {
                "cif_file": Path(off_tpl_cif_file),
                "chain_id": tpl_c,
            }

    _off_tgt_save_dir = save_dir / "off_tgt"
    off_input_cfg = InputConfig(
        name=inputs["offtgt_sample_id"],
        seqs=off_tgt_seqs,
        num_seeds=inputs.get("num_seeds", 1),
        save_dir=_off_tgt_save_dir,
        use_template=len(off_tgt_tpl_cfg) > 0,
        tpl_cfg=off_tgt_tpl_cfg,
    )

    return input_cfg, off_input_cfg


@cli_main.command("run_fold")
@click.argument("in_dir", type=click.Path(exists=True))
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--recursive", is_flag=True, default=False, help="Whether to process directories recursively.")
def run_folding_pipeline(in_dir, config_file, recursive):
    cfg = OmegaConf.load(config_file)
    in_dir = Path(in_dir)
    batched_args = []
    for _d in (in_dir.iterdir() if not recursive else in_dir.rglob("*")):
        if not _d.is_dir():
            continue
        input_file = _d / "inputs.json"
        if not input_file.exists():
            continue
        save_d = _d / "af3_results"
        timing_file = save_d / "timings.json"
        if timing_file.exists():
            continue
        batched_args.append([input_file, save_d, timing_file])
    print(f"Running folding pipeline on {len(batched_args)} inputs...")
    model_cfg, pipeline_cfg = cfg.inference, cfg.pipeline
    for input_file, save_d, timing_file in batched_args:
        timings = _run_model(input_file, save_d, model_cfg, pipeline_cfg)
        json.dump(timings, open(timing_file, "w"), indent=4)


def format_inputs(cfg):
    seqs = cfg.seqs
    save_dir = Path(cfg.save_dir)
    num_seeds = cfg.num_seeds
    model_seeds = np.random.randint(0, 2**16, size=num_seeds).tolist()
    inputs = {
        "name": cfg.name,
        "modelSeeds": model_seeds,
        "dialect": "alphafold3",
        "version": 1,
    }
    if cfg.use_template:
        parse_mmcif_template_inputs(inputs, seqs, save_dir, cfg.tpl_cfg)
    input_file = save_dir / "inputs.json"
    assert input_file.exists(), f"Input file {input_file} not found after formatting."
    return input_file


def _run_model(in_file, save_dir, model_cfg, pipeline_cfg):

    assert model_cfg is not None, "Model configuration must be provided."
    timings = {}
    worker_instantiation_start_time = time.time()
    worker = AlphaFold3Worker(model_cfg=model_cfg, pipeline_cfg=pipeline_cfg)
    worker_running_start_time = time.time()
    timings["worker_instantiation_time"] = worker_running_start_time - worker_instantiation_start_time
    worker.run_model(in_file, save_dir, run_inference=True)
    worker_finished_time = time.time()
    timings["worker_running_time"] = worker_finished_time - worker_running_start_time
    return timings





@cli_main.command("run_relax")
@click.argument("df_file", type=click.Path(exists=True))
@click.argument("save_root", type=click.Path())
@click.option("--cfg_file", type=click.Path(exists=True), help="Path to the Rosetta worker configuration file.")
@click.option("--n_max", type=int, default=None, help="Maximum number of models to process.")
def run_rosetta_relax_pipeline_from_parsed_confid_df(df_file, save_root, cfg_file, n_max):
    df = pd.read_csv(df_file)
    worker_cfg = OmegaConf.load(cfg_file)
    save_root = Path(save_root)
    if not save_root.exists():
        save_root.mkdir(parents=True)
    runner = RosettaWorker(worker_cfg)

    batched_args = []
    strict = True
    for i, row in df.iterrows():
        model_cif = Path(row["model_cif"])
        assert model_cif.exists(), f"Model CIF file {model_cif} not found."
        in_dir = model_cif.parents[3]
        # auth_chain_id -> chain_id
        chain_id_mapping = json.load(open(in_dir / "chain_id_remapping.json", "r"))
        dsn_chain_id = chain_id_mapping[row["auth_dsn_chain"]]
        if strict:
            assert len(chain_id_mapping) == 2, f"Expected 2 chains in chain_id_mapping, got {len(chain_id_mapping)}"
            tgt_chain_id = [v for k, v in chain_id_mapping.items() if k != row["auth_dsn_chain"]][0]
            assert dsn_chain_id != tgt_chain_id, "DSN chain ID and target chain ID should be different."
        else:
            tgt_chain_id = chain_id_mapping[row['auth_tgt_chain']]
            
        model_name = row["model_name"]
        save_dir = save_root / model_name

        relaxed_pdb = save_dir / f"relaxed.pdb"
        result_file = save_dir / f"relaxed.json"
        error_file = save_dir / f"relaxation_error.json"
        if relaxed_pdb.exists() and (result_file.exists() or error_file.exists()):
            continue
        timing_file = save_dir / "relaxation_timings.json"
        batched_args.append(
            {
                "relaxed_pdb": relaxed_pdb,
                "result_file": result_file,
                "timing_file": timing_file,
                "model_cif": model_cif,
                "error_file": error_file,
                'dsn_chain_id': dsn_chain_id,
                'tgt_chain_id': tgt_chain_id,
            }
        )
        if n_max is not None and len(batched_args) >= n_max:
            break

    parallel_process(_run_rosetta, batched_args, shared_args={"runner": runner}, n_jobs=NUM_CPUS // 2)


def _run_rosetta(model_cif, relaxed_pdb, result_file, timing_file, error_file, dsn_chain_id, tgt_chain_id, runner):
    t0 = time.time()
    runner.relax_n_score(
        model_cif,
        out_file=relaxed_pdb,
        result_file=result_file,
        error_file=error_file,
        run_relaxation=True,
        dsn_chain_id=dsn_chain_id,
        tgt_chain_id=tgt_chain_id,
    )
    t1 = time.time()
    timings = {"rosetta_relaxation_time": t1 - t0}
    with open(timing_file, "w") as f:
        json.dump(timings, f, indent=4)


import multiprocessing
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from typing import List, Mapping, Any, Callable

# number of cpu cores
NUM_CPUS = multiprocessing.cpu_count()


def shard(vs, num_shards, shard_id):
    return [vs[i] for i in range(shard_id, len(vs), num_shards)]


def batch_wrapper(fn, batched_args, shared_args, use_tqdm: bool = True, tqdm_desc=""):
    results = []
    if use_tqdm:
        pbar = tqdm(batched_args, desc=tqdm_desc)
    else:
        pbar = batched_args
    for args in pbar:
        result = fn(**args, **shared_args)
        results.append(result)
    if all(result is None for result in results):
        return
    return results


def parallel_process(
    fn: Callable,
    batched_args: List[Mapping[str, Any]],
    shared_args: Mapping[str, Any] = {},
    n_jobs: int = NUM_CPUS,
    use_tqdm: bool = False,
):
    """
    Args:
        batched_args: list of batched args
    """
    assert n_jobs > 0, "n_jobs must be greater than 0"
    n_jobs = min(n_jobs, len(batched_args))
    print(f"Running {len(batched_args)} jobs with {n_jobs} processes")

    def _shard(vs):
        return [shard(vs, n_jobs, i) for i in range(n_jobs)]

    delayed_fns = []
    for i, bargs in enumerate(_shard(batched_args)):
        _batched_fn = partial(batch_wrapper, fn, use_tqdm=use_tqdm, tqdm_desc=f"process-{i}")
        delayed_fns.append(delayed(_batched_fn)(bargs, shared_args))
    result = Parallel(n_jobs=n_jobs)(delayed_fns)
    final_results = []
    [final_results.extend(r) for r in result if r is not None]
    return final_results


if __name__ == "__main__":
    cli_main()
