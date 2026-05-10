import click
import dotenv

dotenv.load_dotenv()
import json
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, asdict
import math
import hydra
import collections
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from faust.tokenizer import Tokenizer
from atomtools.redesign.redesign_worker import RedesignWorker

from rednet.data import SkempiDataset
from rednet.sampling_utils import contrast_decode_batch
from rednet.lightning.base_task import build_task
from rednet.common_utils import get_logger, move_to_cuda

log = get_logger(__name__)


@click.group()
def cli_main():
    """
    check: check test set schedule
    run: run inference pipeline
        * run: evaluate
        * run_hdimer: heterodimer self-consistency test
        * run_sel: selective binder design
    """
    pass


def get_sharded_data(cfg, shard_id=0, num_shards=1, sel_sample_ids=None, sel_ids=None, **kwargs):
    dataset = build_dataset(cfg.data, selected_ids=sel_ids, selected_sample_ids=sel_sample_ids, **kwargs)
    dataloader = dataset.get_dataloader(deterministic_sampling=True, rank=shard_id, num_replicas=num_shards, **kwargs)
    return dataset, dataloader


def build_dataset(ds_cfg, selected_sample_ids=None, selected_ids=None, **kwargs):
    dataset_cls = hydra.utils.get_class(ds_cfg._target_)
    print(f"Building dataset: {dataset_cls.__name__}")
    dataset = dataset_cls(
        ds_cfg.config,
        **ds_cfg.get("extra_args", {}),
        selected_ids=selected_ids,
        selected_sample_ids=selected_sample_ids,
        **kwargs,
    )
    return dataset


@torch.no_grad()
def run_logits_sampling(model, sample, valid=False, ignore_prefix=False):
    sample = move_to_cuda(sample)
    if valid:
        out = model.score(sample)
    else:
        if ignore_prefix:
            sample.pop("res_type")
        start = time.perf_counter()
        gen_out = model.sample(sample)
        elapsed = time.perf_counter() - start
        gen_out["timing"] = elapsed
        start = time.perf_counter()
        score_out = model.score(sample)
        elapsed = time.perf_counter() - start
        score_out["timing"] = elapsed
        out = {"generation": gen_out, "scoring": score_out}

    return out


@cli_main.command("check")
@click.option("--config_file", type=click.Path(exists=True))
@click.option("--repr_file", type=click.Path(), default=None)
@click.option("--pdb_df_file", type=click.Path(), default=None)
@click.option("--output_file", type=click.Path())
@click.option(
    "--ds_type",
    type=click.Choice(["pdb_test", "pdb_hdimer", "megascale", "skempi"]),
    required=True,
    help="Type of dataset to check.",
)
def check_dataset_schedule(config_file, repr_file, pdb_df_file, output_file, ds_type):
    def _check_date(date_str):
        d = datetime.fromisoformat(date_str)
        return d > datetime(2022, 12, 31) and d < datetime(2024, 1, 1)

    repr_ids, sel_sample_ids = None, None
    if repr_file:
        repr_df = pd.read_csv(repr_file)
        mask = repr_df.apply(lambda row: _check_date(row["release_date"]), axis=1)
        repr_df = repr_df[mask]
        repr_ids = set(
            [v.split("-")[0].lower() + "-assembly" + v.split("-")[1] for v in repr_df["assembly_id"].tolist()]
        )
    elif pdb_df_file:
        pdb_df = pd.read_csv(pdb_df_file)
        sel_sample_ids = set([v for v in pdb_df["sample_id"].tolist()])

    # dataset and sharded dataloader (one shard per GPU)
    cfg = OmegaConf.load(config_file)
    dataset, dataloader = get_sharded_data(cfg, return_sampler=True, sel_ids=repr_ids, sel_sample_ids=sel_sample_ids)

    if ds_type in {"pdb_test", "pdb_hdimer", "pdb_sel"}:
        df = collections.defaultdict(list)
        for i in dataloader:
            _id = dataset.ids[i]
            _entry = dataset.metadata[_id]

            def _add_val(val):
                if val is None:
                    return 0
                if isinstance(val, tuple):
                    val = "_".join(val)
                return val

            for k, v in _entry.items():
                df[k].append(_add_val(v))
        df = pd.DataFrame(df)
        print(df["interface_type"].value_counts())
    else:
        print("Dataset size:", len(dataset), type(dataset))

    if ds_type == "pdb_test":
        # randomly sample 300 for monomer, 150 for heterodimer, and 150 for homodimer
        seed = 42
        selected_samples = []
        for interface_type, group in df.groupby("interface_type"):
            if interface_type == "none":
                sampled = group.sample(n=min(300, len(group)), random_state=seed)
            else:
                sampled = group.sample(n=min(150, len(group)), random_state=seed)
            selected_samples.append(sampled)
        df = pd.concat(selected_samples, axis=0)
        print(df["interface_type"].value_counts())
        if output_file:
            df.to_csv(output_file, index=False)
    elif ds_type == "pdb_hdimer":
        hdimer_df = df[(df["interface_type"] == "hetero") & (df["sample_size"] <= 500)]
        print("Heterodimer with less than 500 residues:", len(hdimer_df))
        if output_file:
            hdimer_df.to_csv(output_file, index=False)
    elif ds_type == "skempi":
        print("Skempi samples:", len(dataset))
        for i in dataloader:
            print(dataset[i])
            break


def get_sample_ids(df):
    # sample_ids = set([v.split('_')[0] for v in df['sample_id'].tolist()])
    sample_ids = set([v for v in df["sample_id"].tolist()])
    return sample_ids


@dataclass
class SamplingConfig:
    temperature: float = 1e-3
    alpha: float = 0
    beta: float = 0.9


def get_sampling_config(sampling_config_file):
    sampling_cfg = SamplingConfig(temperature=1e-3, alpha=0.0, beta=0.0)
    if sampling_config_file:
        _sampling_cfg = OmegaConf.load(sampling_config_file)
        sampling_cfg = SamplingConfig(
            temperature=_sampling_cfg.temperature,
            alpha=_sampling_cfg.alpha,
            beta=_sampling_cfg.beta,
        )
    return sampling_cfg


@cli_main.command("check_sel")
@click.option("--config_file", type=click.Path(exists=True))
@click.option("--repr_file", type=click.Path(), default=None)
def check_sel_dataset(config_file, repr_file):
    cfg = OmegaConf.load(config_file)
    dataset, dataloader = get_sharded_data(cfg, shard_id=0, num_shards=1, use_pfeats=True)
    print(len(dataset))
    # _keys = ['lig', 'rec', 'assessment', 'off_tgt_id', 'on_tgt_id', 'lig_aligned_out', 'rec_aligned_out']
    df = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # sample_id, off_sample_id = sample['sample_id'], sample['off_sample_id']
        off_tgt_id = sample["off_tgt_id"]
        on_tgt_id = sample["on_tgt_id"]
        rec_seqid = float(sample["rec_aligned_out"]["aln_seqid"])
        jaccard_index = float(sample["assessment"]["jaccard_index"])
        rmsd = float(sample["rec"]["rmsd"])
        tmscore = float(sample["rec"]["tm_score_1"])
        df.append(
            {
                "on_tgt_id": on_tgt_id,
                "off_tgt_id": off_tgt_id,
                "rec_seqid": rec_seqid,
                "jaccard_index": jaccard_index,
                "rmsd": rmsd,
                "tmscore": tmscore,
            }
        )
    df = pd.DataFrame(df)
    df["jaccard_bin"] = pd.cut(df["jaccard_index"], bins=np.arange(0, 1.1, 0.1), right=False)
    sample_inds = []
    for i, grp in df.groupby("jaccard_bin"):
        sampled_idx = grp.sample(min(len(grp), 20)).index
        sample_inds.extend(sampled_idx.tolist())
    df = df.loc[sample_inds]
    if repr_file:
        df.to_csv(repr_file, index=False)


@cli_main.command("run_sel")
@click.option("--config_file", type=click.Path(exists=True))
@click.option("--ckpt_file", type=click.Path())
@click.option("--check_fixed", is_flag=True, default=False, help="Whether to check fixed residues.")
@click.option("--use_cpu", is_flag=True, default=False, help="Whether to use CPU for inference.")
@click.option("--sampling_config_file", type=click.Path(), default=None)
@click.option("--output_dir", type=click.Path(), default=None)
@click.option("--repr_file", type=click.Path(), default=None)
@click.option('--native', is_flag=True, default=False, help="Whether to use native sequences for unbound chains.")
def run_seletive_binder_design_pipeline(
    config_file, ckpt_file, check_fixed, use_cpu, sampling_config_file, output_dir, repr_file, native
):
    if not native:
        ckpt_file = Path(ckpt_file)
        is_benchmark = ckpt_file.suffix == ".yaml"
        if is_benchmark:
            task = RedesignWorker(OmegaConf.load(ckpt_file))
            task.load_ckpt()
        else:
            task = build_task(ckpt_file)
    else:
        is_benchmark = False

    cfg = OmegaConf.load(config_file)

    sel_ids = None
    if repr_file:
        repr_df = pd.read_csv(repr_file)
        sel_ids = []
        for i, row in repr_df.iterrows():
            _id = ":".join(eval(row["on_tgt_id"])) + "+" + ":".join(eval(row["off_tgt_id"]))
            sel_ids.append(_id)
        print(f"Selected {len(sel_ids)} sample ids for selective binder design. {list(sel_ids)[:5]}")

    dataset, dataloader = get_sharded_data(cfg, shard_id=0, num_shards=1, sel_ids=sel_ids)
    print(len(dataset))
    if not native:
        model = task.model.model
        if not use_cpu:
            model.cuda()
        model.eval()

    _use_offtgt = not check_fixed

    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    sampling_cfg = get_sampling_config(sampling_config_file)
    print(sampling_cfg)
    for batch in tqdm(dataloader):
        if not use_cpu:
            batch = move_to_cuda(batch)
        # batch["input_tokens"] = batch["res_type"]
        # get off-target batch
        off_batch = {k.split("off_")[-1]: v for k, v in batch.items() if k.startswith("off_")}
        # get on-target batch
        batch = {k: v for k, v in batch.items() if not k.startswith("off_")}

        if native:
            pred_tokens = batch["res_type"][0]
            off_dsn_mask = off_batch['dsn_mask'][0].bool()
            off_pred_tokens = off_batch["res_type"][0].clone()
            off_pred_tokens[off_dsn_mask] = pred_tokens[batch['dsn_mask'][0].bool()]
        elif is_benchmark:
            gt_res_type = batch["res_type"].clone()
            pred_tokens = run_benchmark(task, batch, temperature=sampling_cfg.temperature, evaluation=False)[
                "generation"
            ]["pred_tokens"][0].to(device=gt_res_type.device)
            batch["res_type"] = gt_res_type
            off_mask = off_batch['dsn_mask'][0].bool()
            _pred_tokens = pred_tokens[batch['dsn_mask'][0].bool()]
            assert _pred_tokens.shape[0] == off_mask.sum().item(), f"Mismatch in designed residues {_pred_tokens.shape[0]} vs {off_mask.sum().item()}"
            off_pred_tokens = off_batch['res_type'][0].clone()
            off_pred_tokens[off_mask] = _pred_tokens
        else:
            pred_tokens, off_pred_tokens = contrast_decode_batch(
                batch, model, sampling_cfg, con_batch=off_batch, check_fixed=check_fixed, use_con=_use_offtgt
            )

        output = make_sel_outputs(batch, off_batch, pred_tokens, off_pred_tokens, translate_fn)

        if output_dir:
            sample_id, off_sample_id = batch["sample_id"][0], off_batch["sample_id"][0]
            output_file = output_dir / f"{sample_id}+{off_sample_id}.json"
            output["hparams"] = asdict(sampling_cfg)
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
        else:
            print(output)


def _parse_output(batch, pred_tokens, translate_fn, replace_pred_with_gt=False):
    # on target chains
    chain_id_mapping = batch["chain_id_mapping"][0]
    chain_index = batch["chain_index"][0]
    gt_res_type = batch["res_type"][0]
    dsn_mask = batch["dsn_mask"][0]
    dsn_seq = None

    output = {}
    for chain_id, i in chain_id_mapping.items():
        _chain_mask = chain_index == i
        _dsn_mask = dsn_mask[_chain_mask]
        num_res = _chain_mask.sum().item()
        same_res = pred_tokens[chain_index == i] == gt_res_type[chain_index == i]
        nsr = same_res.sum().item() / num_res

        num_dsn_res = _dsn_mask.sum().item()

        pred_seq = translate_fn(pred_tokens[chain_index == i])
        gt_seq = translate_fn(gt_res_type[chain_index == i])

        _is_dsn = torch.any(_dsn_mask).item()
        if _is_dsn:
            dsn_seq = pred_seq
        else:
            _unmatched_res_types = set([gt_seq[i] for i in range(len(gt_seq)) if pred_seq[i] != gt_seq[i]])
            if set(_unmatched_res_types) - {"X"}:
                assert (
                    pred_seq == gt_seq
                ), f"Non-designed chains must match ground truth sequence. {chain_id}, {pred_seq}, {gt_seq}"
            elif replace_pred_with_gt:
                pred_seq = gt_seq

        output[chain_id] = {
            "is_dsn": _is_dsn,
            "num_res": num_res,
            "num_dsn_res": num_dsn_res,
            "nsr": nsr,
            "dsn_seq": pred_seq,
            "gt_seq": gt_seq,
        }
    assert dsn_seq is not None, "No designed residues found in on-target chains."

    return output, dsn_seq


def make_sel_outputs(batch, offtgt_batch, pred_tokens, offtgt_pred_tokens, translate_fn):

    sample_id, offtgt_sample_id = batch["sample_id"][0], offtgt_batch["sample_id"][0]
    output = {"sample_id": sample_id, "offtgt_sample_id": offtgt_sample_id, "chains": {}, "offtgt_chains": {}}

    # on target chains
    on_out, on_tgt_dsn_seq = _parse_output(batch, pred_tokens, translate_fn)
    output["chains"] = on_out

    # off target chains
    if offtgt_pred_tokens is None:
        return output

    off_out, off_tgt_dsn_seq = _parse_output(offtgt_batch, offtgt_pred_tokens, translate_fn)
    output["offtgt_chains"] = off_out
    assert on_tgt_dsn_seq == off_tgt_dsn_seq, "On-target and off-target designed sequences do not match."

    return output


def translate_fn(tokens):
    tokenizer = Tokenizer()
    seq = tokenizer.decode(tokens.cpu())
    return seq


@cli_main.command("run_hdimer")
@click.argument("id_file")
@click.option("--config_file", type=click.Path(exists=True))
@click.option("--ckpt_file", type=click.Path())
@click.option("--check_fixed", is_flag=True, default=False, help="Whether to check fixed residues.")
@click.option("--use_cpu", is_flag=True, default=False, help="Whether to use CPU for inference.")
@click.option("--output_dir", type=click.Path(), default=None)
@click.option("--sampling_config_file", type=click.Path(), default=None)
@click.option("--native", is_flag=True, default=False, help="Whether to use native sequences for unbound chains.")
@click.option("--verbose", is_flag=True, default=False, help="Whether to print verbose output.")
def run_hdimer_design_pipeline(
    id_file, config_file, ckpt_file, check_fixed, use_cpu, output_dir, sampling_config_file, native, verbose
):

    ckpt_file = Path(ckpt_file)
    is_benchmark = ckpt_file.suffix == ".yaml"
    if is_benchmark:
        task = RedesignWorker(OmegaConf.load(ckpt_file))
        task.load_ckpt()
    else:
        task = build_task(ckpt_file)

    id_df = pd.read_csv(id_file)
    sample_ids = get_sample_ids(id_df)
    cfg = OmegaConf.load(config_file)
    dataset, dataloader = get_sharded_data(cfg, shard_id=0, num_shards=1, sel_sample_ids=sample_ids)
    print(len(dataset))
    model = task.model.model
    if not use_cpu:
        model.cuda()
    model.eval()

    _use_unbound = not check_fixed

    sampling_cfg = get_sampling_config(sampling_config_file)

    output_dir = Path(output_dir) if output_dir else None
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True)

    def _crop_batch(crop_mask, batch):
        _cropped = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                assert v.shape[0] == 1, "batch size must be 1"
                _cropped[k] = v[:, crop_mask]
            else:
                _cropped[k] = v
        return _cropped

    for batch in tqdm(dataloader):
        if not use_cpu:
            batch = move_to_cuda(batch)
        ubd_batch = _crop_batch(batch["dsn_mask"][0].bool(), batch)

        _replace_pred_with_gt = False
        if native:
            pred_tokens = batch["res_type"][0]
        elif is_benchmark:
            _replace_pred_with_gt = task.model_type in {"pifold"}

            gt_res_type = batch["res_type"].clone()
            pred_tokens = run_benchmark(task, batch, temperature=sampling_cfg.temperature, evaluation=False)[
                "generation"
            ]["pred_tokens"][0].to(device=gt_res_type.device)
            batch["res_type"] = gt_res_type
        else:
            pred_tokens, _ = contrast_decode_batch(
                batch,
                model,
                sampling_cfg,
                con_batch=ubd_batch,
                check_fixed=check_fixed,
                use_con=_use_unbound,
                verbose=verbose,
            )
        sample_id = batch["sample_id"][0]
        output = {"sample_id": sample_id}
        _out, _ = _parse_output(batch, pred_tokens, translate_fn, replace_pred_with_gt=_replace_pred_with_gt)
        output["chains"] = _out
        if output_dir:
            output_file = output_dir / f"{sample_id}.json"
            output["hparams"] = asdict(sampling_cfg)
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
        else:
            print(output)


@cli_main.command("run_seq")
@click.argument("id_file")
@click.argument("save_dir")
@click.option("--ckpt_file", type=click.Path())
@click.option("--config_file", type=click.Path(exists=True))
def run_sequence_recovery_inference_pipeline(id_file, save_dir, ckpt_file, config_file):
    id_df = pd.read_csv(id_file)
    sample_ids = get_sample_ids(id_df)
    # build inference task and load checkpoint
    ckpt_file = Path(ckpt_file)
    is_benchmark = ckpt_file.suffix == ".yaml"
    if is_benchmark:
        assert ckpt_file.suffix == ".yaml", "Benchmark mode requires a config file."
        _cfg = OmegaConf.load(ckpt_file)
        task = RedesignWorker(_cfg)
        task.load_ckpt()
    else:
        task = build_task(ckpt_file)
    # print(task)
    # dataset and sharded dataloader (one shard per GPU)
    cfg = OmegaConf.load(config_file)
    dataset, dataloader = get_sharded_data(cfg, shard_id=0, num_shards=1, sel_sample_ids=sample_ids)
    # run inference algorithms
    task.cuda()
    click.echo(f"Running inference on {len(dataloader)} samples")
    results, design_results = [], {}
    for sample in tqdm(dataloader):
        sample_id = sample["sample_id"][0]
        if is_benchmark:
            out = run_benchmark(task, sample)
        #     res = get_logits_sampling_result(sample, out, dataset.metadata[sample_id])
        #     dsn_seqs = get_logits_sampling_design_sequences(sample, out["generation"])
        #     dsn_res = {**res, "chain_seqs": dsn_seqs}
        else:
            out = run_logits_sampling(task, sample, valid=False, ignore_prefix=False)
        res = get_logits_sampling_result(sample, out, dataset.metadata[sample_id])
        dsn_seqs = get_logits_sampling_design_sequences(sample, out["generation"])
        dsn_res = {**res, "chain_seqs": dsn_seqs}
        results.append(res)
        design_results[res["id"]] = dsn_res
    results = pd.DataFrame(results)
    # store results
    save_redesign_results(save_dir, results, design_results)


def run_benchmark(worker, sample, evaluation=True, **kwargs):
    if worker.model_type in {"protein_mpnn", "pifold", "soluble_mpnn"}:
        sample["res_type"] = worker.convert_tokens(sample["res_type"])
        sample["mask"] = sample["mask"].float()
    elif worker.model_type == "esmif":
        kwargs["check_nsr"] = True

    model = worker.model
    model.eval()
    final_out = {}
    sample = move_to_cuda(sample)
    t0 = time.perf_counter()
    if evaluation:
        score_out = model.score(sample)
        score_out["timing"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        final_out["scoring"] = score_out

    sample_out = model.sample(sample, **kwargs)
    sample_out["timing"] = time.perf_counter() - t0
    if worker.model_type in {"protein_mpnn", "pifold", "soluble_mpnn"}:
        sample_out["pred_tokens"] = worker.convert_tokens(sample_out["pred_res_type"], reverse=True)
    elif worker.model_type == "esmif":
        sample_out["pred_tokens"] = worker.convert_tokens(seq=sample_out["pred_seq"])[None]
    final_out["generation"] = sample_out
    return final_out



def save_redesign_results(save_dir, results: pd.DataFrame, design_seqs):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    results_file = save_dir / "redsn_stats.csv"
    results.to_csv(results_file, index=False)
    seqs_file = save_dir / "design_sequences.json"
    with open(seqs_file, "w") as f:
        json.dump(design_seqs, f, indent=2)


def get_logits_sampling_design_sequences(sample, gen_out):
    bsz, seqlen = sample["chain_index"].shape
    assert bsz == 1
    chain_index = sample["chain_index"][0].cpu()
    assert gen_out["pred_tokens"].shape == (1, seqlen)
    pred_tokens = gen_out["pred_tokens"][0].cpu()
    chain_id_mapping = sample["chain_id_mapping"][0]
    tokenizer = Tokenizer()
    chain_seqs = tokenizer.decode(pred_tokens, chain_index)
    chain_seqs = {c: chain_seqs[i] for c, i in chain_id_mapping.items() if i in chain_seqs}
    return chain_seqs


def get_logits_sampling_result(sample, model_out, metadata):
    gen_out = model_out["generation"]
    score_out = model_out["scoring"]
    sample_id = sample["sample_id"][0]

    ll_val = gen_out.get("ll", 0)
    ll = float(ll_val.item() if hasattr(ll_val, "item") else ll_val)

    nat_ll_val = score_out.get("ll", 0)
    nat_ll = float(nat_ll_val.item() if hasattr(nat_ll_val, "item") else nat_ll_val)

    result = {
        "nsr": float(gen_out["nsr"]),
        "ll": ll,
        "ppl": math.exp(-ll),
        "nat_ll": nat_ll,
        "nat_ppl": math.exp(-nat_ll),
        "dsn_nsr": gen_out["dsn_nsr"],
        "dsn_site_nsr": gen_out["dsn_site_nsr"],
        "id": sample_id,
        "clu_id": metadata["clu_id"],
        "interface_type": metadata["interface_type"],
        "generation_time": gen_out.get("timing", -1),
        "scoring_time": score_out.get("timing", -1),
    }

    return result

def score_benchmark(worker, sample, **kwargs):
    if worker.model_type in {"protein_mpnn", "pifold", "soluble_mpnn"}:
        sample["res_type"] = worker.convert_tokens(sample["res_type"])
        sample["pdb_res_type"] = worker.convert_tokens(sample["pdb_res_type"])
        sample["mask"] = sample["mask"].float()
    elif worker.model_type == "esmif":
        kwargs["check_nsr"] = True
        
        
    # model = worker.model
    # model_out = model.score(sample)
    final_out = score_single_batch(worker.model, sample, worker.model_type, worker=worker)
    
    return final_out

@torch.no_grad()
def score_single_batch(model, batch, worker_type, worker=None):
    
    model.eval()
    
    nres = batch["res_type"].shape[1]
    # TODO: remove decoding order?
    batch["decoding_order_randn"] = torch.arange(nres, dtype=torch.float32, device=batch['res_type'].device)[None]

    _res = {"nres": nres, "file_id": batch["file_id"], "full_idx": batch["index"], "within_idx": batch["df_index"]}

    out = score_ll(batch, model, worker_type, worker=worker)

    _res.update({"-logkd": -batch["score"].item(), "-logkd_wt": -batch["score_wt"].item()})

    num_dsn_res = batch["dsn_mask"][0].sum().item()
    _res.update({
        "nmut": out["nmut"],
        "num_dsn_res": num_dsn_res,
        'll': out['ll'],
        "ll_mt": out["ll_mt"],
        "ll_ref": out["ll_ref"],
        'll_global': out['ll_global'],
    })

    ub_out = score_ll_ub(batch, model, worker_type, worker=worker)
    _res.update({
        'ub_ll': ub_out['ll'],
        "ub_ll_mt": ub_out["ll_mt"],
        "ub_ll_ref": ub_out["ll_ref"],
        'ub_ll_global': ub_out['ll_global']
    })
    _res.update({
        "cd_ll": _res['ll'] - _res['ub_ll'],
        "cd_ll_ref": _res["ll_ref"] - _res["ub_ll_ref"],
    })

    return _res

@torch.no_grad()
def score_ll(batch, model, worker_type, worker=None):
    model = model.eval()
    if worker_type in {'rednet'}:
        model_out = model.score(batch, reduction="per_sample")
        lprob = model_out["log_probs"][0].cpu()
    elif worker_type in {'pifold', 'protein_mpnn', 'soluble_mpnn'}:
        model_out = model.score(batch)
        lprob = torch.from_numpy(model_out["log_probs"][0])
    elif worker_type in {'esmif'}:
        model_out = model.score(batch)
        batch['res_type'] = worker.convert_tokens(batch['res_type'][0])[1:][None]
        batch['pdb_res_type'] = worker.convert_tokens(batch['pdb_res_type'][0])[1:][None]
        assert batch['res_type'].shape == model_out['tgt_tokens'].shape, f"Mismatch in res_type shape {batch['res_type'].shape} vs {model_out['tgt_tokens'].shape}"
        assert torch.all(batch['res_type'] == model_out['tgt_tokens'].to(device=batch['res_type'].device)), f"res_type and tgt_type do not match,  {batch['res_type']} vs {model_out['tgt_tokens']}"
        lprob = model_out["log_probs"][0].cpu()
    else:
        raise NotImplementedError(f"Unknown worker type: {worker_type}")
        
    out = {'ll': model_out['ll'].item() if not isinstance(model_out['ll'], float) else model_out['ll']}
    res_type = batch["res_type"][0].cpu()
    wt_type = batch["pdb_res_type"][0].cpu()
    num_cls = lprob.shape[-1]
    ll = torch.sum(lprob * F.one_hot(res_type.long(), num_classes=num_cls).float(), dim=-1)
    wt_ll = torch.sum(lprob * F.one_hot(wt_type.long(), num_classes=num_cls).float(), dim=-1)
    mask = wt_type != res_type
    num_mut = mask.sum().item()
    # with ref
    out['ll_ref'] = (ll - wt_ll)[mask].mean().item()
    # overall score
    out["ll_global"] = ll.mean().item()
    # mut area
    out["ll_mt"] = ll[mask].mean().item()
    out["nmut"] = num_mut
    return out

@torch.no_grad()
def score_ll_ub(batch, model, worker_type, worker=None):
    dsn_mask = batch["dsn_mask"][0]
    
    def _crop_batch(crop_mask, batch):
        _cropped = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                assert v.shape[0] == 1, "batch size must be 1"
                if v.ndim == 1 or (v.ndim == 2 and v.shape[1] == 1):
                    _cropped[k] = v
                else:
                    _cropped[k] = v[:, crop_mask]
            else:
                _cropped[k] = v
        return _cropped

    ub_batch = _crop_batch(dsn_mask, batch)
    out = score_ll(ub_batch, model, worker_type, worker=worker)
    return out

@cli_main.command("run_skempi")
@click.option("--ckpt_file", type=click.Path())
@click.option("--cfg_file", type=click.Path())
@click.option("--output_dir", type=click.Path(), default=None)
@click.option("--save_ids", is_flag=True, default=False)
@click.option("--id_file", type=click.Path(), default=None)
def run_skempi_pipeline(ckpt_file, cfg_file, output_dir, save_ids, id_file):
    cfg = OmegaConf.load(cfg_file)
    # cfg = OmegaConf.create({**cfg.data.config})
    # print(cfg)
    dataset = SkempiDataset(cfg.data.config, use_mut_chain_id=True)

    def get_file_ids(dataset, min_num_variants=1, max_num_res=1000):
        def _get_num(file_id):
            start, end = dataset.get_index_range(file_id)
            unique_seqs = set()
            for i in range(start, end + 1):
                sample = dataset[i]
                res_type = sample['res_type'].cpu().numpy()
                unique_seqs.add(tuple(res_type.tolist())) 
            return len(unique_seqs)
        def _get_num_res(file_id):
            start, end = dataset.get_index_range(file_id)
            sample = dataset[start]
            return sample['res_type'].shape[0]
        out = []
        for f in dataset.file_ids:
            num_variants = _get_num(f)
            num_res = _get_num_res(f)
            if num_variants >= min_num_variants and num_res <= max_num_res:
                out.append({'file_id': f, 'num_unique_variants': num_variants, 'num_res': num_res})
        return pd.DataFrame(out)

    if id_file and not save_ids:
        file_df = pd.read_csv(id_file)
    else:
        file_df = get_file_ids(dataset, min_num_variants=10)
    if save_ids:
        assert id_file is not None, "id_file must be provided to save ids."
        file_df.to_csv(id_file, index=False)
        
    file_ids = file_df['file_id'].to_list() 
    print(f"Selected {len(file_ids)} files for scoring.") 
    def get_data_ids(file_ids):
        data_ids = []
        for file_id in file_ids:
            start, end = dataset.get_index_range(file_id)
            data_ids += list(range(start, end+1))
        return data_ids
    
    _inds = get_data_ids(file_ids)

    # build model
    ckpt_file = Path(ckpt_file)
    is_benchmark = ckpt_file.suffix == ".yaml"
    if is_benchmark:
        task = RedesignWorker(OmegaConf.load(ckpt_file))
        task.load_ckpt()
        task.cuda()
    else:
        task = build_task(ckpt_file)
        model = task.model
        model.cuda()
    
    # score selected indices
    dms_result = []
    for i in tqdm(_inds):
        batch = move_to_cuda({k: v[None] if torch.is_tensor(v) else v for k, v in dataset[i].items()})
        if is_benchmark:
            out = score_benchmark(task, batch)
        else:
            model.eval()
            out = score_single_batch(model, batch, 'rednet')
        if out is None:
            continue
        dms_result.append(out)

    dms_result = pd.DataFrame(dms_result)

    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        output_file = output_dir / "results.csv"
        dms_result.to_csv(output_file, index=False)
        
if __name__ == "__main__":
    cli_main()
