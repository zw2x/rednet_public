import os
import json
import rich
import time
import click
import joblib
import dotenv

dotenv.load_dotenv()

import collections
from pathlib import Path
from omegaconf import OmegaConf

import torch
import numpy as np
import pandas as pd
from einops import repeat, rearrange

from faust.tokenizer import Tokenizer
from faust.utils.parallel import parallel_process
from faust.tools.struct_align import align_cif, parse_usalign_output

from rednet.data.structure_pipeline import StructurePipeline

PDB_DATASET_DIR = Path(os.getenv("PDB_DATASET_DIR"))


@click.group()
def cli_main():
    """
    Make selective binding dataset
    """
    pass


def get_auth_chain_id(struct, chain_id):
    ent = struct["entry"]["chains"]
    return ent[chain_id]["pdbx_strand_id"]


@cli_main.command("test_tmalign")
@click.argument("qry_id")
@click.argument("tgt_id")
@click.option("--cif_dir", default=PDB_DATASET_DIR / "mmcif_files")
@click.option("--struct_dir", default=PDB_DATASET_DIR / "parsed_structures")
@click.option("--out_root", default=PDB_DATASET_DIR / "all_seqs_mmseqs_90" / "tmalign_results")
def test_tmalign(qry_id, tgt_id, cif_dir, struct_dir, out_root):

    cif_dir = Path(cif_dir)
    struct_dir = Path(struct_dir)
    _run_tmalign(qry_id, tgt_id, cif_dir, struct_dir, out_root)


@cli_main.command("tmalign_cluster")
@click.argument("clu_file")
@click.option("--cif_dir", default=PDB_DATASET_DIR / "mmcif_files")
@click.option("--struct_dir", default=PDB_DATASET_DIR / "parsed_structures")
@click.option("--out_root", default=PDB_DATASET_DIR / "all_seqs_mmseqs_90" / "tmalign_results")
def tmalign_cluster(clu_file, cif_dir, struct_dir, out_root):
    clu_df = pd.read_csv(clu_file)
    clus = _collect_pairs_from_clu_df(clu_df)

    batched_args = []
    for lig_clu, pairs in clus.items():
        tgt_lig_id, tgt_rec_id = pairs[0]
        for off_tgt_id in pairs[1:]:
            qry_lig_id, qry_rec_id = off_tgt_id
            # _run_tmalign(qry_id, tgt_id, cif_dir, struct_dir, out_root)
            batched_args.append({"qry_id": qry_lig_id, "tgt_id": tgt_lig_id})
            batched_args.append({"qry_id": qry_rec_id, "tgt_id": tgt_rec_id})

    print(f"Total {len(batched_args)} alignments to run.")
    # parallel run
    parallel_process(
        _run_tmalign,
        batched_args=batched_args,
        shared_args={"cif_dir": cif_dir, "struct_dir": struct_dir, "out_root": out_root},
        n_jobs=16,
    )


def _run_tmalign(qry_id, tgt_id, cif_dir, struct_dir, out_root):
    cif_dir = Path(cif_dir)
    struct_dir = Path(struct_dir)

    def _parse_ids(sample_id):
        asm_id, chain_id = sample_id.split("_")
        return asm_id, chain_id

    qry_asm_id, qry_chain_id = _parse_ids(qry_id)
    qry_file, qry_struct_file = cif_dir / f"{qry_asm_id}.cif", struct_dir / f"{qry_asm_id}.lz4"
    if not qry_file.exists():
        rich.print(f"[red]Warning: Missing cif file {qry_file}[/red]")
        return
    if not qry_struct_file.exists():
        rich.print(f"[red]Warning: Missing structure file {qry_struct_file}[/red]")
        return

    tgt_asm_id, tgt_chain_id = _parse_ids(tgt_id)
    tgt_file, tgt_struct_file = cif_dir / f"{tgt_asm_id}.cif", struct_dir / f"{tgt_asm_id}.lz4"
    if not tgt_file.exists():
        rich.print(f"[red]Warning: Missing cif file {tgt_file}[/red]")
        return
    if not tgt_struct_file.exists():
        rich.print(f"[red]Warning: Missing structure file {tgt_struct_file}[/red]")
        return

    qry_struct, tgt_struct = joblib.load(qry_struct_file), joblib.load(tgt_struct_file)
    qry_auth_chain_id = get_auth_chain_id(qry_struct, qry_chain_id)
    tgt_auth_chain_id = get_auth_chain_id(tgt_struct, tgt_chain_id)

    out_dir = Path(out_root) / f"{qry_id}+{tgt_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"stdout.txt"
    if out_file.exists():
        return
    res = align_cif(qry_file, tgt_file, out_dir, chain1=qry_auth_chain_id, chain2=tgt_auth_chain_id)
    return res


def _collect_pairs_from_clu_df(clu_df):
    clus = {}
    for lig_clu, group in clu_df.groupby("lig_clu"):
        _clus = []
        for i, (idx, row) in enumerate(group.iterrows()):
            pairs = row["pairs"].split(" ")
            asm_id, chain_pair = pairs[0].split("_")
            lig_id, rec_id = chain_pair.split(":")
            lig_id = f"{asm_id}_{lig_id}"
            rec_id = f"{asm_id}_{rec_id}"
            _clus.append((lig_id, rec_id))
        clus[lig_clu] = _clus
    return clus


@cli_main.command("parse_tmalign_cluster")
@click.argument("in_dir")
@click.argument("clu_file")
@click.option("--out_root")
@click.option("--struct_dir", default=PDB_DATASET_DIR / "parsed_structures")
@click.option("--cfg_file", default="configs/pdb_sel.yaml")
def parse_tmalign_cluster_results(in_dir, clu_file, out_root, struct_dir, cfg_file):
    cfg = OmegaConf.load(cfg_file)
    clu_df = pd.read_csv(clu_file)
    print(len(clu_df))
    in_dir = Path(in_dir)
    clus = _collect_pairs_from_clu_df(clu_df)
    num_potential_off_targets = sum([len(v) - 1 for v in clus.values()])
    results = {}

    def _lig_filter(res):
        # check coverage
        tgt_cov = res["aligned_length"] / res["length_2"]
        seq_id = res["seq_id"]
        if tgt_cov < 0.9 and seq_id < 0.9:
            return False
        rmsd = res["rmsd"]
        if rmsd > 2.5:
            return False
        return True

    def _rec_filter(res):
        seq_id = res["seq_id"]
        if seq_id >= 1:
            return False
        return True

    struct_dir = Path(struct_dir)
    out_dir = Path(out_root) / "feats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pipeline = PdbSelDataset(cfg)
    # pipeline = StructurePipeline(cfg, Tokenizer())
    pipeline = StructurePipeline(config=cfg.feature_config, deterministic=True, tokenizer=Tokenizer())
    parsed_structure_dir = Path(cfg.pdb_config.structure_dir)

    cluster_df = collections.defaultdict(list)
    missed = []
    for lig_clu, pairs in clus.items():
        tgt_lig_id, tgt_rec_id = pairs[0]
        off_tgts = []
        for off_tgt_id in pairs[1:]:
            qry_lig_id, qry_rec_id = off_tgt_id
            lig_out_file = in_dir / f"{qry_lig_id}+{tgt_lig_id}" / "stdout.txt"
            rec_out_file = in_dir / f"{qry_rec_id}+{tgt_rec_id}" / "stdout.txt"
            if not (lig_out_file.exists() and rec_out_file.exists()):
                continue
            lig_res = parse_tmalign_result(lig_out_file)
            if not _lig_filter(lig_res):
                continue
            rec_res = parse_tmalign_result(rec_out_file)
            if not _rec_filter(rec_res):
                continue

            # make aligned sequences
            qry_asm_id, qry_lig_chain_id = qry_lig_id.split("_")
            qry_struct = joblib.load(struct_dir / f"{qry_asm_id}.lz4")
            tgt_asm_id, tgt_lig_chain_id = tgt_lig_id.split("_")
            tgt_struct = joblib.load(struct_dir / f"{tgt_asm_id}.lz4")

            _, qry_rec_chain_id = qry_rec_id.split("_")
            _, tgt_rec_chain_id = tgt_rec_id.split("_")
            try:
                lig_out = make_aligned_seqs(lig_res, qry_struct, qry_lig_chain_id, tgt_struct, tgt_lig_chain_id)
                rec_out = make_aligned_seqs(rec_res, qry_struct, qry_rec_chain_id, tgt_struct, tgt_rec_chain_id)
            except RuntimeError:
                missed.append((pairs[0], off_tgt_id))
                continue
            qry_pairs = get_contact_res_pairs(
                qry_struct, qry_lig_chain_id, lig_out, qry_rec_chain_id, rec_out, id_key="aln_ids1"
            )
            tgt_pairs = get_contact_res_pairs(
                tgt_struct, tgt_lig_chain_id, lig_out, tgt_rec_chain_id, rec_out, id_key="aln_ids2"
            )
            jaccard_index = cal_jaccard_index(qry_pairs, tgt_pairs)
            ass_res = {"jaccard_index": jaccard_index}
            off_tgts.append((off_tgt_id, lig_res, rec_res, ass_res))

            _id = f"{tgt_lig_id}:{tgt_rec_id}+{qry_lig_id}:{qry_rec_id}"
            _out_dir = out_dir / _id
            _out_dir.mkdir(exist_ok=True, parents=True)
            feat = {
                "lig": lig_res,
                "rec": rec_res,
                "assessment": ass_res,
                "off_tgt_id": (qry_lig_id, qry_rec_id),
                "on_tgt_id": (tgt_lig_id, tgt_rec_id),
                "lig_aligned_out": lig_out,
                "rec_aligned_out": rec_out,
            }
            aln_ids1, aln_ids2 = lig_out["aln_ids1"], lig_out["aln_ids2"]
            # TODO: this is a bug. on_target_feat and off_target_feat are swapped
            _on_target_id = (qry_asm_id, (qry_lig_chain_id, qry_rec_chain_id))
            _off_target_id = (tgt_asm_id, (tgt_lig_chain_id, tgt_rec_chain_id))
            # parse features
            on_target_feat, off_target_feat = parse_pair_feats(
                parsed_structure_dir, pipeline, _on_target_id, _off_target_id, aln_ids1, aln_ids2
            )
            feat["on_target_feat"] = on_target_feat
            feat["off_target_feat"] = off_target_feat

            _out_file = _out_dir / "pfeats.lz4"
            joblib.dump(feat, str(_out_file))

        if len(off_tgts) > 0:
            results[lig_clu] = {"on_target": (tgt_lig_id, tgt_rec_id), "off_targets": off_tgts}

    print(len(results))
    print(len(missed), "missed pairs due to seq length mismatch.")

    for lig_clu, entry in results.items():
        off_tgts = entry["off_targets"]
        tgt_lig_id, tgt_rec_id = entry["on_target"]
        for off_tgt in off_tgts:
            cluster_df["lig_clu"].append(lig_clu)
            cluster_df["tgt_lig_id"].append(tgt_lig_id)
            cluster_df["tgt_rec_id"].append(tgt_rec_id)
            off_tgt_id, lig_res, rec_res, ass_res = off_tgt
            off_tgt_lig_id, off_tgt_rec_id = off_tgt_id
            cluster_df["off_tgt_lig_id"].append(off_tgt_lig_id)
            cluster_df["off_tgt_rec_id"].append(off_tgt_rec_id)
            tri_id = f"{off_tgt_lig_id}+{tgt_lig_id}:{off_tgt_rec_id}+{tgt_rec_id}"
            cluster_df["tri_id"].append(tri_id)
            # add tmalign results
            cluster_df["lig_seq_id"].append(lig_res["seq_id"])
            cluster_df["lig_rmsd"].append(lig_res["rmsd"])
            cluster_df["lig_aligned_length"].append(lig_res["aligned_length"])
            cluster_df["lig_qry_tm_score"].append(lig_res["tm_score_1"])
            cluster_df["lig_tgt_tm_score"].append(lig_res["tm_score_2"])

            cluster_df["rec_seq_id"].append(rec_res["seq_id"])
            cluster_df["rec_rmsd"].append(rec_res["rmsd"])
            cluster_df["rec_aligned_length"].append(rec_res["aligned_length"])
            cluster_df["rec_qry_tm_score"].append(rec_res["tm_score_1"])
            cluster_df["rec_tgt_tm_score"].append(rec_res["tm_score_2"])

            # add assessment results
            cluster_df["jaccard_index"].append(ass_res["jaccard_index"])

    print("==== Summary of tmalign results ====")
    cluster_df = pd.DataFrame(cluster_df)
    print(cluster_df)

    df_file = Path(out_root) / "cluster_df.csv"
    cluster_df.to_csv(df_file)
    print(f"Total {len(results)} clusters with valid tmalign results. from {len(clus)} clusters.")
    num_trials = sum([len(v["off_targets"]) for v in results.values()])
    print(
        f"Total {num_trials} tmalign trials with valid results. from {num_potential_off_targets} potential off-targets."
    )
    print(num_trials + len(results))


@cli_main.command("check_select")
@click.argument("in_dir")
def check_seletive_dataset(in_dir):
    in_dir = Path(in_dir)
    df_file = in_dir / "cluster_df.csv"
    clu_df = pd.read_csv(df_file)

    k = 1
    top_k_per_group = clu_df.sort_values("jaccard_index").groupby("lig_clu").head(k)

    clu_df = top_k_per_group.reset_index(drop=True).drop("Unnamed: 0", axis=1)
    sel_mask = clu_df["jaccard_index"] <= 0.9
    clu_df = clu_df[sel_mask]

    out_file = in_dir / "selected_cluster_df.csv"
    clu_df.to_csv(out_file, index=False)


def parse_pair_feats(structure_dir: Path, pipeline: StructurePipeline, on_tgt_id, off_tgt_id, aln_ids1, aln_ids2):
    # tgt_id, off_id = sample_id.split('+')

    # def _parse_id(_id):
    #     asm_id, chain_ids = _id.split('_')
    #     chain_ids = chain_ids.split(':')
    #     return asm_id, chain_ids

    # tgt_asm_id, tgt_chain_ids = _parse_id(tgt_id)
    # off_asm_id, off_chain_ids = _parse_id(off_id)
    tgt_asm_id, tgt_chain_ids = on_tgt_id
    off_asm_id, off_chain_ids = off_tgt_id

    def _make_crop_mask(struct, chain_id, aln_ids):
        chain_index = struct["chain_index"]
        res_type = struct["res_type"]
        ca_mask = struct["atom_mask"][..., 1].to(bool)
        is_std = res_type < 20
        i = struct["chain_id_mapping"][chain_id]
        mask = (chain_index == i) * ca_mask * is_std
        inds = torch.arange(len(mask))
        sel_inds = inds[mask][aln_ids]
        crop_mask = torch.zeros_like(mask, dtype=torch.bool)
        crop_mask[chain_index != i] = True
        crop_mask[sel_inds] = True
        return crop_mask, res_type[mask]

    def _crop_sample(struct, mask):
        cropped = {}
        mask = mask.bool()
        for k, v in struct.items():
            if isinstance(v, torch.Tensor):
                cropped[k] = v[mask]
            else:
                cropped[k] = v
        return cropped

    tgt_struct = parse_sample(structure_dir, pipeline, tgt_asm_id, 0, tgt_chain_ids, transform=True)
    off_struct = parse_sample(structure_dir, pipeline, off_asm_id, 0, off_chain_ids, transform=True)
    tgt_mask, tgt_restype = _make_crop_mask(tgt_struct, tgt_chain_ids[0], aln_ids1)
    off_mask, off_restype = _make_crop_mask(off_struct, off_chain_ids[0], aln_ids2)

    tgt_struct = _crop_sample(tgt_struct, tgt_mask)
    off_struct = _crop_sample(off_struct, off_mask)
    # _num_tgt_res = tgt_struct['dsn_mask'].sum()
    assert (
        tgt_struct["dsn_mask"].sum() == off_struct["dsn_mask"].sum()
    ), f"Mismatched residue number after cropping: {tgt_struct['dsn_mask'].sum()} vs {off_struct['dsn_mask'].sum()}"

    return tgt_struct, off_struct


def parse_sample(structure_dir: Path, pipeline: StructurePipeline, sample_id, idx, chain_ids, transform=True):
    raw_sample = {}
    sample = {"sample_id": sample_id}
    t0 = time.time()
    structure_path = structure_dir / f"{sample_id}.lz4"
    assert structure_path.exists(), f"Structure file {structure_path} does not exist!"
    target_chain_ids = chain_ids[:1]
    _structure = pipeline.load_parsed_structure(structure_path, chain_ids, target_chain_ids)
    raw_sample.update(_structure)
    sample["load_structure_time"] = time.time() - t0
    t1 = time.time()
    if not transform:
        sample.update(raw_sample)
    else:
        sample.update(pipeline.transform(raw_sample, index=idx))
    sample["transform_time"] = time.time() - t1
    return sample


def _get_clu_stats(grp):
    def _get_range(v):
        return [v.min(), v.max(), v.mean(), v.median()]

    return {
        k: _get_range(grp[k])
        for k in [
            "lig_seq_id",
            "lig_rmsd",
            "rec_seq_id",
            "rec_rmsd",
            "jaccard_index",
            "lig_aligned_length",
            "lig_qry_tm_score",
            "lig_tgt_tm_score",
            "rec_aligned_length",
            "rec_qry_tm_score",
            "rec_tgt_tm_score",
        ]
    }


def parse_tmalign_result(out_file):
    out_str = open(out_file).read()
    result = parse_usalign_output(out_str)
    return result


def get_contact_res_pairs(struct, lig_id, lig_out, rec_id, rec_out, id_key="aln_ids1"):

    lig_res_type, lig_mask = get_chain_res_type_from_struct(struct, lig_id)
    rec_res_type, rec_mask = get_chain_res_type_from_struct(struct, rec_id)

    pos = struct["atom_positions"][:, 1, :]
    mask = struct["atom_mask"][:, 1].astype(bool)
    lig_pos, lig_pos_mask = pos[lig_mask][lig_out[id_key]], mask[lig_mask][lig_out[id_key]]
    rec_pos, rec_pos_mask = pos[rec_mask][rec_out[id_key]], mask[rec_mask][rec_out[id_key]]
    d = np.linalg.norm(lig_pos[:, None, :] - rec_pos[None, :, :], axis=-1)
    pmask = lig_pos_mask[:, None] * rec_pos_mask[None, :]
    d = d * pmask + (1 - pmask) * 1e6

    flattend_mask = rearrange(d <= 10, "m n -> (m n)")
    sel_ids = np.arange(len(flattend_mask))[flattend_mask]
    m, n = len(lig_pos), len(rec_pos)
    pair_inds = rearrange(
        np.stack([repeat(np.arange(m), "m -> m n", n=n), repeat(np.arange(n), "n -> m n", m=m)], axis=-1),
        "m n c -> (m n) c",
    )[sel_ids]
    sel_lig_res_type = lig_res_type[lig_out[id_key]][pair_inds[:, 0]]
    sel_rec_res_type = rec_res_type[rec_out[id_key]][pair_inds[:, 1]]
    # pairs (n, 4): [lig_res_idx, rec_res_idx, lig_res_type, rec_res_type]
    pairs = np.concatenate([pair_inds, np.stack([sel_lig_res_type, sel_rec_res_type], axis=-1)], axis=-1)
    pairs = [tuple(p) for p in pairs.tolist()]
    return pairs


def cal_jaccard_index(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    return len(set1 & set2) / len(set1 | set2)


def get_chain_res_type_from_struct(struct, chain_id):
    chain_index = struct["chain_index"]
    res_type = struct["aatype"]
    ca_mask = struct["atom_mask"][..., 1].astype(bool)
    is_std = res_type < 20
    i = struct["chain_id_mapping"][chain_id]
    mask = (chain_index == i) * ca_mask * is_std
    return res_type[mask], mask


def get_aln_ids_from_tmalign(seq1, seq2):
    ids1 = []
    ids2 = []
    i, j = 0, 0
    for a, b in zip(seq1, seq2):
        if a != "-" and b != "-":
            ids1.append(i)
            ids2.append(j)
        if a != "-":
            i += 1
        if b != "-":
            j += 1
    return np.array(ids1).astype(np.int64), np.array(ids2).astype(np.int64)


def make_aligned_seqs(res, qry_struct, qry_id, tgt_struct, tgt_id):
    qry_res_type, _ = get_chain_res_type_from_struct(qry_struct, qry_id)  # off target
    tgt_res_type, _ = get_chain_res_type_from_struct(tgt_struct, tgt_id)  # on target
    if len(tgt_res_type) != len(res["alignment_target"].replace("-", "")):
        # print(tgt_struct["entry"]["chains"])
        # print(f"Warning: Mismatched target residue length for {qry_id}+{tgt_id}, skip.")
        # print(tgt_res_type)
        # print(len(tgt_res_type), len(res["alignment_target"].replace("-", "")))
        # print(res)
        raise RuntimeError

    if len(qry_res_type) != len(res["alignment_query"].replace("-", "")):
        # print(f"Warning: Mismatched off-target residue length for {qry_id}+{tgt_id}, skip.")
        raise RuntimeError

    qids, tids = get_aln_ids_from_tmalign(res["alignment_query"], res["alignment_target"])
    qry = qry_res_type[qids]
    tgt = tgt_res_type[tids]
    aln_seqid = np.mean(tgt == qry)
    # print(len(qids), len(tids), aln_seqid)

    out = {
        "aln_seqid": aln_seqid,
        "aln_ids1": qids.tolist(),
        "aln_ids2": tids.tolist(),
    }

    return out


if __name__ == "__main__":
    cli_main()
