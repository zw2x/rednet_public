import json
import click
import dotenv

dotenv.load_dotenv()
import pandas as pd
import numpy as np
from pathlib import Path


@click.group()
def cli_main():
    """Main entry point for the CLI application."""
    pass


@cli_main.command("hdimer")
@click.argument("in_root", type=click.Path(exists=True))
@click.argument("save_dir", type=click.Path())
def check_hdimer_results(in_root, save_dir):
    """Check heterodimer prediction results."""
    strict = True
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df = []
    # all_results = {}
    # for in_dir in Path(in_root).iterdir():
    #     if not in_dir.is_dir():
    #         continue
    #     # check_prediction_confidences(in_dir)
    #     in_dirs = list(in_dir.glob("af3_results/*/"))
    #     assert len(in_dirs) > 0, f"No af3_results found in {in_dir}"
    #     if len(in_dirs) > 1:
    #         print(f"Multiple af3_results found in {in_dir} ({in_dirs}), checking all...")
    #     if strict:
    #         assert len(in_dirs) == 1, f"Multiple af3_results found in {in_dir} ({in_dirs})"
    #     for _in_dir in in_dirs:
    #         results = parse_confidences(_in_dir)
    #         if results:
    #             all_results[in_dir.stem] = results
                
    all_results = _find_results(in_root, pattern="af3_results/*/", strict=strict)
    for result_name, results in all_results.items():
        # assume sorted by ranking score
        best_result = results[0]
        dsn_file = Path(in_root) / f"{result_name}.json"
        assert dsn_file.exists(), f"Design configuration file not found: {dsn_file}"
        dsn_data = json.load(open(dsn_file, "r"))
        dsn_chains = [c for c, d in dsn_data["chains"].items() if d['is_dsn']]
        assert len(dsn_chains) == 1, f"Multiple or no designed chains found in {dsn_file}"
        tgt_chains = [c for c, d in dsn_data["chains"].items() if not d['is_dsn']]
        assert len(tgt_chains) == 1, f"Multiple or no target chains found in {dsn_file}"
        row = _parse_best(dsn_chains[0], tgt_chains[0], result_name, best_result)
        df.append(row)

    df = pd.DataFrame(df)
    df_file = save_dir / "best_summary.csv"
    df.to_csv(df_file, index=False)

    result_file = save_dir / "all_results.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=4)

def _find_results(in_root, pattern, strict=True):
    all_results = {}
    for in_dir in Path(in_root).iterdir():
        if not in_dir.is_dir():
            continue
        # check_prediction_confidences(in_dir)
        in_dirs = list(in_dir.glob(pattern))
        assert len(in_dirs) > 0, f"No af3_results found in {in_dir}"
        if len(in_dirs) > 1:
            print(f"Multiple af3_results found in {in_dir} ({in_dirs}), checking all...")
        if strict:
            assert len(in_dirs) == 1, f"Multiple af3_results found in {in_dir} ({in_dirs})"
        for _in_dir in in_dirs:
            results = parse_confidences(_in_dir)
            if results:
                all_results[in_dir.stem] = results
    return all_results

def _parse_best(auth_dsn_chain, auth_tgt_chain, result_name, best_result):
    dsn_chain = best_result['chain_id_mapping'][auth_dsn_chain]
    tgt_chain = best_result['chain_id_mapping'][auth_tgt_chain]
    row = {
        "model_name": result_name,
        "auth_dsn_chain": auth_dsn_chain,
        "model_cif": best_result["model_cif"],
        "ranking_score": best_result["ranking_score"],
        "ptm": best_result["ptm"],
        "iptm": best_result["iptm"],
        "mean_ipae": best_result["mean_ipae"],
        "mean_ipae_with_contact": best_result["mean_ipae_with_contact"],
        "min_ipae_with_contact": min(best_result["chain_min_ipaes"]),
        
        "dsn_chain_plddt": best_result["chainwise_plddts"][dsn_chain],
        "dsn_chain_ptm": best_result["chain_ptms"][dsn_chain],
        "dsn_seq": best_result["input_seqs"][dsn_chain],
        
        'tgt_chain_plddt': best_result["chainwise_plddts"][tgt_chain],
        'tgt_chain_ptm': best_result["chain_ptms"][tgt_chain],
        'tgt_seq': best_result["input_seqs"][tgt_chain],
    }
    return row

@cli_main.command("pdb_sel")
@click.argument("in_root", type=click.Path(exists=True))
@click.argument("save_dir", type=click.Path())
def check_selective_binder_results(in_root, save_dir):
    """Check selective binder prediction results."""
    strict = True
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    on_tgt_results = _find_results(in_root, pattern="on_tgt/af3_results/*/", strict=strict)
    off_tgt_results = _find_results(in_root, pattern="off_tgt/af3_results/*/", strict=strict)
    names = set(on_tgt_results.keys()).intersection(set(off_tgt_results.keys()))
    df = []
    if strict:
        assert len(names) == len(on_tgt_results)
    for result_name in sorted(names):
        dsn_file = Path(in_root) / f"{result_name}.json"
        assert dsn_file.exists(), f"Design configuration file not found: {dsn_file}"
        dsn_data = json.load(open(dsn_file, "r"))
        on_tgt_dsn_chains = [c for c, d in dsn_data["chains"].items() if d['is_dsn']]
        assert len(on_tgt_dsn_chains) == 1, f"Multiple or no designed chains found in {dsn_file}"
        # assume sorted by ranking score
        on_tgt_tgt_chains = [c for c, d in dsn_data["chains"].items() if not d['is_dsn']]
        assert len(on_tgt_tgt_chains) == 1, f"Multiple or no target chains found in {dsn_file}"
        row = _parse_best(on_tgt_dsn_chains[0], on_tgt_tgt_chains[0], result_name, on_tgt_results[result_name][0])
        row['model_name'] += "-on"
        df.append(row)
        # get off-target designed chain
        
        off_tgt_dsn_chains = [c for c, d in dsn_data["offtgt_chains"].items() if d['is_dsn']]
        assert len(off_tgt_dsn_chains) == 1, f"Multiple or no designed chains found in {dsn_file}"
        off_tgt_tgt_chains = [c for c, d in dsn_data["offtgt_chains"].items() if not d['is_dsn']]
        assert len(off_tgt_tgt_chains) == 1, f"Multiple or no target chains found in {dsn_file}"
        row = _parse_best(off_tgt_dsn_chains[0], off_tgt_tgt_chains[0], result_name, off_tgt_results[result_name][0])
        row['model_name'] += "-off"
        df.append(row)

    df = pd.DataFrame(df)
    df_file = save_dir / "best_summary.csv"
    df.to_csv(df_file, index=False)

    result_file = save_dir / "on_tgt_results.json"
    with open(result_file, "w") as f:
        json.dump(on_tgt_results, f, indent=4)
        
    result_file = save_dir / "off_tgt_results.json"
    with open(result_file, "w") as f:
        json.dump(off_tgt_results, f, indent=4)
   


def parse_confidences(in_dir, match_pattern: str = "seed-*_sample-*"):
    """
    Parse AlphaFold3 confidence outputs.

    Args:
        in_dir: Directory containing AF3 outputs (with seed-*_sample-* subdirectories)
        return_all_samples: If True, return results for all samples; otherwise return best by ranking_score

    Returns:
        Dictionary with confidence metrics
    """
    in_dir = Path(in_dir)

    # Check for ranking_scores.csv to get all samples
    ranking_file = in_dir / "ranking_scores.csv"
    if not ranking_file.exists():
        return None

    # # Parse ranking scores
    # rankings = pd.read_csv(ranking_file)

    # Get sample directories
    if match_pattern is None:
        sample_dirs = [in_dir]
    else:
        sample_dirs = sorted(in_dir.glob(match_pattern))
        if not sample_dirs:
            return None

    # chain_id_mapping: auth_chain_id -> model_chain_id 
    chain_id_mapping_file = in_dir.parent.parent / "chain_id_remapping.json"
    assert chain_id_mapping_file.exists(), f"Chain ID remapping file not found: {chain_id_mapping_file}"
    chain_id_mapping = json.load(open(chain_id_mapping_file, "r"))
   
    input_file = in_dir.parent.parent / "inputs.json" 
    assert input_file.exists(), f"Input file not found: {input_file}"
    input_data = json.load(open(input_file, "r"))
    input_seqs = {v['protein']['id']: v['protein']['sequence'] for v in input_data['sequences']}
    all_results = []
    for sample_dir in sample_dirs:
        # Parse seed and sample from directory name
        dir_name = sample_dir.name
        if "seed-" not in dir_name or "sample-" not in dir_name:
            seed, sample = 0, 0
            summary_file = list(sample_dir.glob("*summary_confidences.json"))[0]
            confid_file = [
                d for d in sample_dir.glob("*confidences.json") if not d.name.endswith("summary_confidences.json")
            ][0]
        else:
            seed, sample = dir_name.replace("seed-", "").replace("sample-", "").split("_")
            seed, sample = int(seed), int(sample)

            summary_file = sample_dir / "summary_confidences.json"

            confid_file = sample_dir / "confidences.json"

        assert summary_file.exists(), f"Summary confidences file not found: {summary_file}"
        summary = json.load(open(summary_file, "r"))

        assert confid_file.exists(), f"Confidences file not found: {confid_file}"
        confid_data = json.load(open(confid_file, "r"))
        confid_data = {k: np.array(v) for k, v in confid_data.items()}

        contact_probs = confid_data["contact_probs"]
        pae = confid_data["pae"]
        token_chain_ids = confid_data["token_chain_ids"]
        unique_chains = np.unique(token_chain_ids)
        same_chain = token_chain_ids[None, :] == token_chain_ids[:, None]
        _has_contact = contact_probs > 0.5
        is_icontact = _has_contact * (~same_chain)
        ipae = pae * is_icontact + 1000 * (~is_icontact)  # ignore non-contact pairs for iptm calculation
        min_ipae = np.min(ipae, axis=1)
        # has confident contacts
        min_ipaes = [float(np.min(min_ipae[token_chain_ids == c])) for c in unique_chains]

        target_mask = token_chain_ids == unique_chains[0]  # assume chain A is target
        binder_mask = token_chain_ids == unique_chains[1]  # assume chain B is binder
        mean_ipae = float(get_pae_loss(pae, binder_mask, target_mask))
        mean_ipae_with_contact = float(get_pae_loss(pae, binder_mask, target_mask, _has_contact))

        model_cif = sample_dir / "model.cif"
        assert model_cif.exists(), f"Model CIF file not found: {model_cif}"
       
        # Extract metrics
        result = {
            "seed": seed,
            "sample": sample,
            "ranking_score": summary.get("ranking_score", 0),
            "ptm": summary.get("ptm", 0),
            "iptm": summary.get("iptm", 0),
            "fraction_disordered": summary.get("fraction_disordered", 0),
            "has_clash": summary.get("has_clash", 0),
            "chain_ptm": summary.get("chain_ptm", []),
            "chain_iptm": summary.get("chain_iptm", []),
            "chain_pair_iptm": summary.get("chain_pair_iptm", []),
            "chain_pair_pae_min": summary.get("chain_pair_pae_min", []),
            "chain_min_ipaes": min_ipaes,
            "mean_ipae": mean_ipae,
            "mean_ipae_with_contact": mean_ipae_with_contact,
            "model_cif": str(model_cif.resolve()),
            "chain_id_mapping": chain_id_mapping,
            "input_seqs": input_seqs,
        }

        # Compute per-chain pLDDT
        atom_chain_ids = confid_data["atom_chain_ids"]
        atom_plddts = confid_data["atom_plddts"]
        unique_chains, unique_inds = np.unique(atom_chain_ids, return_index=True)
        assert len(unique_chains) == len(chain_id_mapping), "Mismatch in number of chains between confidences and chain ID mapping"
        
        result["chain_ids"] = [str(atom_chain_ids[i]) for i in unique_inds]
        result["chainwise_plddts"] = {str(c): float(np.mean(atom_plddts[atom_chain_ids == c])) for c in unique_chains}
        result["chain_ptms"] = {str(c): float(result["chain_ptm"][i]) for i, c in enumerate(result["chain_ids"])}

        all_results.append(result)
    all_results = sorted(all_results, key=lambda x: x["ranking_score"], reverse=True)
    return all_results


# from colabdesign
def mask_loss(x, mask):
    x_masked = (x * mask).sum() / (1e-8 + mask.sum())
    return x_masked


def get_pae_loss(p: np.ndarray, mask_1d=None, mask_1b=None, mask_2d=None):
    p = (p + p.T) / 2
    L = p.shape[0]
    if mask_1d is None:
        mask_1d = np.ones(L)
    if mask_1b is None:
        mask_1b = np.ones(L)
    if mask_2d is None:
        mask_2d = np.ones((L, L))
    mask_2d = mask_2d * mask_1d[:, None] * mask_1b[None, :]
    return mask_loss(p, mask_2d)


if __name__ == "__main__":
    cli_main()
