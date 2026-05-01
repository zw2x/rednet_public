```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import OrderedDict, defaultdict
import json
from numbers import Number
```


```python
root_dir = Path("/home/zwxie/experiments/sel_experiments/")
print(root_dir.resolve())
models_dir = root_dir / 'models'
results_dir = root_dir / 'results'
assert models_dir.exists() and results_dir.exists()
```

    /home/zwxie/experiments/sel_experiments



```python
def group_pairs(df):
    df['target_type'] = df['model_name'].str.extract(r'-(on|off)$')[0]
    df['pair_name'] = df['model_name'].str.replace(r'-(on|off)$', '', regex=True)
    # print(df)
    numeric_cols = ['ranking_score', 'ptm', 'iptm', 'dsn_chain_plddt', 'dsn_chain_ptm',
                  'mean_ipae', 'mean_ipae_with_contact', 'min_ipae_with_contact']

    pair_df = []
    for pair_name, pair in df.groupby('pair_name'):
        on_row = pair[pair['target_type'] == 'on']
        off_row = pair[pair['target_type'] == 'off']
        res = {'pair_name': pair_name}
        assert len(on_row) == 1 and len(off_row) == 1
        for col in numeric_cols:
            res[f'{col}_diff'] = float(on_row[col].iloc[0]) - float(off_row[col].iloc[0])
            res[f'{col}_on'] = float(on_row[col].iloc[0])
            res[f'{col}_off'] = float(off_row[col].iloc[0])
        pair_df.append(res)
    return pd.DataFrame(pair_df)

def update_pair_chain_name_(df, prefix='tgt_lig'):
    # df[f'{prefix}_chain_name'] = df[f'{prefix}_id'].str.split('-assembly').str[0] + '_' + df[f'{prefix}_id'].str.split('_').str[-1]
    df[f'{prefix}_chain_name'] = df[f'{prefix}_id'] # .str.split('-assembly').str[0] + '_' + df[f'{prefix}_id'].str.split('_').str[-1]


def update_pair_model_(df):
    update_pair_chain_name_(df, 'tgt_lig')
    update_pair_chain_name_(df, 'off_tgt_lig')
    df['lig_pair'] = df['tgt_lig_chain_name'] + '+' + df['off_tgt_lig_chain_name']
    update_pair_chain_name_(df, 'tgt_rec')
    update_pair_chain_name_(df, 'off_tgt_rec')
    df['rec_pair'] = df['tgt_rec_chain_name'] + '+' + df['off_tgt_rec_chain_name']
    df['pair'] = df['lig_pair'] + ':' + df['rec_pair']
```


```python
def parse_result_seqs(in_file):
    in_data = json.load(open(in_file))
    results = {}
    for k, items in in_data.items():
        item = items[0]
        row = {
            'key': k,
            'ptm': item['ptm'],
            'iptm': item['iptm'],
            'input_seqs': item['input_seqs'],
            'chain_id_mapping': item['chain_id_mapping'],
        }
        results[k] = row
    return results

def parse_pair_results(in_dir, cluster_df):
    sel_pairs, pair_results = parse_pair_dfs(in_dir / 'best_summary.csv')
    off_tgt_data = parse_result_seqs(in_dir / 'off_tgt_results.json')
    on_tgt_data = parse_result_seqs(in_dir / 'on_tgt_results.json')
    pa = OrderedDict()
    for k in off_tgt_data:
        on_d, off_d = pair_results[k]['on'], pair_results[k]['off']
        on_id, off_id = k.split('+')
        on_dsn_chain_id = on_d['auth_dsn_chain']
        on_tgt_chain_id = [k for k in on_tgt_data[k]['chain_id_mapping'].keys() if k != on_dsn_chain_id][0]
        off_dsn_chain_id = off_d['auth_dsn_chain']
        off_tgt_chain_id = [k for k in off_tgt_data[k]['chain_id_mapping'].keys() if k != off_dsn_chain_id][0]

        pair_name = f'{on_id}_{on_dsn_chain_id}+{off_id}_{off_dsn_chain_id}:{on_id}_{on_tgt_chain_id}+{off_id}_{off_tgt_chain_id}'
        rows = cluster_df[cluster_df['pair'] == pair_name]
        assert len(rows) == 1, k
        res = rows.iloc[0].to_dict()
        pa[pair_name] = {'clu_info': res, 'on_info': on_d, 'off_info': off_d}
    return pa

def parse_pair_dfs(in_file, filter_high_confidence=False):
    df = pd.read_csv(in_file)
    # select confident pairs: reduce possiblity of prediction errors of the targets
    confident_tgts = {'on': set(), 'off': set()}
    if filter_high_confidence:
        df = df[df['tgt_chain_plddt'] > 70]
    model_names = df['model_name']
    pair_results = defaultdict(dict)
    for model_name, (i, row) in zip(model_names, df.iterrows()):
        if model_name.endswith('-on'):
            pair_name, m = model_name[:-3], 'on'
        else:
            assert model_name.endswith('-off')
            pair_name, m = model_name[:-4], 'off'
        confident_tgts[m].add(pair_name)
        assert m not in pair_results[pair_name]
        pair_results[pair_name][m] = row.to_dict()
    pairs = confident_tgts['on'].intersection(confident_tgts['off'])
    print(f'Load {len(pairs)} pairs')
    return pairs, pair_results
```


```python
native_df = pd.read_csv(results_dir / 'native/best_summary.csv')
native_pair_df = group_pairs(native_df)


# select confident pairs: reduce possiblity of prediction errors of the targets
confident_tgts = {'on': set(), 'off': set()}
for model_name in native_df[native_df['tgt_chain_plddt'] > 70]['model_name']:
    if model_name.endswith('-on'):
        pair_name, m = model_name[:-3], 'on'
    else:
        assert model_name.endswith('-off')
        pair_name, m = model_name[:-4], 'off'
    confident_tgts[m].add(pair_name)
pairs = confident_tgts['on'].intersection(confident_tgts['off'])
print(len(pairs))

cluster_df_file = Path("/home/zwxie/db/all_seqs_mmseqs_90_select_results/selected_cluster_df.csv")
assert cluster_df_file.exists()
cluster_df = pd.read_csv(cluster_df_file)

update_pair_model_(cluster_df)
```

    98



```python
rgat_b64_a1_b09_t0001_result = parse_pair_results(results_dir / 'rgat_b64_a1_b09_t0001/', cluster_df)
```

    Load 174 pairs



```python
def check_result(res):
    for k, d in res.items():
        print(k)
        print(d['clu_info'])
        print(d['on_info'])
        print(d['off_info'])
        break

def get_relaxed_structure_names(result):
    names = []
    for k in result.keys():
        # pair_name = f'{on_id}_{on_dsn_chain_id}+{off_id}_{off_dsn_chain_id}:{on_id}_{on_tgt_chain_id}+{off_id}_{off_tgt_chain_id}'
        on_pair, off_pair = k.split(':')
        on_dsn_chain, off_dsn_chain = on_pair.split('+')
        on_tgt_chain, off_tgt_chain = off_pair.split('+')
        on_id, _ = on_dsn_chain.split('_')
        off_id, _ = off_dsn_chain.split('_')
        names.extend([f'{on_id}+{off_id}-on', f'{on_id}+{off_id}-off'])
    print(len(names), len(set(names)))
    return sorted(set(names))

def parse_energy_differences(result, df):
    diffs = []
    for k in list(result.keys()):
        d = result[k]
        on_model = d['on_info']['model_name']
        on_rows = df[df['name'] == on_model]
        if len(on_rows) == 0:
            print(f'Missing {on_model}')
            continue
        elif len(on_rows) > 1:
            print(f'Multiple rows for {on_model}: {on_rows}')
            continue

        off_model = d['off_info']['model_name']
        off_rows = df[df['name'] == off_model]
        if len(off_rows) == 0:
            print(f'Missing {off_model}')
            continue
        elif len(off_rows) > 1:
            print(f'Multiple rows for {off_model}: {off_rows}')
            continue

        on_energy = on_rows.iloc[0].to_dict()
        off_energy = off_rows.iloc[0].to_dict()

        diff_energy = {_k: v - off_energy[_k] if _k != 'name' else k for _k, v in on_energy.items()}

        d['on_energy'] = on_energy
        d['off_energy'] = off_energy
        d['diff_energy'] = diff_energy

        diffs.append(diff_energy)
        
    diff_df = pd.DataFrame(diffs)
    
    return diff_df
```


```python
check_result(rgat_b64_a1_b09_t0001_result)
relaxed_structure_names = get_relaxed_structure_names(rgat_b64_a1_b09_t0001_result)
```

    7lvs-assembly1_B+7qgs-assembly1_A:7lvs-assembly1_A+7qgs-assembly1_B
    {'lig_clu': '7qgs-assembly1_A', 'tgt_lig_id': '7lvs-assembly1_B', 'tgt_rec_id': '7lvs-assembly1_A', 'off_tgt_lig_id': '7qgs-assembly1_A', 'off_tgt_rec_id': '7qgs-assembly1_B', 'tri_id': '7qgs-assembly1_A+7lvs-assembly1_B:7qgs-assembly1_B+7lvs-assembly1_A', 'lig_seq_id': 0.964, 'lig_rmsd': 1.37, 'lig_aligned_length': 55, 'lig_qry_tm_score': 0.88068, 'lig_tgt_tm_score': 0.84041, 'rec_seq_id': 0.935, 'rec_rmsd': 0.99, 'rec_aligned_length': 93, 'rec_qry_tm_score': 0.94733, 'rec_tgt_tm_score': 0.91012, 'jaccard_index': 0.7348066298342542, 'tgt_lig_chain_name': '7lvs-assembly1_B', 'off_tgt_lig_chain_name': '7qgs-assembly1_A', 'lig_pair': '7lvs-assembly1_B+7qgs-assembly1_A', 'tgt_rec_chain_name': '7lvs-assembly1_A', 'off_tgt_rec_chain_name': '7qgs-assembly1_B', 'rec_pair': '7lvs-assembly1_A+7qgs-assembly1_B', 'pair': '7lvs-assembly1_B+7qgs-assembly1_A:7lvs-assembly1_A+7qgs-assembly1_B'}
    {'model_name': '7lvs-assembly1+7qgs-assembly1-on', 'auth_dsn_chain': 'B', 'model_cif': '/home/zwxie/experiments/sel_experiments/models/rgat_b64_a1_b09_t0001/7lvs-assembly1+7qgs-assembly1/on_tgt/af3_results/7lvs-assembly1/seed-19168_sample-2/model.cif', 'ranking_score': 1.04, 'ptm': 0.88, 'iptm': 0.87, 'dsn_chain_plddt': 87.05966265060239, 'dsn_chain_ptm': 0.64, 'dsn_seq': 'MDEAVLLDLIRALGLDKEKELPVLSYEDCAVNAPIASDADLLTGQEISDALDENK', 'mean_ipae': 4.75925023429286, 'mean_ipae_with_contact': 1.8598039213862936, 'min_ipae_with_contact': 0.9}
    {'model_name': '7lvs-assembly1+7qgs-assembly1-off', 'auth_dsn_chain': 'A', 'model_cif': '/home/zwxie/experiments/sel_experiments/models/rgat_b64_a1_b09_t0001/7lvs-assembly1+7qgs-assembly1/off_tgt/af3_results/7qgs-assembly1/seed-31276_sample-0/model.cif', 'ranking_score': 1.04, 'ptm': 0.88, 'iptm': 0.86, 'dsn_chain_plddt': 83.18212048192771, 'dsn_chain_ptm': 0.64, 'dsn_seq': 'MDEAVLLDLIRALGLDKEKELPVLSYEDCAVNAPIASDADLLTGQEISDALDENK', 'mean_ipae': 4.549931573793646, 'mean_ipae_with_contact': 1.86949999981305, 'min_ipae_with_contact': 0.9}



```python
def parse_relaxed_result(in_dir, strict=False):
    res_file = Path(in_dir) / 'relaxed.json'
    if not res_file.exists():
        if strict:
            raise ValueError()
        return
    res = json.load(open(res_file))
    return res

def _filter_value(res):
    return {k: v for k, v in res.items() if isinstance(v, Number)}
    
def build_df(in_root, names, strict=False):
    df = []
    for name in names:
        res = parse_relaxed_result(in_root / name, strict=False)
        if res is None:
            continue
        res = _filter_value(res)
        res['name'] = name
        df.append(res)
    df = pd.DataFrame(df)
    return df

def select_rows(dfs):
    df = []
    names = set.intersection(*[set(d['name']) for d in dfs])
    for name in names:
        all_rows = pd.concat([d[d['name'] == name] for d in dfs])
        best_row = all_rows.iloc[all_rows['binder_score'].argmin()]
        df.append(best_row)
    df = pd.DataFrame(df)
    return df

def print_summary(df):
      for c in df.select_dtypes(include=np.number).columns:
          print(c, len(df[c]), np.mean(df[c]))

def build_min_df(in_dir, names, dir_names = ['relaxed', 'r2', 'r3']):
    dfs = []
    for dir_name in dir_names:
        d = in_dir / dir_name
        if d.exists():
            dfs.append(build_df(d, names))
        else:
            print(f'Missing {d}')
    df = select_rows(dfs)
    return df
```


```python
def plot_outliers(x, y, names, xlabel, ylabel):
  # Detect outliers (z-score > 2)
  z_scores = np.abs(stats.zscore(np.column_stack([x, y])))
  outliers = (z_scores > 2).any(axis=1)

  # Plot
  plt.figure(figsize=(8, 6))
  sns.scatterplot(x=x, y=y)

  # Compute correlation
  r_pearson, p_pearson = stats.pearsonr(x, y)
  r_spearman, p_spearman = stats.spearmanr(x, y)

  # Add regression line
  sns.regplot(x=x, y=y, scatter=False, color='red', line_kws={'linestyle': '--'})

  # Add y=x line
  lims = [min(min(x), min(y)), max(max(x), max(y))]
  plt.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    
  # Add correlation text
  plt.text(0.05, 0.95, f'Pearson r = {r_pearson:.3f} (p = {p_pearson:.2e})\nSpearman ρ = {r_spearman:.3f} (p = {p_spearman:.2e})',
    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
    
  # Label outliers
  print('Outliers (by zscore)')
  for i, is_outlier in enumerate(outliers):
      if is_outlier:
          print(f'({x[i]}, {y[i]}, {i}, {names[i]})')
          # plt.annotate(f'({x[i]}, {y[i]}, {i}, {names[i]})', (x[i], y[i]),
          #              textcoords="offset points", xytext=(5, 5))

  lower_points = (np.array(x) - np.array(y)) < -10
  print(f'Lower scores: {np.sum(lower_points)}')
  # print(np.array(x))
  # print(np.array(y))
  # print(np.array(x) - np.array(y))
  for i, is_lower in enumerate(lower_points):
      if is_lower:
          print(f'({x[i]}, {y[i]}, {i}, {names[i]})') 
  plt.show()

  hi_points = (np.array(x) - np.array(y)) > 10
  print(f'Higher scores: {np.sum(hi_points)}')
  # print(np.array(x))
  # print(np.array(y))
  # print(np.array(x) - np.array(y))
  for i, is_hi in enumerate(hi_points):
      if is_hi:
          print(f'({x[i]}, {y[i]}, {i}, {names[i]})') 
  plt.show()

def order_values(results, names, fn=lambda res: float(res['diff_energy']['binder_score'])):
    values = [fn(results[n]) for n in names]
    return values

def get_summary(results, names):
    values = []
    for name in names:
        if name not in results:
            print(f'Missing {name}')
            continue
        d = results[name]
        v = {
            'sel_succ@-10': d['diff_energy']['binder_score'] < -10,
            'sel_succ@-5': d['diff_energy']['binder_score'] < -5,
            'sel_succ': d['diff_energy']['binder_score'] < 0,
            **{f'{k}_diff': v for k, v in d['diff_energy'].items() if k not in {'name'}}
        }
        values.append(v)
    df = pd.DataFrame(values)
    keys = df.columns
    summary = {}
    for k in keys:
        summary[k] = summarize_values(df[k].values)
    return summary

def summarize_values(values):
    desc = stats.describe(values)
    return {
      'count': desc.nobs,
      'avg': desc.mean,
      'std': np.sqrt(desc.variance),
      'min': desc.minmax[0],
      'max': desc.minmax[1],
      'median': np.median(values),
   }

def collect_summaries(named_results, pair_names):
    named_summaries = {name: get_summary(result, pair_names) for name, result in named_results.items()}
    summaries = []
    for name, summary in named_summaries.items():
        summaries.append({**{k: v['avg'] for k, v in summary.items()}, 'pair_name': name})
    summaries = pd.DataFrame(summaries)
    return summaries
```


```python
all_results = {}
native_energy_df = build_min_df(root_dir / 'results/native', relaxed_structure_names)
native_result = parse_pair_results(results_dir / 'native/', cluster_df)
native_diff_df = parse_energy_differences(native_result, native_energy_df)
all_results['native'] = native_result
```

    Load 174 pairs



```python
pair_names = filter_pairs(native_result)
print(len(pair_names))
```

    54



```python
rgat_b64_t0001_energy_df = build_min_df(root_dir / 'results/rgat_b64_t0001', relaxed_structure_names)
rgat_b64_t0001_result = parse_pair_results(results_dir / 'rgat_b64_t0001/', cluster_df)
rgat_b64_t0001_diff_df = parse_energy_differences(rgat_b64_t0001_result, rgat_b64_t0001_energy_df)
all_results['rgat_b64_t0001'] = rgat_b64_t0001_result
```

    Load 174 pairs



```python
def filter_pairs(result):
    out = []
    for k, d in result.items():
        cinfo = d['clu_info']
        ji = cinfo['jaccard_index']
        if ji < 0.5 and d['on_info']['tgt_chain_plddt'] > 70 and d['off_info']['tgt_chain_plddt'] > 70:
            out.append(k)
    return out
```


```python
rgat_b64_a1_b09_t0001_energy_df = build_min_df(root_dir / 'results/rgat_b64_a1_b09_t0001', relaxed_structure_names)
rgat_b64_a1_b09_t0001_diff_df = parse_energy_differences(rgat_b64_a1_b09_t0001_result, rgat_b64_a1_b09_t0001_energy_df)
all_results['rgat_b64_a1_b09_t0001'] = rgat_b64_a1_b09_t0001_result
```

# ProteinMPNN


```python
pmpnn_n002_t0001_energy_df = build_min_df(root_dir / 'results/pmpnn_n002_t0001', relaxed_structure_names)
pmpnn_n002_t0001_result = parse_pair_results(results_dir / 'pmpnn_n002_t0001/', cluster_df)
pmpnn_n002_t0001_diff_df = parse_energy_differences(pmpnn_n002_t0001_result, pmpnn_n002_t0001_energy_df)
all_results['pmpnn_n002_t0001'] = pmpnn_n002_t0001_result
```

    Load 174 pairs



```python
pmpnn_n010_t0001_energy_df = build_min_df(root_dir / 'results/pmpnn_n010_t0001', relaxed_structure_names)
pmpnn_n010_t0001_result = parse_pair_results(results_dir / 'pmpnn_n010_t0001/', cluster_df)
pmpnn_n010_t0001_diff_df = parse_energy_differences(pmpnn_n010_t0001_result, pmpnn_n010_t0001_energy_df)
all_results['pmpnn_n010_t0001'] = pmpnn_n010_t0001_result
```

    Load 174 pairs



```python
pmpnn_n020_t0001_energy_df = build_min_df(root_dir / 'results/pmpnn_n020_t0001', relaxed_structure_names)
pmpnn_n020_t0001_result = parse_pair_results(results_dir / 'pmpnn_n020_t0001/', cluster_df)
pmpnn_n020_t0001_diff_df = parse_energy_differences(pmpnn_n020_t0001_result, pmpnn_n020_t0001_energy_df)
all_results['pmpnn_n020_t0001'] = pmpnn_n020_t0001_result
```

    Load 174 pairs



```python
pmpnn_n030_t0001_energy_df = build_min_df(root_dir / 'results/pmpnn_n030_t0001', relaxed_structure_names)
pmpnn_n030_t0001_result = parse_pair_results(results_dir / 'pmpnn_n030_t0001/', cluster_df)
pmpnn_n030_t0001_diff_df = parse_energy_differences(pmpnn_n030_t0001_result, pmpnn_n030_t0001_energy_df)
all_results['pmpnn_n030_t0001'] = pmpnn_n030_t0001_result
```

    Load 174 pairs



```python
pifold_t0001_energy_df = build_min_df(root_dir / 'results/pifold_t0001', relaxed_structure_names)
pifold_t0001_result = parse_pair_results(results_dir / 'pifold_t0001/', cluster_df)
pifold_t0001_diff_df = parse_energy_differences(pifold_t0001_result, pifold_t0001_energy_df)
all_results['pifold_t0001'] = pifold_t0001_result
```

    Load 174 pairs



```python
esmif_t0001_energy_df = build_min_df(root_dir / 'results/esmif_t0001', relaxed_structure_names)
esmif_t0001_result = parse_pair_results(results_dir / 'esmif_t0001/', cluster_df)
esmif_t0001_diff_df = parse_energy_differences(esmif_t0001_result, esmif_t0001_energy_df)
all_results['esmif_t0001'] = esmif_t0001_result
```

    Load 174 pairs



```python
summaries_df = collect_summaries(all_results, pair_names)
summaries_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sel_succ@-10</th>
      <th>sel_succ@-5</th>
      <th>sel_succ</th>
      <th>binder_score_diff</th>
      <th>surface_hydrophobicity_diff</th>
      <th>interface_sc_diff</th>
      <th>interface_packstat_diff</th>
      <th>interface_dG_diff</th>
      <th>interface_dSASA_diff</th>
      <th>interface_dG_SASA_ratio_diff</th>
      <th>interface_fraction_diff</th>
      <th>interface_hydrophobicity_diff</th>
      <th>interface_nres_diff</th>
      <th>interface_interface_hbonds_diff</th>
      <th>interface_hbond_percentage_diff</th>
      <th>interface_delta_unsat_hbonds_diff</th>
      <th>interface_delta_unsat_hbonds_percentage_diff</th>
      <th>pair_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.259259</td>
      <td>0.333333</td>
      <td>0.500000</td>
      <td>2.425926</td>
      <td>-0.001111</td>
      <td>0.000556</td>
      <td>-0.004074</td>
      <td>1.120741</td>
      <td>-36.808333</td>
      <td>0.021296</td>
      <td>0.557407</td>
      <td>-0.602037</td>
      <td>-0.648148</td>
      <td>0.129630</td>
      <td>0.366852</td>
      <td>0.111111</td>
      <td>1.044444</td>
      <td>native</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.333333</td>
      <td>0.518519</td>
      <td>0.648148</td>
      <td>-3.681667</td>
      <td>-0.003519</td>
      <td>0.000370</td>
      <td>0.001667</td>
      <td>-0.340926</td>
      <td>-27.576111</td>
      <td>-0.052593</td>
      <td>-0.208333</td>
      <td>-0.990741</td>
      <td>0.425926</td>
      <td>-0.240741</td>
      <td>-0.514259</td>
      <td>-0.092593</td>
      <td>-0.712037</td>
      <td>rgat_b64_a1_b09_t0001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.314815</td>
      <td>0.444444</td>
      <td>0.537037</td>
      <td>0.615556</td>
      <td>-0.002593</td>
      <td>0.005000</td>
      <td>-0.001296</td>
      <td>-1.058704</td>
      <td>-13.606667</td>
      <td>-0.032407</td>
      <td>0.588148</td>
      <td>-0.891667</td>
      <td>-0.259259</td>
      <td>0.018519</td>
      <td>1.075556</td>
      <td>0.074074</td>
      <td>0.963333</td>
      <td>pmpnn_n002_t0001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.314815</td>
      <td>0.388889</td>
      <td>0.555556</td>
      <td>-0.719074</td>
      <td>0.002037</td>
      <td>-0.009630</td>
      <td>-0.006111</td>
      <td>-2.707593</td>
      <td>68.805185</td>
      <td>-0.041852</td>
      <td>1.242037</td>
      <td>0.382963</td>
      <td>0.814815</td>
      <td>0.481481</td>
      <td>0.866481</td>
      <td>-0.722222</td>
      <td>-0.962407</td>
      <td>pmpnn_n010_t0001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.314815</td>
      <td>0.444444</td>
      <td>0.537037</td>
      <td>-2.322963</td>
      <td>0.003889</td>
      <td>0.007037</td>
      <td>-0.003148</td>
      <td>-4.857037</td>
      <td>144.841481</td>
      <td>-0.045741</td>
      <td>2.773519</td>
      <td>0.791296</td>
      <td>1.962963</td>
      <td>1.629630</td>
      <td>1.886852</td>
      <td>0.685185</td>
      <td>1.558148</td>
      <td>pmpnn_n020_t0001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.277778</td>
      <td>0.462963</td>
      <td>0.611111</td>
      <td>-7.822037</td>
      <td>0.000926</td>
      <td>0.008148</td>
      <td>-0.007593</td>
      <td>0.092963</td>
      <td>-65.231481</td>
      <td>-0.040741</td>
      <td>-0.337222</td>
      <td>-2.185000</td>
      <td>-0.518519</td>
      <td>0.277778</td>
      <td>1.806667</td>
      <td>-1.203704</td>
      <td>-2.253333</td>
      <td>pmpnn_n030_t0001</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.277778</td>
      <td>0.407407</td>
      <td>0.555556</td>
      <td>-5.687593</td>
      <td>0.002222</td>
      <td>0.019630</td>
      <td>0.011481</td>
      <td>-0.762778</td>
      <td>24.218333</td>
      <td>-0.044630</td>
      <td>-1.908704</td>
      <td>1.065000</td>
      <td>0.629630</td>
      <td>0.222222</td>
      <td>NaN</td>
      <td>-0.555556</td>
      <td>NaN</td>
      <td>pifold_t0001</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.351852</td>
      <td>0.425926</td>
      <td>0.537037</td>
      <td>-1.000370</td>
      <td>0.000185</td>
      <td>0.020000</td>
      <td>0.011111</td>
      <td>-2.246111</td>
      <td>28.179630</td>
      <td>-0.067037</td>
      <td>1.473519</td>
      <td>-1.189444</td>
      <td>0.351852</td>
      <td>-0.129630</td>
      <td>-1.235556</td>
      <td>-0.574074</td>
      <td>-0.402407</td>
      <td>esmif_t0001</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.222222</td>
      <td>0.277778</td>
      <td>0.333333</td>
      <td>8.896296</td>
      <td>0.001852</td>
      <td>0.007778</td>
      <td>0.002778</td>
      <td>-1.172963</td>
      <td>15.810370</td>
      <td>-0.084815</td>
      <td>0.691111</td>
      <td>1.228333</td>
      <td>0.203704</td>
      <td>0.074074</td>
      <td>1.588704</td>
      <td>0.296296</td>
      <td>1.110556</td>
      <td>rgat_b64_t0001</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
