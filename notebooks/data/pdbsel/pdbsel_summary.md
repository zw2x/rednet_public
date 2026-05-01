```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
import json
from collections import defaultdict, OrderedDict
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
```


```python
def get_summary(df1, cluster_df, sel_pairs=None):
      results = []
      for model_name in df1['model_name'].unique():
          if model_name.endswith('-on'):
              pair_name = model_name.strip('-on')
              if sel_pairs and pair_name not in sel_pairs:
                  continue
              pair_row = cluster_df[cluster_df['pair_name'] == pair_name]
              assert len(pair_row) > 0
              pair_row = pair_row.iloc[0]

              row1 = df1[df1['model_name'] == model_name]

              if len(row1) == 0:
                  continue

              if float(pair_row['jaccard_index'] < 0.5):
                  row1_off = df1[df1['model_name'] == model_name.replace('-on', '-off')]
                  results.append({
                      'model_name': model_name,
                      'iptm_diff_df1': float(row1['iptm'].iloc[0]) - float(row1_off['iptm'].iloc[0]),
                      'iptm_sel_succ_df1': (float(row1['iptm'].iloc[0]) > 0.55) * (float(row1_off['iptm'].iloc[0]) < 0.55),
                      'iptm_succ_df1': float(float(row1['iptm'].iloc[0]) > 0.55),
                      'iptm_off_succ_df1': float(float(row1_off['iptm'].iloc[0]) > 0.55),
                      'rec_seq_id': pair_row['rec_seq_id'],
                      'jac_index': pair_row['jaccard_index'],
                  })

      results = pd.DataFrame(results)
      return results
```


```python
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

# cluster_df['pair_name'] = cluster_df['tgt_lig_id'].str.split('_').str[0] + '+' + cluster_df['off_tgt_lig_id'].str.split('_').str[0]
update_pair_model_(cluster_df)
```

    98



```python
print(get_summary(native_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean       -0.002037           0.037037       0.740741           0.740741   
    std         0.138610           0.190626       0.442343           0.442343   
    min        -0.350000           0.000000       0.000000           0.000000   
    25%        -0.027500           0.000000       0.250000           0.250000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.040000           0.000000       1.000000           1.000000   
    max         0.320000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
esmif_df = pd.read_csv(results_dir / 'esmif_t0001/best_summary.csv')
esmif_pair_df = group_pairs(esmif_df)
print(get_summary(esmif_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.022593           0.074074       0.777778           0.722222   
    std         0.129297           0.264351       0.419643           0.452109   
    min        -0.300000           0.000000       0.000000           0.000000   
    25%        -0.037500           0.000000       1.000000           0.000000   
    50%         0.020000           0.000000       1.000000           1.000000   
    75%         0.047500           0.000000       1.000000           1.000000   
    max         0.380000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
pifold_df = pd.read_csv(results_dir / 'pifold_t0001/best_summary.csv')
pifold_pair_df = group_pairs(pifold_df)
print(get_summary(pifold_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.001111           0.037037       0.722222           0.759259   
    std         0.126739           0.190626       0.452109           0.431548   
    min        -0.430000           0.000000       0.000000           0.000000   
    25%        -0.030000           0.000000       0.000000           1.000000   
    50%         0.005000           0.000000       1.000000           1.000000   
    75%         0.047500           0.000000       1.000000           1.000000   
    max         0.320000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
pmpnn_002_df = pd.read_csv(results_dir / 'pmpnn_n002_t0001/best_summary.csv')
print(get_summary(pmpnn_002_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.017963           0.074074       0.777778           0.759259   
    std         0.174273           0.264351       0.419643           0.431548   
    min        -0.530000           0.000000       0.000000           0.000000   
    25%        -0.037500           0.000000       1.000000           1.000000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.067500           0.000000       1.000000           1.000000   
    max         0.510000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
pmpnn_010_df = pd.read_csv(results_dir / 'pmpnn_n010_t0001/best_summary.csv')
print(get_summary(pmpnn_010_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.036667           0.018519       0.796296           0.796296   
    std         0.132437           0.136083       0.406533           0.406533   
    min        -0.220000           0.000000       0.000000           0.000000   
    25%        -0.010000           0.000000       1.000000           1.000000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.050000           0.000000       1.000000           1.000000   
    max         0.620000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
pmpnn_020_df = pd.read_csv(results_dir / 'pmpnn_n020_t0001/best_summary.csv')
print(get_summary(pmpnn_020_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.008889           0.055556       0.796296           0.759259   
    std         0.117291           0.231212       0.406533           0.431548   
    min        -0.370000           0.000000       0.000000           0.000000   
    25%        -0.027500           0.000000       1.000000           1.000000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.030000           0.000000       1.000000           1.000000   
    max         0.420000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
pmpnn_030_df = pd.read_csv(results_dir / 'pmpnn_n030_t0001/best_summary.csv')
print(get_summary(pmpnn_030_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.053519           0.074074       0.814815           0.759259   
    std         0.158688           0.264351       0.392095           0.431548   
    min        -0.300000           0.000000       0.000000           0.000000   
    25%        -0.017500           0.000000       1.000000           1.000000   
    50%         0.030000           0.000000       1.000000           1.000000   
    75%         0.080000           0.000000       1.000000           1.000000   
    max         0.570000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_df = pd.read_csv(results_dir / 'rgat_b64_t0001/best_summary.csv')
print(get_summary(rgat_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.010556           0.055556       0.740741           0.722222   
    std         0.187037           0.231212       0.442343           0.452109   
    min        -0.340000           0.000000       0.000000           0.000000   
    25%        -0.055000           0.000000       0.250000           0.000000   
    50%         0.000000           0.000000       1.000000           1.000000   
    75%         0.040000           0.000000       1.000000           1.000000   
    max         0.740000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  


# CD Hparams

Alpha = 1 and Noise level = 0.02


```python
rgat_a1_b09_t0001_cd_df = pd.read_csv(results_dir / 'rgat_b64_a1_b09_t0001/best_summary.csv')
print(get_summary(rgat_a1_b09_t0001_cd_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.022037           0.092593       0.722222           0.703704   
    std         0.190257           0.292582       0.452109           0.460911   
    min        -0.380000           0.000000       0.000000           0.000000   
    25%        -0.037500           0.000000       0.000000           0.000000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.050000           0.000000       1.000000           1.000000   
    max         0.680000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_b64_a2_b09_t0001 = pd.read_csv(results_dir / 'rgat_b64_a2_b09_t0001/best_summary.csv')
print(get_summary(rgat_b64_a2_b09_t0001, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.006481           0.074074       0.722222           0.740741   
    std         0.211679           0.264351       0.452109           0.442343   
    min        -0.600000           0.000000       0.000000           0.000000   
    25%        -0.050000           0.000000       0.000000           0.250000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.057500           0.000000       1.000000           1.000000   
    max         0.620000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_b64_a05_b09_t0001 = pd.read_csv(results_dir / 'rgat_b64_a05_b09_t0001/best_summary.csv')
print(get_summary(rgat_b64_a05_b09_t0001, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.007593           0.037037       0.759259           0.722222   
    std         0.151619           0.190626       0.431548           0.452109   
    min        -0.350000           0.000000       0.000000           0.000000   
    25%        -0.040000           0.000000       1.000000           0.000000   
    50%         0.010000           0.000000       1.000000           1.000000   
    75%         0.040000           0.000000       1.000000           1.000000   
    max         0.600000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_n002_a1_b07_t01_df = pd.read_csv(results_dir / 'rgat_n002_a1_b07_t01/best_summary.csv')
print(get_summary(rgat_n002_a1_b07_t01_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean       -0.013333           0.055556       0.740741           0.740741   
    std         0.150496           0.231212       0.442343           0.442343   
    min        -0.470000           0.000000       0.000000           0.000000   
    25%        -0.040000           0.000000       0.250000           0.250000   
    50%         0.000000           0.000000       1.000000           1.000000   
    75%         0.030000           0.000000       1.000000           1.000000   
    max         0.380000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_fa_n002_a2_b07_t0001 = pd.read_csv(results_dir / 'rgat_fa_n002_a2_b07_t0001/best_summary.csv')
print(get_summary(rgat_fa_n002_a2_b07_t0001, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean       -0.000185           0.055556       0.777778           0.777778   
    std         0.171832           0.231212       0.419643           0.419643   
    min        -0.370000           0.000000       0.000000           0.000000   
    25%        -0.057500           0.000000       1.000000           1.000000   
    50%         0.000000           0.000000       1.000000           1.000000   
    75%         0.047500           0.000000       1.000000           1.000000   
    max         0.740000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  


Alpha = 10 and Noise level = 0.02


```python
rgat_n002_a10_b07_t01_df = pd.read_csv(results_dir / 'rgat_n002_a10_b07_t01/best_summary.csv')
print(get_summary(rgat_n002_a10_b07_t01_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean       -0.012778           0.037037       0.703704           0.722222   
    std         0.140705           0.190626       0.460911           0.452109   
    min        -0.340000           0.000000       0.000000           0.000000   
    25%        -0.050000           0.000000       0.000000           0.000000   
    50%         0.000000           0.000000       1.000000           1.000000   
    75%         0.030000           0.000000       1.000000           1.000000   
    max         0.330000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_n002_a10_b07_t0001_df = pd.read_csv(results_dir / 'rgat_n002_a10_b07_t0001/best_summary.csv')
print(get_summary(rgat_n002_a10_b07_t0001_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean       -0.018704           0.055556       0.759259           0.777778   
    std         0.177839           0.231212       0.431548           0.419643   
    min        -0.450000           0.000000       0.000000           0.000000   
    25%        -0.055000           0.000000       1.000000           1.000000   
    50%         0.005000           0.000000       1.000000           1.000000   
    75%         0.050000           0.000000       1.000000           1.000000   
    max         0.600000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



```python
rgat_n002_a10_b09_t0001_df = pd.read_csv(results_dir / 'rgat_n002_a10_b09_t0001/best_summary.csv')
print(get_summary(rgat_n002_a10_b09_t0001_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.007407           0.055556       0.740741           0.740741   
    std         0.192992           0.231212       0.442343           0.442343   
    min        -0.680000           0.000000       0.000000           0.000000   
    25%        -0.045000           0.000000       0.250000           0.250000   
    50%         0.015000           0.000000       1.000000           1.000000   
    75%         0.060000           0.000000       1.000000           1.000000   
    max         0.570000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  


Alpha = 10 and Noise level = 0.2


```python
rgat_n02_a10_b07_df = pd.read_csv(results_dir / 'rgat_n02_a10_b07_t01/best_summary.csv')
print(get_summary(rgat_n02_a10_b07_df, cluster_df, sel_pairs=pairs).describe())
```

           iptm_diff_df1  iptm_sel_succ_df1  iptm_succ_df1  iptm_off_succ_df1  \
    count      54.000000          54.000000      54.000000          54.000000   
    mean        0.000370           0.055556       0.740741           0.740741   
    std         0.169727           0.231212       0.442343           0.442343   
    min        -0.390000           0.000000       0.000000           0.000000   
    25%        -0.030000           0.000000       0.250000           0.250000   
    50%         0.005000           0.000000       1.000000           1.000000   
    75%         0.047500           0.000000       1.000000           1.000000   
    max         0.600000           1.000000       1.000000           1.000000   
    
           rec_seq_id  jac_index  
    count   54.000000  54.000000  
    mean     0.590259   0.285912  
    std      0.200751   0.157473  
    min      0.034000   0.000000  
    25%      0.422250   0.149658  
    50%      0.634000   0.315726  
    75%      0.754750   0.432672  
    max      0.980000   0.489231  



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
rgat_b64_a1_b09_t0001_result = parse_pair_results(results_dir / 'rgat_b64_a1_b09_t0001/', cluster_df)
```

    Load 174 pairs



```python
native_result = parse_pair_results(results_dir / 'native/', cluster_df)
```

    Load 174 pairs



```python
rgat_b64_t0001_result = parse_pair_results(results_dir / 'rgat_b64_t0001/', cluster_df)
```

    Load 174 pairs



```python
list(native_result.items())[0][-1]['on_info']
```




    {'model_name': '7lvs-assembly1+7qgs-assembly1-on',
     'auth_dsn_chain': 'B',
     'model_cif': '/home/zwxie/experiments/sel_experiments/models/native/7lvs-assembly1+7qgs-assembly1/on_tgt/af3_results/7lvs-assembly1/seed-28143_sample-1/model.cif',
     'ranking_score': 1.05,
     'ptm': 0.87,
     'iptm': 0.87,
     'mean_ipae': 4.89965323335539,
     'mean_ipae_with_contact': 1.9874999998447265,
     'min_ipae_with_contact': 0.9,
     'dsn_chain_plddt': 85.84690531177829,
     'dsn_chain_ptm': 0.63,
     'dsn_seq': 'IDEEVLMSLVIEMGLDRIKELPELTSYDCEVNAPIQGSRNLLQGEELLRALDQVN',
     'tgt_chain_plddt': 87.15366402116402,
     'tgt_chain_ptm': 0.82,
     'tgt_seq': 'TGPTADPEKRKLIQQQLVLLLHAHKCQRREQANGEVRACSLPHCRTMKNVLNHMTHCQAGKACQVAHCASSRQIISHWKNCTRHDCPVCLPLKNASD'}




```python

```
