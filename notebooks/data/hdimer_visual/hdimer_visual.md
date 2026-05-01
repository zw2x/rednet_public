```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import OrderedDict
```


```python
hdimer_dir = Path("/home/zwxie/experiments/hdimer_experiments/")
hdimer_models_dir = hdimer_dir / 'models'
hdimer_results_dir = hdimer_dir / 'results'
assert hdimer_models_dir.exists() and hdimer_results_dir.exists()

all_dfs = OrderedDict()
```


```python
def calc_success_rate(df):
    is_success = (df['dsn_chain_plddt'] > 0.8) * (df['ptm'] > 0.55) * (df['iptm'] > 0.5)
    # is_success *= (df['min_ipae_with_contact'] < 5) * (df['min_ipae_with_contact'] > 0)
    return is_success
    
def print_summary(df, sel_model_names=None):
  # # Summary statistics
  # print("=" * 60)
  # print("SUMMARY STATISTICS")
  # print("=" * 60)
  # print(df.describe())

  if sel_model_names is not None:
      df = df[df['model_name'].isin(sel_model_names)]
      
  # print("\n" + "=" * 60)
  # print("COLUMN INFO")
  # print("=" * 60)
  # print(df.dtypes)
  # print(f"\nTotal rows: {len(df)}")

  # Key metrics summary
  print("\n" + "=" * 60)
  print("KEY METRICS")
  print("=" * 60)
  numeric_cols = ['ranking_score', 'ptm', 'iptm', 'dsn_chain_plddt',
                  'dsn_chain_ptm', 'mean_ipae', 'mean_ipae_with_contact', 'min_ipae_with_contact', 'success_rate', 'tgt_chain_plddt']

  for col in numeric_cols:
      if col in df.columns:
          print(f"{col:30s}: mean={df[col].mean():.3f}, median={df[col].median():.3f}, std={df[col].std():.3f}")

# Plot distributions
def plot_distribution(df):
  fig, axes = plt.subplots(3, 3, figsize=(12, 10))
  axes = axes.flatten()
  numeric_cols = ['ranking_score', 'ptm', 'iptm', 'dsn_chain_plddt',
                  'dsn_chain_ptm', 'mean_ipae', 'mean_ipae_with_contact', 'min_ipae_with_contact', 'tgt_chain_plddt']
  for i, col in enumerate(numeric_cols):
      if col in df.columns and i < len(axes):
          ax = axes[i]
          df[col].hist(ax=ax, bins=20, edgecolor='black', alpha=0.7)
          ax.axvline(df[col].mean(), color='red', linestyle='--', label='mean')
          ax.axvline(df[col].median(), color='green', linestyle='--', label='median')
          ax.set_title(col)
          ax.legend(fontsize=8)
          
def plot_all_dataframes(named_dfs, plot_type='hist'):
      """
      Plot distributions for multiple dataframes.

      Args:
          named_dfs: list of (name, df) tuples
          plot_type: 'hist', 'kde', 'violin', or 'box'
      """
      cols = ['ranking_score', 'ptm', 'iptm', 'dsn_chain_plddt', 'dsn_chain_ptm', 'tgt_chain_plddt']

      fig, axes = plt.subplots(2, 3, figsize=(12, 8))
      axes = axes.flatten()

      if plot_type in ['violin', 'box']:
          # Combine dataframes for seaborn
          dfs = []
          for label, df in named_dfs.items():
              df = df.copy()
              df['source'] = label
              dfs.append(df)
          df_all = pd.concat(dfs, ignore_index=True)

          for i, col in enumerate(cols):
              if plot_type == 'violin':
                  sns.violinplot(data=df_all, x='source', y=col, ax=axes[i])
              else:
                  sns.boxplot(data=df_all, x='source', y=col, ax=axes[i])
              axes[i].set_title(col)
              axes[i].tick_params(axis='x', rotation=45)

      else:
          # Overlay hist or kde
          for i, col in enumerate(cols):
              ax = axes[i]
              for label, df in named_dfs.items():
                  if plot_type == 'hist':
                      df[col].hist(ax=ax, bins=20, alpha=0.5, label=label, edgecolor='black')
                  else:  # kde
                      df[col].plot.kde(ax=ax, label=label)
              ax.set_title(col)
              ax.legend()

      plt.tight_layout()
      plt.show()

def plot_pairwise(named_df1, named_df2):
  cols = ['ranking_score', 'ptm', 'iptm', 'dsn_chain_plddt', 'tgt_chain_plddt', 'mean_ipae']

  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  axes = axes.flatten()

  n1, df1 = named_df1
  n2, df2 = named_df2
  for i, col in enumerate(cols):
      ax = axes[i]
      x, y = df1[col], df2[col]

      # Scatter
      ax.scatter(x, y, alpha=0.6, edgecolor='black', linewidth=0.5)

      # Diagonal line (y=x)
      lims = [min(x.min(), y.min()), max(x.max(), y.max())]
      ax.plot(lims, lims, 'r--', label='y=x')

      # Correlation
      r, p = stats.pearsonr(x, y)
      ax.set_title(f"{col}\nr={r:.3f}, p={p:.2e}")
      ax.set_xlabel(n1)
      ax.set_ylabel(n2)

  plt.tight_layout()
  plt.show()

```


```python
expr_name = 'native'
nat_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
nat_df['success_rate'] = calc_success_rate(nat_df)
print_summary(nat_df)
all_dfs[expr_name] = nat_df
sel_model_names = nat_df['model_name'][nat_df['tgt_chain_plddt'] > 70]
_old_n = len(sel_model_names)
sel_model_names = set(sel_model_names.unique())
print(len(sel_model_names), _old_n)
print_summary(nat_df, sel_model_names)
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.496, median=0.430, std=0.255
    ptm                           : mean=0.534, median=0.470, std=0.245
    iptm                          : mean=0.387, median=0.330, std=0.250
    dsn_chain_plddt               : mean=51.308, median=48.809, std=20.960
    dsn_chain_ptm                 : mean=0.184, median=0.030, std=0.220
    mean_ipae                     : mean=18.869, median=19.928, std=7.885
    mean_ipae_with_contact        : mean=1.761, median=0.000, std=2.897
    min_ipae_with_contact         : mean=646.084, median=1000.000, std=479.152
    success_rate                  : mean=0.290, median=0.000, std=0.456
    tgt_chain_plddt               : mean=62.606, median=59.288, std=22.638
    44 44
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.732, median=0.755, std=0.203
    ptm                           : mean=0.781, median=0.810, std=0.139
    iptm                          : mean=0.606, median=0.630, std=0.211
    dsn_chain_plddt               : mean=68.238, median=70.273, std=17.315
    dsn_chain_ptm                 : mean=0.143, median=0.020, std=0.228
    mean_ipae                     : mean=11.784, median=10.666, std=5.669
    mean_ipae_with_contact        : mean=3.094, median=3.051, std=2.976
    min_ipae_with_contact         : mean=297.527, median=3.350, std=460.167
    success_rate                  : mean=0.682, median=1.000, std=0.471
    tgt_chain_plddt               : mean=86.755, median=87.217, std=7.125



```python
expr_name = 'pmpnn_n002_t0001'
pmpnn_n002_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
pmpnn_n002_t0001_df['success_rate'] = calc_success_rate(pmpnn_n002_t0001_df)
print_summary(pmpnn_n002_t0001_df)
print_summary(pmpnn_n002_t0001_df, sel_model_names=sel_model_names)
all_dfs[expr_name] = pmpnn_n002_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.486, median=0.400, std=0.262
    ptm                           : mean=0.534, median=0.500, std=0.245
    iptm                          : mean=0.360, median=0.310, std=0.258
    dsn_chain_plddt               : mean=55.118, median=57.798, std=20.765
    dsn_chain_ptm                 : mean=0.208, median=0.030, std=0.246
    mean_ipae                     : mean=19.400, median=20.176, std=8.364
    mean_ipae_with_contact        : mean=2.024, median=0.000, std=3.315
    min_ipae_with_contact         : mean=627.590, median=1000.000, std=484.251
    success_rate                  : mean=0.299, median=0.000, std=0.460
    tgt_chain_plddt               : mean=62.214, median=60.935, std=23.049
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.688, median=0.725, std=0.241
    ptm                           : mean=0.753, median=0.780, std=0.171
    iptm                          : mean=0.554, median=0.590, std=0.240
    dsn_chain_plddt               : mean=67.484, median=68.793, std=15.950
    dsn_chain_ptm                 : mean=0.145, median=0.025, std=0.224
    mean_ipae                     : mean=13.062, median=11.651, std=7.038
    mean_ipae_with_contact        : mean=2.845, median=2.217, std=2.828
    min_ipae_with_contact         : mean=342.727, median=3.800, std=478.174
    success_rate                  : mean=0.591, median=1.000, std=0.497
    tgt_chain_plddt               : mean=84.976, median=88.480, std=10.786



```python
expr_name = 'pmpnn_n010_t0001'
pmpnn_n010_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
pmpnn_n010_t0001_df['success_rate'] = calc_success_rate(pmpnn_n010_t0001_df)
print_summary(pmpnn_n010_t0001_df)
print_summary(pmpnn_n010_t0001_df, sel_model_names)
all_dfs[expr_name] = pmpnn_n010_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.506, median=0.430, std=0.281
    ptm                           : mean=0.548, median=0.520, std=0.244
    iptm                          : mean=0.388, median=0.320, std=0.273
    dsn_chain_plddt               : mean=55.891, median=56.304, std=21.551
    dsn_chain_ptm                 : mean=0.213, median=0.030, std=0.252
    mean_ipae                     : mean=18.537, median=18.898, std=8.201
    mean_ipae_with_contact        : mean=1.457, median=0.000, std=2.788
    min_ipae_with_contact         : mean=673.890, median=1000.000, std=469.936
    success_rate                  : mean=0.299, median=0.000, std=0.460
    tgt_chain_plddt               : mean=63.677, median=63.000, std=22.891
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.736, median=0.740, std=0.235
    ptm                           : mean=0.778, median=0.835, std=0.146
    iptm                          : mean=0.606, median=0.595, std=0.232
    dsn_chain_plddt               : mean=70.412, median=70.875, std=16.563
    dsn_chain_ptm                 : mean=0.153, median=0.025, std=0.240
    mean_ipae                     : mean=11.870, median=10.874, std=6.361
    mean_ipae_with_contact        : mean=2.552, median=1.559, std=3.282
    min_ipae_with_contact         : mean=388.102, median=4.250, std=491.155
    success_rate                  : mean=0.659, median=1.000, std=0.479
    tgt_chain_plddt               : mean=86.722, median=88.141, std=7.850



```python
expr_name = 'pmpnn_n020_t0001'
pmpnn_n020_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
pmpnn_n020_t0001_df['success_rate'] = calc_success_rate(pmpnn_n020_t0001_df)
print_summary(pmpnn_n020_t0001_df)
print_summary(pmpnn_n020_t0001_df, sel_model_names)
all_dfs[expr_name] = pmpnn_n020_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.478, median=0.400, std=0.261
    ptm                           : mean=0.540, median=0.500, std=0.243
    iptm                          : mean=0.363, median=0.290, std=0.260
    dsn_chain_plddt               : mean=55.582, median=55.993, std=22.126
    dsn_chain_ptm                 : mean=0.216, median=0.030, std=0.258
    mean_ipae                     : mean=19.101, median=20.150, std=8.251
    mean_ipae_with_contact        : mean=1.645, median=0.000, std=3.183
    min_ipae_with_contact         : mean=692.749, median=1000.000, std=462.269
    success_rate                  : mean=0.243, median=0.000, std=0.431
    tgt_chain_plddt               : mean=62.585, median=58.888, std=22.199
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.688, median=0.735, std=0.243
    ptm                           : mean=0.751, median=0.805, std=0.184
    iptm                          : mean=0.559, median=0.590, std=0.256
    dsn_chain_plddt               : mean=68.892, median=71.326, std=17.835
    dsn_chain_ptm                 : mean=0.142, median=0.030, std=0.219
    mean_ipae                     : mean=12.811, median=11.410, std=7.285
    mean_ipae_with_contact        : mean=2.098, median=1.413, std=2.546
    min_ipae_with_contact         : mean=456.041, median=5.300, std=502.308
    success_rate                  : mean=0.523, median=1.000, std=0.505
    tgt_chain_plddt               : mean=84.665, median=88.830, std=11.645



```python
expr_name = 'pmpnn_n030_t0001'
pmpnn_n030_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
pmpnn_n030_t0001_df['success_rate'] = calc_success_rate(pmpnn_n030_t0001_df)
print_summary(pmpnn_n030_t0001_df)
print_summary(pmpnn_n030_t0001_df, sel_model_names)
all_dfs[expr_name] = pmpnn_n030_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.498, median=0.440, std=0.260
    ptm                           : mean=0.546, median=0.510, std=0.246
    iptm                          : mean=0.380, median=0.300, std=0.259
    dsn_chain_plddt               : mean=56.585, median=58.972, std=20.785
    dsn_chain_ptm                 : mean=0.221, median=0.030, std=0.264
    mean_ipae                     : mean=18.814, median=18.690, std=8.074
    mean_ipae_with_contact        : mean=2.670, median=0.000, std=4.535
    min_ipae_with_contact         : mean=600.084, median=1000.000, std=490.197
    success_rate                  : mean=0.299, median=0.000, std=0.460
    tgt_chain_plddt               : mean=63.450, median=62.435, std=22.343
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.703, median=0.765, std=0.222
    ptm                           : mean=0.759, median=0.815, std=0.172
    iptm                          : mean=0.571, median=0.550, std=0.228
    dsn_chain_plddt               : mean=69.155, median=69.054, std=15.646
    dsn_chain_ptm                 : mean=0.145, median=0.030, std=0.225
    mean_ipae                     : mean=12.622, median=11.665, std=6.116
    mean_ipae_with_contact        : mean=3.827, median=2.683, std=4.551
    min_ipae_with_contact         : mean=320.930, median=4.500, std=469.267
    success_rate                  : mean=0.568, median=1.000, std=0.501
    tgt_chain_plddt               : mean=85.635, median=89.296, std=9.885



```python
expr_name = 'solmpnn_t0001'
solmpnn_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
solmpnn_t0001_df['success_rate'] = calc_success_rate(solmpnn_t0001_df)
print_summary(solmpnn_t0001_df)
print_summary(solmpnn_t0001_df, sel_model_names)
all_dfs[expr_name] = solmpnn_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.499, median=0.430, std=0.268
    ptm                           : mean=0.545, median=0.500, std=0.245
    iptm                          : mean=0.386, median=0.320, std=0.261
    dsn_chain_plddt               : mean=54.790, median=54.972, std=21.019
    dsn_chain_ptm                 : mean=0.202, median=0.030, std=0.239
    mean_ipae                     : mean=18.826, median=20.202, std=8.078
    mean_ipae_with_contact        : mean=2.071, median=0.000, std=3.459
    min_ipae_with_contact         : mean=618.292, median=1000.000, std=486.580
    success_rate                  : mean=0.280, median=0.000, std=0.451
    tgt_chain_plddt               : mean=63.288, median=61.507, std=22.508
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.733, median=0.755, std=0.231
    ptm                           : mean=0.784, median=0.835, std=0.149
    iptm                          : mean=0.603, median=0.650, std=0.228
    dsn_chain_plddt               : mean=69.113, median=70.971, std=17.288
    dsn_chain_ptm                 : mean=0.146, median=0.020, std=0.232
    mean_ipae                     : mean=11.887, median=11.369, std=5.997
    mean_ipae_with_contact        : mean=3.221, median=2.213, std=3.191
    min_ipae_with_contact         : mean=275.050, median=3.600, std=449.077
    success_rate                  : mean=0.659, median=1.000, std=0.479
    tgt_chain_plddt               : mean=86.667, median=88.543, std=7.844



```python
expr_name = 'esmif_t0001'
esmif_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
esmif_t0001_df['success_rate'] = calc_success_rate(esmif_t0001_df)
print_summary(esmif_t0001_df)
print_summary(esmif_t0001_df, sel_model_names)
all_dfs[expr_name] = esmif_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.490, median=0.430, std=0.263
    ptm                           : mean=0.532, median=0.500, std=0.247
    iptm                          : mean=0.377, median=0.290, std=0.261
    dsn_chain_plddt               : mean=53.768, median=53.781, std=20.909
    dsn_chain_ptm                 : mean=0.202, median=0.030, std=0.239
    mean_ipae                     : mean=18.987, median=20.101, std=7.924
    mean_ipae_with_contact        : mean=1.481, median=0.000, std=2.678
    min_ipae_with_contact         : mean=683.192, median=1000.000, std=466.401
    success_rate                  : mean=0.271, median=0.000, std=0.447
    tgt_chain_plddt               : mean=62.497, median=60.766, std=22.833
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.710, median=0.760, std=0.228
    ptm                           : mean=0.765, median=0.800, std=0.163
    iptm                          : mean=0.585, median=0.655, std=0.242
    dsn_chain_plddt               : mean=67.351, median=68.980, std=16.890
    dsn_chain_ptm                 : mean=0.145, median=0.020, std=0.225
    mean_ipae                     : mean=12.581, median=12.323, std=6.684
    mean_ipae_with_contact        : mean=2.182, median=1.731, std=2.392
    min_ipae_with_contact         : mean=410.432, median=3.600, std=496.223
    success_rate                  : mean=0.614, median=1.000, std=0.493
    tgt_chain_plddt               : mean=85.753, median=88.941, std=9.327



```python
expr_name = 'pifold_t0001'
pifold_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
pifold_t0001_df['success_rate'] = calc_success_rate(pifold_t0001_df)
print_summary(pifold_t0001_df)
print_summary(pifold_t0001_df, sel_model_names)
all_dfs[expr_name] = pifold_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.475, median=0.400, std=0.255
    ptm                           : mean=0.519, median=0.470, std=0.247
    iptm                          : mean=0.364, median=0.300, std=0.245
    dsn_chain_plddt               : mean=50.292, median=48.430, std=19.572
    dsn_chain_ptm                 : mean=0.190, median=0.020, std=0.228
    mean_ipae                     : mean=19.530, median=20.135, std=7.609
    mean_ipae_with_contact        : mean=1.916, median=0.000, std=3.457
    min_ipae_with_contact         : mean=674.384, median=1000.000, std=469.224
    success_rate                  : mean=0.280, median=0.000, std=0.451
    tgt_chain_plddt               : mean=61.712, median=59.422, std=22.809
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.692, median=0.705, std=0.218
    ptm                           : mean=0.758, median=0.810, std=0.165
    iptm                          : mean=0.558, median=0.580, std=0.218
    dsn_chain_plddt               : mean=62.905, median=61.733, std=17.023
    dsn_chain_ptm                 : mean=0.142, median=0.020, std=0.224
    mean_ipae                     : mean=13.285, median=12.304, std=6.190
    mean_ipae_with_contact        : mean=1.934, median=0.000, std=2.849
    min_ipae_with_contact         : mean=524.198, median=1000.000, std=503.704
    success_rate                  : mean=0.636, median=1.000, std=0.487
    tgt_chain_plddt               : mean=85.020, median=88.149, std=9.839



```python
expr_name = 'rgat_b64_n002_a1_b07_t0001'
rgat_b64_n002_a1_b07_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
rgat_b64_n002_a1_b07_t0001_df['success_rate'] = calc_success_rate(rgat_b64_n002_a1_b07_t0001_df)
print_summary(rgat_b64_n002_a1_b07_t0001_df)
print_summary(rgat_b64_n002_a1_b07_t0001_df, sel_model_names)
all_dfs[expr_name] = rgat_b64_n002_a1_b07_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.490, median=0.460, std=0.257
    ptm                           : mean=0.540, median=0.490, std=0.249
    iptm                          : mean=0.371, median=0.300, std=0.257
    dsn_chain_plddt               : mean=52.600, median=51.617, std=20.980
    dsn_chain_ptm                 : mean=0.201, median=0.030, std=0.242
    mean_ipae                     : mean=18.904, median=19.841, std=8.051
    mean_ipae_with_contact        : mean=1.770, median=0.000, std=3.233
    min_ipae_with_contact         : mean=655.513, median=1000.000, std=476.062
    success_rate                  : mean=0.280, median=0.000, std=0.451
    tgt_chain_plddt               : mean=62.206, median=59.958, std=23.003
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.700, median=0.740, std=0.230
    ptm                           : mean=0.761, median=0.820, std=0.186
    iptm                          : mean=0.573, median=0.635, std=0.243
    dsn_chain_plddt               : mean=66.974, median=66.387, std=17.341
    dsn_chain_ptm                 : mean=0.142, median=0.025, std=0.234
    mean_ipae                     : mean=12.632, median=11.444, std=6.823
    mean_ipae_with_contact        : mean=2.909, median=2.294, std=3.199
    min_ipae_with_contact         : mean=343.011, median=4.050, std=477.970
    success_rate                  : mean=0.591, median=1.000, std=0.497
    tgt_chain_plddt               : mean=85.258, median=89.963, std=11.559



```python
expr_name = 'rgat_b64_n002_a1_b03_t0001'
rgat_b64_n002_a1_b03_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
rgat_b64_n002_a1_b03_t0001_df['success_rate'] = calc_success_rate(rgat_b64_n002_a1_b03_t0001_df)
print_summary(rgat_b64_n002_a1_b03_t0001_df)
print_summary(rgat_b64_n002_a1_b03_t0001_df, sel_model_names)
all_dfs[expr_name] = df_cd
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.487, median=0.450, std=0.257
    ptm                           : mean=0.531, median=0.490, std=0.243
    iptm                          : mean=0.365, median=0.300, std=0.254
    dsn_chain_plddt               : mean=51.799, median=49.729, std=20.451
    dsn_chain_ptm                 : mean=0.192, median=0.020, std=0.221
    mean_ipae                     : mean=19.158, median=19.710, std=8.119
    mean_ipae_with_contact        : mean=1.816, median=0.000, std=3.519
    min_ipae_with_contact         : mean=674.228, median=1000.000, std=469.450
    success_rate                  : mean=0.280, median=0.000, std=0.451
    tgt_chain_plddt               : mean=62.270, median=59.693, std=22.921
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.695, median=0.735, std=0.232
    ptm                           : mean=0.757, median=0.825, std=0.172
    iptm                          : mean=0.566, median=0.635, std=0.239
    dsn_chain_plddt               : mean=66.175, median=67.388, std=16.916
    dsn_chain_ptm                 : mean=0.142, median=0.020, std=0.222
    mean_ipae                     : mean=12.988, median=11.791, std=7.034
    mean_ipae_with_contact        : mean=2.502, median=1.789, std=3.701
    min_ipae_with_contact         : mean=433.509, median=4.950, std=499.571
    success_rate                  : mean=0.636, median=1.000, std=0.487
    tgt_chain_plddt               : mean=85.199, median=87.843, std=10.779



```python
expr_name = 'rgat_b64_n002_a1_b09_t0001'
rgat_b64_n002_a1_b09_t0001_df = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
rgat_b64_n002_a1_b09_t0001_df['success_rate'] = calc_success_rate(rgat_b64_n002_a1_b09_t0001_df)
print_summary(rgat_b64_n002_a1_b09_t0001_df)
print_summary(rgat_b64_n002_a1_b09_t0001_df, sel_model_names)
all_dfs[expr_name] = rgat_b64_n002_a1_b09_t0001_df
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.487, median=0.390, std=0.258
    ptm                           : mean=0.532, median=0.460, std=0.242
    iptm                          : mean=0.370, median=0.310, std=0.254
    dsn_chain_plddt               : mean=53.115, median=54.141, std=21.273
    dsn_chain_ptm                 : mean=0.198, median=0.030, std=0.233
    mean_ipae                     : mean=19.070, median=19.696, std=8.115
    mean_ipae_with_contact        : mean=1.621, median=0.000, std=2.921
    min_ipae_with_contact         : mean=664.646, median=1000.000, std=473.176
    success_rate                  : mean=0.299, median=0.000, std=0.460
    tgt_chain_plddt               : mean=61.862, median=58.916, std=22.948
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.694, median=0.705, std=0.236
    ptm                           : mean=0.754, median=0.800, std=0.181
    iptm                          : mean=0.571, median=0.605, std=0.243
    dsn_chain_plddt               : mean=67.710, median=68.917, std=16.894
    dsn_chain_ptm                 : mean=0.139, median=0.030, std=0.215
    mean_ipae                     : mean=12.752, median=11.643, std=7.143
    mean_ipae_with_contact        : mean=2.323, median=1.929, std=2.566
    min_ipae_with_contact         : mean=387.870, median=3.400, std=491.337
    success_rate                  : mean=0.682, median=1.000, std=0.471
    tgt_chain_plddt               : mean=84.877, median=87.483, std=11.628



```python
expr_name = 'rgat_b64_n002_t0001'
df_t0001 = pd.read_csv(hdimer_results_dir / expr_name / 'best_summary.csv')
df_t0001['success_rate'] = calc_success_rate(df_t0001)
print_summary(df_t0001)
print_summary(df_t0001, sel_model_names)
all_dfs[expr_name] = df_t0001
```

    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.495, median=0.450, std=0.259
    ptm                           : mean=0.542, median=0.490, std=0.242
    iptm                          : mean=0.371, median=0.310, std=0.254
    dsn_chain_plddt               : mean=53.455, median=52.806, std=20.308
    dsn_chain_ptm                 : mean=0.206, median=0.030, std=0.244
    mean_ipae                     : mean=19.162, median=19.590, std=7.972
    mean_ipae_with_contact        : mean=1.803, median=0.000, std=3.276
    min_ipae_with_contact         : mean=655.507, median=1000.000, std=476.070
    success_rate                  : mean=0.252, median=0.000, std=0.436
    tgt_chain_plddt               : mean=62.449, median=61.584, std=22.343
    
    ============================================================
    KEY METRICS
    ============================================================
    ranking_score                 : mean=0.697, median=0.685, std=0.247
    ptm                           : mean=0.762, median=0.815, std=0.170
    iptm                          : mean=0.562, median=0.600, std=0.251
    dsn_chain_plddt               : mean=66.048, median=65.410, std=17.715
    dsn_chain_ptm                 : mean=0.145, median=0.030, std=0.232
    mean_ipae                     : mean=13.182, median=12.943, std=7.325
    mean_ipae_with_contact        : mean=2.606, median=1.557, std=2.907
    min_ipae_with_contact         : mean=365.355, median=3.550, std=485.296
    success_rate                  : mean=0.568, median=1.000, std=0.501
    tgt_chain_plddt               : mean=84.927, median=86.552, std=10.484



```python
def plot_cm(y_true, y_pred):
  labels = ['Failed', 'Success']
  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot(cmap='Blues')
  plt.xlabel("Predicted")
  plt.ylabel("Native")
  plt.title('Confusion Matrix')
  plt.show()

def rank_row(df_native, df_pmpnn):
  df_native['key'] = df_native['model_name'] + '_' + df_native['auth_dsn_chain']
  df_pmpnn['key'] = df_pmpnn['model_name'] + '_' + df_pmpnn['auth_dsn_chain']

  # Find keys unique to each dataframe
  native_keys = set(df_native['key'])
  pmpnn_keys = set(df_pmpnn['key'])
  only_in_native = native_keys - pmpnn_keys
  only_in_pmpnn = pmpnn_keys - native_keys

  print(f"Total rows in native: {len(df_native)}")
  print(f"Total rows in pmpnn: {len(df_pmpnn)}")
  print(f"Common rows: {len(native_keys & pmpnn_keys)}")

  print(f"=== Only in native ({len(only_in_native)}) ===")
  if only_in_native:
      print(df_native[df_native['key'].isin(only_in_native)][['model_name', 'auth_dsn_chain']])
  else:
      print("None")

  print()

  print(f"=== Only in pmpnn ({len(only_in_pmpnn)}) ===")
  if only_in_pmpnn:
      print(df_pmpnn[df_pmpnn['key'].isin(only_in_pmpnn)][['model_name', 'auth_dsn_chain']])
  else:
      print("None")

  df_pmpnn_sorted = df_pmpnn.set_index('key').loc[df_native['key']].reset_index()
  return df_pmpnn_sorted
```


```python
# parse energetics
expr_name = 'native'
nat_dir = hdimer_results_dir / expr_name / 'relaxed'
assert nat_dir.exists()
print(list(nat_dir.iterdir())[0])
```

    /home/zwxie/experiments/hdimer_experiments/results/native/relaxed/8ivz-assembly1_A-2:C_C



```python

```
