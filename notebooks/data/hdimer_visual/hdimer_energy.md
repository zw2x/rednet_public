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
import json
from numbers import Number
```


```python
hdimer_dir = Path("/home/zwxie/experiments/hdimer_experiments/")
hdimer_models_dir = hdimer_dir / 'models'
hdimer_results_dir = hdimer_dir / 'results'
assert hdimer_models_dir.exists() and hdimer_results_dir.exists()

all_dfs = OrderedDict()
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
hdimer_names = sorted([p.stem for p in nat_dir.iterdir()])
print(len(hdimer_names), hdimer_names[:1])
```

    100 ['7q98-assembly1_A:C_C']



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


def fuse_dfs(dfs):
    return select_rows(dfs)
```


```python
# nat_df1 = build_df(hdimer_results_dir / 'native/relaxed', hdimer_names)
# nat_df2 = build_df(hdimer_results_dir / 'native/r2', hdimer_names)
# nat_df3 = build_df(hdimer_results_dir / 'native/r3', hdimer_names)

# nat_df = select_rows([nat_df1, nat_df2, nat_df3])

nat_df = build_min_df(hdimer_results_dir / 'native/', hdimer_names)
nat_names = list(nat_df['name'])
print(len(nat_names))
print_summary(nat_df)
# print(nat_df)
```

    92
    binder_score 92 -170.74989130434784
    surface_hydrophobicity 92 0.43576086956521737
    interface_sc 92 0.646304347826087
    interface_packstat 92 0.5325000000000001
    interface_dG 92 -53.04380434782608
    interface_dSASA 92 1906.7077173913044
    interface_dG_SASA_ratio 92 -2.6816304347826088
    interface_fraction 92 104.80815217391304
    interface_hydrophobicity 92 47.91815217391304
    interface_nres 92 19.956521739130434
    interface_interface_hbonds 92 6.913043478260869
    interface_hbond_percentage 92 45.95626373626375
    interface_delta_unsat_hbonds 92 2.967391304347826
    interface_delta_unsat_hbonds_percentage 92 18.017582417582418


# Rednet and contrastive decoding


```python
rgat_b64_n002_t0001_df = build_min_df(hdimer_results_dir / "rgat_b64_n002_t0001", nat_names)
print_summary(rgat_b64_n002_t0001_df)
```

    binder_score 91 -179.8837362637363
    surface_hydrophobicity 91 0.4464835164835164
    interface_sc 91 0.6597802197802197
    interface_packstat 91 0.5394505494505495
    interface_dG 91 -52.49241758241757
    interface_dSASA 91 1868.2913186813187
    interface_dG_SASA_ratio 91 -2.7667032967032967
    interface_fraction 91 111.29802197802199
    interface_hydrophobicity 91 48.93351648351649
    interface_nres 91 19.417582417582416
    interface_interface_hbonds 91 6.208791208791209
    interface_hbond_percentage 91 44.803
    interface_delta_unsat_hbonds 91 2.4395604395604398
    interface_delta_unsat_hbonds_percentage 91 15.396888888888888



```python
rgat_names = list(rgat_b64_n002_t0001_df['name'])
print(len(rgat_names))
```

    91



```python
rgat_b64_n002_a1_b09_t0001_df = build_min_df(hdimer_results_dir / "rgat_b64_n002_a1_b09_t0001", rgat_names)
print_summary(rgat_b64_n002_a1_b09_t0001_df)
```

    binder_score 91 -182.26153846153846
    surface_hydrophobicity 91 0.4425274725274725
    interface_sc 91 0.6628571428571427
    interface_packstat 91 0.5369230769230768
    interface_dG 91 -54.465934065934064
    interface_dSASA 91 1894.6878021978025
    interface_dG_SASA_ratio 91 -2.7828571428571425
    interface_fraction 91 112.55285714285712
    interface_hydrophobicity 91 49.23296703296703
    interface_nres 91 19.703296703296704
    interface_interface_hbonds 91 7.010989010989011
    interface_hbond_percentage 91 48.22561797752809
    interface_delta_unsat_hbonds 91 2.6263736263736264
    interface_delta_unsat_hbonds_percentage 91 14.774269662921347



```python
rgat_b64_n002_a1_b03_t0001_df = build_min_df(hdimer_results_dir / "rgat_b64_n002_a1_b03_t0001", rgat_names)
print_summary(rgat_b64_n002_a1_b03_t0001_df)
```

    binder_score 91 -133.23659340659339
    surface_hydrophobicity 91 0.45241758241758234
    interface_sc 91 0.6503296703296704
    interface_packstat 91 0.5409890109890111
    interface_dG 91 -52.63098901098901
    interface_dSASA 91 1894.3713186813188
    interface_dG_SASA_ratio 91 -2.7639560439560436
    interface_fraction 91 143.94285714285712
    interface_hydrophobicity 91 48.44538461538461
    interface_nres 91 18.35164835164835
    interface_interface_hbonds 91 6.472527472527473
    interface_hbond_percentage 91 47.45241758241758
    interface_delta_unsat_hbonds 91 2.868131868131868
    interface_delta_unsat_hbonds_percentage 91 20.370109890109887



```python
rgat_b64_n020_e025_t0001_df = build_min_df(hdimer_results_dir / "rgat_b64_n020_e025_t0001", rgat_names)
print_summary(rgat_b64_n020_e025_t0001_df)
```

    binder_score 91 -149.47318681318683
    surface_hydrophobicity 91 0.48703296703296706
    interface_sc 91 0.6561538461538462
    interface_packstat 91 0.5438461538461539
    interface_dG 91 -52.1321978021978
    interface_dSASA 91 1829.8325274725275
    interface_dG_SASA_ratio 91 -2.758241758241758
    interface_fraction 91 135.39989010989012
    interface_hydrophobicity 91 51.89043956043956
    interface_nres 91 18.285714285714285
    interface_interface_hbonds 91 6.21978021978022
    interface_hbond_percentage 91 44.463111111111104
    interface_delta_unsat_hbonds 91 2.5714285714285716
    interface_delta_unsat_hbonds_percentage 91 16.05011111111111



```python
rgat_b64_n020_e025_cd_a1_b09_t0001_df = build_min_df(hdimer_results_dir / "rgat_b64_n020_e025_cd_a1_b09_t0001", rgat_names)
print_summary(rgat_b64_n020_e025_cd_a1_b09_t0001_df)
```

    binder_score 91 -143.38747252747254
    surface_hydrophobicity 91 0.49725274725274715
    interface_sc 91 0.6578021978021978
    interface_packstat 91 0.5453846153846155
    interface_dG 91 -55.96923076923077
    interface_dSASA 91 1978.796703296703
    interface_dG_SASA_ratio 91 -2.748901098901099
    interface_fraction 91 125.50747252747252
    interface_hydrophobicity 91 53.28076923076923
    interface_nres 91 19.593406593406595
    interface_interface_hbonds 91 6.868131868131868
    interface_hbond_percentage 91 45.914
    interface_delta_unsat_hbonds 91 2.8461538461538463
    interface_delta_unsat_hbonds_percentage 91 17.012999999999998


# Baselines


```python
pifold_t0001_df = build_min_df(hdimer_results_dir / "pifold_t0001", rgat_names)
print_summary(pifold_t0001_df)
```

    binder_score 91 -179.55109890109887
    surface_hydrophobicity 91 0.4747252747252747
    interface_sc 91 0.6659340659340659
    interface_packstat 91 0.5286813186813186
    interface_dG 91 -55.18703296703296
    interface_dSASA 91 1870.6557142857143
    interface_dG_SASA_ratio 91 -2.9059340659340664
    interface_fraction 91 193.62703296703293
    interface_hydrophobicity 91 52.55186813186813
    interface_nres 91 18.978021978021978
    interface_interface_hbonds 91 6.065934065934066
    interface_hbond_percentage 91 43.486888888888885
    interface_delta_unsat_hbonds 91 2.912087912087912
    interface_delta_unsat_hbonds_percentage 91 19.320777777777778



```python
esmif_t0001_df = build_min_df(hdimer_results_dir / "esmif_t0001", rgat_names)
print_summary(esmif_t0001_df)
```

    binder_score 91 -181.27439560439558
    surface_hydrophobicity 91 0.4599999999999999
    interface_sc 91 0.6447252747252746
    interface_packstat 91 0.5254945054945056
    interface_dG 91 -54.965714285714284
    interface_dSASA 91 1947.7462637362642
    interface_dG_SASA_ratio 91 -2.678021978021978
    interface_fraction 91 115.88483516483515
    interface_hydrophobicity 91 48.48813186813188
    interface_nres 91 19.923076923076923
    interface_interface_hbonds 91 6.5054945054945055
    interface_hbond_percentage 91 43.19078651685394
    interface_delta_unsat_hbonds 91 2.78021978021978
    interface_delta_unsat_hbonds_percentage 91 17.26370786516854


# ProteinMPNN


```python
pmpnn_n002_t0001_df = build_min_df(hdimer_results_dir / "pmpnn_n002_t0001", rgat_names)
print_summary(pmpnn_n002_t0001_df)
```

    binder_score 91 -184.89054945054946
    surface_hydrophobicity 91 0.4276923076923077
    interface_sc 91 0.6557142857142857
    interface_packstat 91 0.5474725274725275
    interface_dG 91 -46.97505494505494
    interface_dSASA 91 1682.1179120879115
    interface_dG_SASA_ratio 91 -2.729120879120879
    interface_fraction 91 107.30098901098899
    interface_hydrophobicity 91 51.18791208791209
    interface_nres 91 17.34065934065934
    interface_interface_hbonds 91 5.43956043956044
    interface_hbond_percentage 91 40.30033333333333
    interface_delta_unsat_hbonds 91 2.5164835164835164
    interface_delta_unsat_hbonds_percentage 91 17.617333333333335



```python
pmpnn_n010_t0001_df = build_min_df(hdimer_results_dir / "pmpnn_n010_t0001", rgat_names)
print_summary(pmpnn_n010_t0001_df)
```

    binder_score 91 -188.26527472527476
    surface_hydrophobicity 91 0.47868131868131863
    interface_sc 91 0.6730769230769231
    interface_packstat 91 0.5383516483516483
    interface_dG 91 -53.81472527472528
    interface_dSASA 91 1894.3295604395607
    interface_dG_SASA_ratio 91 -2.7597802197802195
    interface_fraction 91 113.58824175824178
    interface_hydrophobicity 91 51.614505494505494
    interface_nres 91 20.032967032967033
    interface_interface_hbonds 91 5.769230769230769
    interface_hbond_percentage 91 38.38477777777777
    interface_delta_unsat_hbonds 91 2.78021978021978
    interface_delta_unsat_hbonds_percentage 91 17.132666666666665



```python
pmpnn_n020_t0001_df = build_min_df(hdimer_results_dir / "pmpnn_n020_t0001", rgat_names)
print_summary(pmpnn_n020_t0001_df)
```

    binder_score 91 -188.091978021978
    surface_hydrophobicity 91 0.4951648351648353
    interface_sc 91 0.656043956043956
    interface_packstat 91 0.5392307692307692
    interface_dG 91 -48.27000000000001
    interface_dSASA 91 1756.2994505494505
    interface_dG_SASA_ratio 91 -2.691538461538462
    interface_fraction 91 109.31901098901099
    interface_hydrophobicity 91 53.12208791208792
    interface_nres 91 18.21978021978022
    interface_interface_hbonds 91 5.802197802197802
    interface_hbond_percentage 91 39.73955555555555
    interface_delta_unsat_hbonds 91 2.6263736263736264
    interface_delta_unsat_hbonds_percentage 91 16.573888888888888



```python
pmpnn_n030_t0001_df = build_min_df(hdimer_results_dir / "pmpnn_n030_t0001", rgat_names)
print_summary(pmpnn_n030_t0001_df)
```

    binder_score 91 -195.1256043956044
    surface_hydrophobicity 91 0.4517582417582417
    interface_sc 91 0.654065934065934
    interface_packstat 91 0.5382417582417582
    interface_dG 91 -48.840439560439556
    interface_dSASA 91 1733.6113186813188
    interface_dG_SASA_ratio 91 -2.7614285714285716
    interface_fraction 91 111.92879120879121
    interface_hydrophobicity 91 52.48274725274726
    interface_nres 91 17.846153846153847
    interface_interface_hbonds 91 5.747252747252747
    interface_hbond_percentage 91 41.634157303370785
    interface_delta_unsat_hbonds 91 2.67032967032967
    interface_delta_unsat_hbonds_percentage 91 19.20044943820225



```python
solmpnn_n020_t0001_df = build_min_df(hdimer_results_dir / "solmpnn_t0001", rgat_names)
print_summary(solmpnn_t0001_df)
```

    binder_score 91 -139.99054945054945
    surface_hydrophobicity 91 0.46076923076923076
    interface_sc 91 0.6565934065934065
    interface_packstat 91 0.5289010989010989
    interface_dG 91 -49.87
    interface_dSASA 91 1803.2352747252744
    interface_dG_SASA_ratio 91 -2.661318681318681
    interface_fraction 91 124.5746153846154
    interface_hydrophobicity 91 50.39527472527473
    interface_nres 91 17.10989010989011
    interface_interface_hbonds 91 6.153846153846154
    interface_hbond_percentage 91 44.37055555555555
    interface_delta_unsat_hbonds 91 2.3626373626373627
    interface_delta_unsat_hbonds_percentage 91 15.501222222222221



```python
sel_nat_df = build_min_df(hdimer_results_dir / 'native/', rgat_names)
print_summary(sel_nat_df)
```

    binder_score 91 -172.42065934065934
    surface_hydrophobicity 91 0.43351648351648353
    interface_sc 91 0.6468131868131869
    interface_packstat 91 0.5331868131868133
    interface_dG 91 -53.35
    interface_dSASA 91 1918.3335164835169
    interface_dG_SASA_ratio 91 -2.6784615384615384
    interface_fraction 91 105.04087912087911
    interface_hydrophobicity 91 47.50285714285714
    interface_nres 91 20.0989010989011
    interface_interface_hbonds 91 6.967032967032967
    interface_hbond_percentage 91 46.149444444444455
    interface_delta_unsat_hbonds 91 2.978021978021978
    interface_delta_unsat_hbonds_percentage 91 17.900333333333332



```python
# fused_df = fuse_dfs([pmpnn_n002_t0001_df, rgat_b64_n002_a1_b09_t0001_df, rgat_b64_n002_t0001_df])
fused_df = fuse_dfs([
    rgat_b64_n002_a1_b09_t0001_df,
    # rgat_b64_n002_a1_b07_t0001_df,
    # rgat_b64_n002_a1_b03_t0001_df,
    rgat_b64_n002_t0001_df,
])
print_summary(fused_df)
```

    binder_score 91 -188.0351648351648
    surface_hydrophobicity 91 0.44087912087912084
    interface_sc 91 0.6741758241758242
    interface_packstat 91 0.5465934065934066
    interface_dG 91 -56.658571428571435
    interface_dSASA 91 1965.9604395604395
    interface_dG_SASA_ratio 91 -2.8846153846153846
    interface_fraction 91 120.64923076923077
    interface_hydrophobicity 91 50.08934065934067
    interface_nres 91 20.186813186813186
    interface_interface_hbonds 91 7.3076923076923075
    interface_hbond_percentage 91 48.99252747252747
    interface_delta_unsat_hbonds 91 2.5934065934065935
    interface_delta_unsat_hbonds_percentage 91 14.267692307692307


# check sample outliers


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
  for i, is_outlier in enumerate(outliers):
      if is_outlier:
          print(f'({x[i]}, {y[i]}, {i}, {names[i]})')
          # plt.annotate(f'({x[i]}, {y[i]}, {i}, {names[i]})', (x[i], y[i]),
          #              textcoords="offset points", xytext=(5, 5))

  plt.show()
```


```python
plot_outliers(
    pmpnn_n002_t0001_df['binder_score'].values,
    sel_nat_df['binder_score'].values,
    sel_nat_df['name'], 'pmpnn', 'native'
)
# print(sel_nat_df)
```

    (-1095.28, -1123.52, 14, 7umi-assembly1_A:B_A)
    (-772.44, -681.78, 16, 7yn3-assembly1_A:C_A)
    (-1271.56, -1137.75, 20, 8itg-assembly1_A-5:B-5_A-5)
    (-709.19, -812.65, 64, 8b6g-assembly1_I:J_I)
    (-780.35, -782.47, 70, 8cjz-assembly1_E:J_E)
    (-753.83, -740.25, 75, 7q9a-assembly1_C:E_C)



    
![png](output_26_1.png)
    



```python
plot_outliers(
    esmif_t0001_df['binder_score'].values, sel_nat_df['binder_score'].values, sel_nat_df['name'], 'esmif', 'native'
)
```

    (-1160.31, -1123.52, 14, 7umi-assembly1_A:B_A)
    (-807.85, -681.78, 16, 7yn3-assembly1_A:C_A)
    (-1198.99, -1137.75, 20, 8itg-assembly1_A-5:B-5_A-5)
    (-703.57, -812.65, 64, 8b6g-assembly1_I:J_I)
    (-829.75, -782.47, 70, 8cjz-assembly1_E:J_E)
    (-728.84, -740.25, 75, 7q9a-assembly1_C:E_C)



    
![png](output_27_1.png)
    



```python
plot_outliers(
    rgat_b64_n002_t0001_df['binder_score'].values, sel_nat_df['binder_score'].values, sel_nat_df['name'], 'rgat', 'native'
)
```

    (-1129.59, -1123.52, 14, 7umi-assembly1_A:B_A)
    (-755.69, -681.78, 16, 7yn3-assembly1_A:C_A)
    (-1096.23, -1137.75, 20, 8itg-assembly1_A-5:B-5_A-5)
    (-583.14, -812.65, 64, 8b6g-assembly1_I:J_I)
    (-753.64, -782.47, 70, 8cjz-assembly1_E:J_E)
    (-732.7, -740.25, 75, 7q9a-assembly1_C:E_C)



    
![png](output_28_1.png)
    



```python
plot_outliers(
    rgat_b64_n002_a1_b09_t0001_df['binder_score'].values, rgat_b64_n002_t0001_df['binder_score'].values, 
    sel_nat_df['name'], 'rgat', 'rgat-cd')
```

    (-1113.25, -1129.59, 14, 7umi-assembly1_A:B_A)
    (-824.14, -755.69, 16, 7yn3-assembly1_A:C_A)
    (-1050.58, -1096.23, 20, 8itg-assembly1_A-5:B-5_A-5)
    (-745.03, -583.14, 64, 8b6g-assembly1_I:J_I)
    (-831.54, -753.64, 70, 8cjz-assembly1_E:J_E)
    (-729.18, -732.7, 75, 7q9a-assembly1_C:E_C)



    
![png](output_29_1.png)
    



```python
plot_outliers(
    rgat_b64_n002_a1_b09_t0001_df['binder_score'].values, pmpnn_n002_t0001_df['binder_score'].values, sel_nat_df['name'], 'rgat', 'pmpnn'
)
```

    (-1113.25, -1095.28, 14, 7umi-assembly1_A:B_A)
    (-824.14, -772.44, 16, 7yn3-assembly1_A:C_A)
    (-1050.58, -1271.56, 20, 8itg-assembly1_A-5:B-5_A-5)
    (-745.03, -709.19, 64, 8b6g-assembly1_I:J_I)
    (-831.54, -780.35, 70, 8cjz-assembly1_E:J_E)
    (-729.18, -753.83, 75, 7q9a-assembly1_C:E_C)



    
![png](output_30_1.png)
    



```python
plot_outliers(fused_df['binder_score'].values, sel_nat_df['binder_score'].values, sel_nat_df['name'], 'fused', 'native')
```

    (-1129.59, -1123.52, 14, 7umi-assembly1_A:B_A)
    (-824.14, -681.78, 16, 7yn3-assembly1_A:C_A)
    (-1147.11, -1137.75, 20, 8itg-assembly1_A-5:B-5_A-5)
    (-745.03, -812.65, 64, 8b6g-assembly1_I:J_I)
    (-831.54, -782.47, 70, 8cjz-assembly1_E:J_E)
    (-732.7, -740.25, 75, 7q9a-assembly1_C:E_C)



    
![png](output_31_1.png)
    



```python

```
