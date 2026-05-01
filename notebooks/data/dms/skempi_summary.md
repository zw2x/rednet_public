```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import math

from scipy import stats
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

from collections import OrderedDict
from omegaconf import OmegaConf

import torch

from rednet.data import SkempiDataset
from rednet.lightning.base_task import build_task
from rednet.common_utils import move_to_cuda
```

    /home/zwxie/miniconda3/envs/prot/lib/python3.11/site-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      _torch_pytree._register_pytree_node(



```python
def get_df(dataset, file_ids):
      df = []
      for file_id in file_ids:
          start, end = dataset.get_index_range(file_id)

          df.extend([{
            'file_id': dataset[i]['file_id'],
            'score': dataset[i]['score'].item(),
            'score_wt': dataset[i]['score_wt'].item(),
            'delta': dataset[i]['score'].item() - dataset[i]['score_wt'].item(),
          } for i in range(start, end+1)])
      df = pd.DataFrame(df)
      df['file_id'] = df['file_id'].astype(str)

      # Scatter plot
      g = sns.lmplot(data=df, x='score_wt', y='score', hue='file_id', height=6, aspect=1.2,
                     fit_reg=False, legend=False)  # disable default legend
      _min = min(min(df['score_wt']), min(df['score']))
      _max = max(max(df['score_wt']), max(df['score']))
      plt.plot([_min, _max], [_min, _max], 'k--', alpha=0.5)

      # Add compact legend outside plot
      n_cols = max(1, len(df['file_id'].unique()) // 20)
      plt.legend(title='file_id', fontsize=7, title_fontsize=8,
                 ncol=n_cols, loc='center left', bbox_to_anchor=(1.02, 0.5))
      # plt.tight_layout()
      plt.show()

      # Histogram: ratio of rows with delta < 0 per file_id
      ratio_df = df.groupby('file_id')['delta'].agg(
          lambda x: (x < 0).sum() / len(x)
      ).reset_index(name='ratio_negative')
      print(len(list(ratio_df[ratio_df['ratio_negative'] > 0].groupby('file_id'))))

      plt.figure(figsize=(8, max(5, len(ratio_df) * 0.3)))
      sns.barplot(data=ratio_df, y='file_id', x='ratio_negative', orient='h')
      plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
      plt.xlabel('Ratio of delta < 0')
      plt.ylabel('file_id')
      plt.title('Proportion of samples where score < score_wt')
      # plt.xticks(rotation=45, ha='right')
      # plt.tight_layout()
      plt.show()

      return df

def get_data_ids(file_ids):
    data_ids = []
    for file_id in file_ids:
        start, end = dataset.get_index_range(file_id)
        data_ids += list(range(start, end+1))
    return data_ids

def get_file_ids(dataset, min_num_variants=1):
    def _get_num(file_id):
        start, end = dataset.get_index_range(file_id)
        return end - start + 1
    return [f for f in dataset.file_ids if _get_num(f) >= min_num_variants]
```


```python
cfg_file = Path("/home/zwxie/codes/rednet/configs/test_exptl/skempiv2.yaml")
cfg = OmegaConf.load(cfg_file)
# cfg = OmegaConf.create({**cfg.data.config})
# print(cfg)
dataset = SkempiDataset(cfg.data.config, use_mut_chain_id=True)
print(len(dataset))
```

    2026-01-30 22:20:46,677 - rednet.data.dms_dataset - WARNING - empty assay for 1QRN
    2026-01-30 22:20:46,898 - rednet.data.dms_dataset - WARNING - empty assay for 1QSF
    2026-01-30 22:20:49,486 - rednet.data.dms_dataset - WARNING - empty assay for 5DWU
    2026-01-30 22:21:10,553 - rednet.data.dms_dataset - WARNING - empty assay for 3BIW


    6786
    126 339



```python
def agg_by_file_id(df):
    results = []
    # 'll': design binder
    # 'll_mut': mutated regions
    # 'll_ref': ll - wt_ll of mutated regions
    # 'll_global': complex
    pred_keys = ['ll', 'll_mt', 'll_ref', 'll_global', 'cd_ll_ref', 'cd_ll']
    for file_id, subset in df.groupby('file_id'):
        top_k = min(10, len(subset))
        subset_result = {}
        for pred_key in pred_keys:
            y_pred = subset[pred_key].values
            y_true = subset['-logkd'].values - subset['-logkd_wt'].values

            mask = ~np.isnan(y_pred)
            if not np.all(mask):
                nan_count = np.sum(~mask)
                print(f"y_pred contains {nan_count} NaN values, excluding from calculation")
                y_pred = y_pred[mask]
                y_true = y_true[mask]

            spearman_r, _ = spearmanr(y_pred, y_true)
            assert not np.isnan(spearman_r), y_pred
            # Kendall's tau
            kendall_tau, _ = kendalltau(y_pred, y_true)

            # NDCG (needs non-negative relevance scores)
            y_true_shifted = y_true - y_true.min() + 1
            ndcg = ndcg_score([y_true_shifted], [y_pred], k=top_k)

            subset_result.update({
                f'{pred_key}-spearman': spearman_r,
                f'{pred_key}-kendall': kendall_tau,
                f'{pred_key}-ndcg': ndcg,
            })
        subset_result.update({'n': len(subset), 'file_id': file_id})
        results.append(subset_result)
    results = pd.DataFrame(results)
    return results
 
```


```python
_res_file = Path("/home/zwxie/experiments/skempi_experiments") / "pmpnn_n002" / "results.csv"
assert _res_file.exists()
pmpnn_n002_df = pd.read_csv(_res_file)
print(pmpnn_n002_df.describe())
pmpnn_n002_df_g = agg_by_file_id(pmpnn_n002_df)
print(pmpnn_n002_df_g.describe())
```

                  nres     full_idx   within_idx       -logkd    -logkd_wt  \
    count  5705.000000  5705.000000  5705.000000  5705.000000  5705.000000   
    mean    437.536722  3355.069763    68.550044    17.930981    20.194813   
    std     195.229356  1980.823314    73.848609     5.127743     5.010680   
    min      57.000000     0.000000     0.000000     3.206453     9.596003   
    25%     291.000000  1621.000000    13.000000    13.847001    17.504391   
    50%     392.000000  3418.000000    36.000000    18.325371    20.192638   
    75%     583.000000  5038.000000   104.000000    21.465603    24.134514   
    max     946.000000  6785.000000   294.000000    36.148212    35.647438   
    
                  nmut  num_dsn_res           ll        ll_mt       ll_ref  \
    count  5705.000000  5705.000000  5705.000000  5705.000000  5705.000000   
    mean      1.689746   150.383348    -1.409653    -3.843262    -2.057460   
    std       1.713574    96.465838     0.318018     1.188315     1.894348   
    min       1.000000     5.000000    -2.757601    -6.609410    -6.470608   
    25%       1.000000    58.000000    -1.660333    -4.827355    -3.408881   
    50%       1.000000   129.000000    -1.358652    -3.878473    -1.989296   
    75%       2.000000   210.000000    -1.139047    -3.004929    -0.681046   
    max      27.000000   749.000000    -0.689226    -0.133803     4.299958   
    
             ll_global        ub_ll     ub_ll_mt    ub_ll_ref  ub_ll_global  \
    count  5705.000000  5705.000000  5705.000000  5705.000000   5705.000000   
    mean     -1.393115    -1.533053    -3.510436    -1.178307     -1.533053   
    std       0.275317     0.364486     1.048182     1.716759      0.364486   
    min      -2.188695    -4.125803    -6.494713    -6.408570     -4.125803   
    25%      -1.623768    -1.748117    -4.251237    -2.156485     -1.748117   
    50%      -1.351790    -1.487392    -3.437262    -0.952787     -1.487392   
    75%      -1.209950    -1.268981    -2.774475    -0.044761     -1.268981   
    max      -0.871997    -0.751843    -0.345073     4.068725     -0.751843   
    
                 cd_ll    cd_ll_ref  
    count  5705.000000  5705.000000  
    mean      0.123400    -0.879153  
    std       0.194399     1.689794  
    min      -0.222114    -8.806583  
    25%       0.035694    -1.655429  
    50%       0.079431    -0.496347  
    75%       0.152710     0.141633  
    max       2.134657     5.113363  
           ll-spearman  ll-kendall     ll-ndcg  ll_mt-spearman  ll_mt-kendall  \
    count   110.000000  110.000000  110.000000      110.000000     110.000000   
    mean      0.164711    0.120207    0.779527        0.230820       0.166501   
    std       0.288162    0.209030    0.122998        0.288970       0.210313   
    min      -0.553270   -0.405840    0.392910       -0.625710      -0.459019   
    25%      -0.024261   -0.013955    0.709384        0.068858       0.052212   
    50%       0.184149    0.119253    0.794949        0.254349       0.171879   
    75%       0.338246    0.252165    0.872075        0.392851       0.284348   
    max       0.878788    0.733333    0.990307        0.835187       0.681189   
    
           ll_mt-ndcg  ll_ref-spearman  ll_ref-kendall  ll_ref-ndcg  \
    count  110.000000       110.000000      110.000000   110.000000   
    mean     0.797788         0.258409        0.184244     0.803722   
    std      0.120232         0.287196        0.218632     0.118130   
    min      0.488622        -0.687661       -0.524593     0.489854   
    25%      0.714838         0.088618        0.039382     0.740140   
    50%      0.815537         0.272670        0.203737     0.817887   
    75%      0.886537         0.431569        0.309566     0.896257   
    max      0.989879         0.842424        0.709091     0.983787   
    
           ll_global-spearman  ll_global-kendall  ll_global-ndcg  \
    count          110.000000         110.000000      110.000000   
    mean             0.173726           0.123054        0.778689   
    std              0.243206           0.176794        0.123086   
    min             -0.447489          -0.325221        0.454568   
    25%              0.034957           0.012184        0.698417   
    50%              0.191213           0.132896        0.787962   
    75%              0.361793           0.256887        0.872108   
    max              0.745455           0.636364        0.978708   
    
           cd_ll_ref-spearman  cd_ll_ref-kendall  cd_ll_ref-ndcg  cd_ll-spearman  \
    count          110.000000         110.000000      110.000000      110.000000   
    mean             0.242531           0.176913        0.781984        0.102003   
    std              0.303526           0.221236        0.129721        0.290368   
    min             -0.551648          -0.362637        0.352497       -0.511384   
    25%              0.058892           0.036856        0.710779       -0.109597   
    50%              0.289361           0.195699        0.809459        0.105851   
    75%              0.468299           0.338810        0.880345        0.295195   
    max              0.804510           0.644444        0.974475        0.781818   
    
           cd_ll-kendall  cd_ll-ndcg           n  
    count     110.000000  110.000000  110.000000  
    mean        0.073909    0.764686   51.863636  
    std         0.212326    0.122728   67.183096  
    min        -0.351155    0.460372   10.000000  
    25%        -0.089217    0.686046   17.000000  
    50%         0.057162    0.783103   27.000000  
    75%         0.207403    0.861280   46.000000  
    max         0.600000    0.964765  295.000000  



```python
rgat_n002_res_file = Path("/home/zwxie/experiments/skempi_experiments") / "rgat_b64_n002_e028" / "results.csv"
assert rgat_n002_res_file.exists()
rgat_n002_df = pd.read_csv(rgat_n002_res_file)
rgat_n002_df_g = agg_by_file_id(rgat_n002_df)

_res_file = Path("/home/zwxie/experiments/skempi_experiments") / "rgat_b64_n0" / "results.csv"
assert _res_file.exists()
rgat_n0_df = pd.read_csv(_res_file)
rgat_n0_df_g = agg_by_file_id(rgat_n0_df)
```


```python
print('rgat_n0', rgat_n0_df_g.describe())
print('rgat_n002', rgat_n002_df_g.describe())
```

    rgat_n0        ll-spearman  ll-kendall     ll-ndcg  ll_mt-spearman  ll_mt-kendall  \
    count   110.000000  110.000000  110.000000      110.000000     110.000000   
    mean      0.179256    0.131686    0.785621        0.213979       0.151487   
    std       0.321966    0.237470    0.123289        0.253329       0.192042   
    min      -0.575567   -0.442039    0.323516       -0.459446      -0.345455   
    25%      -0.037640   -0.047917    0.712854        0.074226       0.044737   
    50%       0.239249    0.164148    0.799067        0.231538       0.145730   
    75%       0.399448    0.291823    0.880430        0.356853       0.245456   
    max       0.909091    0.781818    0.992439        0.844291       0.709091   
    
           ll_mt-ndcg  ll_ref-spearman  ll_ref-kendall  ll_ref-ndcg  \
    count  110.000000       110.000000      110.000000   110.000000   
    mean     0.801119         0.222116        0.162279     0.796390   
    std      0.112627         0.272608        0.202310     0.121518   
    min      0.464077        -0.472527       -0.333333     0.442803   
    25%      0.734936         0.047993        0.036745     0.725786   
    50%      0.815221         0.243358        0.171046     0.805240   
    75%      0.893140         0.363887        0.259258     0.886031   
    max      0.983820         0.909091        0.781818     0.992439   
    
           ll_global-spearman  ll_global-kendall  ll_global-ndcg  \
    count          110.000000         110.000000      110.000000   
    mean             0.228063           0.168279        0.800300   
    std              0.312569           0.233137        0.118276   
    min             -0.470588          -0.333333        0.446202   
    25%             -0.021482          -0.013782        0.734385   
    50%              0.256272           0.176132        0.822390   
    75%              0.431177           0.311993        0.883812   
    max              0.909091           0.781818        0.992439   
    
           cd_ll_ref-spearman  cd_ll_ref-kendall  cd_ll_ref-ndcg  cd_ll-spearman  \
    count          110.000000         110.000000      110.000000      110.000000   
    mean             0.255965           0.181819        0.779796        0.228949   
    std              0.291477           0.219254        0.147837        0.319208   
    min             -0.567647          -0.418182        0.295792       -0.677550   
    25%              0.110590           0.062504        0.726988        0.021499   
    50%              0.283895           0.197514        0.818256        0.262694   
    75%              0.414351           0.294197        0.891315        0.401927   
    max              0.872078           0.735499        0.995018        0.943605   
    
           cd_ll-kendall  cd_ll-ndcg           n  
    count     110.000000  110.000000  110.000000  
    mean        0.166309    0.784103   51.863636  
    std         0.242826    0.141060   67.183096  
    min        -0.533744    0.323516   10.000000  
    25%         0.010828    0.724534   17.000000  
    50%         0.185535    0.815212   27.000000  
    75%         0.275253    0.867551   46.000000  
    max         0.838727    0.996495  295.000000  
    rgat_n002        ll-spearman  ll-kendall     ll-ndcg  ll_mt-spearman  ll_mt-kendall  \
    count   110.000000  110.000000  110.000000      110.000000     110.000000   
    mean      0.207055    0.153667    0.790955        0.224126       0.162034   
    std       0.323496    0.237204    0.129419        0.292013       0.214923   
    min      -0.524691   -0.361235    0.323516       -0.661765      -0.500000   
    25%      -0.072896   -0.043595    0.714933        0.034176       0.035597   
    50%       0.228180    0.162304    0.803573        0.245480       0.166782   
    75%       0.436391    0.304034    0.895090        0.415469       0.312603   
    max       0.915152    0.733333    0.988185        0.818182       0.644444   
    
           ll_mt-ndcg  ll_ref-spearman  ll_ref-kendall  ll_ref-ndcg  \
    count  110.000000       110.000000      110.000000   110.000000   
    mean     0.812289         0.241064        0.174509     0.809033   
    std      0.113414         0.263511        0.197420     0.117970   
    min      0.437428        -0.346154       -0.230769     0.425416   
    25%      0.754707         0.021603        0.022088     0.741275   
    50%      0.826213         0.256232        0.171871     0.828670   
    75%      0.889296         0.414305        0.306103     0.896108   
    max      0.987448         0.939394        0.822222     0.995629   
    
           ll_global-spearman  ll_global-kendall  ll_global-ndcg  \
    count          110.000000         110.000000      110.000000   
    mean             0.257520           0.187691        0.802874   
    std              0.298233           0.223634        0.119332   
    min             -0.379121          -0.256410        0.437927   
    25%             -0.001796           0.023875        0.741009   
    50%              0.262163           0.189708        0.813318   
    75%              0.464157           0.338354        0.894870   
    max              0.915152           0.733333        0.988185   
    
           cd_ll_ref-spearman  cd_ll_ref-kendall  cd_ll_ref-ndcg  cd_ll-spearman  \
    count          110.000000         110.000000      110.000000      110.000000   
    mean             0.276354           0.199226        0.782765        0.255760   
    std              0.293214           0.219649        0.149030        0.341852   
    min             -0.781818          -0.636364        0.268124       -0.627273   
    25%              0.117713           0.074085        0.732606        0.016665   
    50%              0.297147           0.207757        0.819609        0.286697   
    75%              0.488941           0.354012        0.880709        0.518464   
    max              0.803302           0.686493        0.994389        0.845455   
    
           cd_ll-kendall  cd_ll-ndcg           n  
    count     110.000000  110.000000  110.000000  
    mean        0.188354    0.787209   51.863636  
    std         0.254435    0.147236   67.183096  
    min        -0.490909    0.302054   10.000000  
    25%         0.016948    0.710207   17.000000  
    50%         0.194850    0.820208   27.000000  
    75%         0.382705    0.901169   46.000000  
    max         0.724632    0.995866  295.000000  



```python
_res_file = Path("/home/zwxie/experiments/skempi_experiments") / "esmif" / "results.csv"
assert _res_file.exists()
esmif_df = pd.read_csv(_res_file)
esmif_df_g = agg_by_file_id(esmif_df)
print(esmif_df_g.describe())
```

    y_pred contains 1 NaN values, excluding from calculation
    y_pred contains 1 NaN values, excluding from calculation
           ll-spearman  ll-kendall     ll-ndcg  ll_mt-spearman  ll_mt-kendall  \
    count   110.000000  110.000000  110.000000      110.000000     110.000000   
    mean      0.162143    0.116931    0.773605        0.197220       0.137251   
    std       0.314192    0.226948    0.134845        0.257014       0.184543   
    min      -0.628137   -0.436914    0.346087       -0.725275      -0.564103   
    25%      -0.080317   -0.058510    0.696940        0.031606       0.024555   
    50%       0.194429    0.136628    0.795024        0.201491       0.134436   
    75%       0.408560    0.272838    0.866177        0.376055       0.251606   
    max       0.827273    0.672727    0.990647        0.802808       0.606838   
    
           ll_mt-ndcg  ll_ref-spearman  ll_ref-kendall  ll_ref-ndcg  \
    count  110.000000       110.000000      110.000000   110.000000   
    mean     0.791099         0.203129        0.142540     0.780153   
    std      0.122932         0.282043        0.208596     0.141027   
    min      0.459211        -0.879121       -0.717949     0.378100   
    25%      0.717044         0.037299        0.026226     0.694700   
    50%      0.793441         0.203803        0.127094     0.805643   
    75%      0.899908         0.368232        0.269253     0.892715   
    max      0.986217         0.845455        0.672727     0.986217   
    
           ll_global-spearman  ll_global-kendall  ll_global-ndcg  \
    count          110.000000         110.000000      110.000000   
    mean             0.242478           0.174925        0.786266   
    std              0.294613           0.217401        0.142384   
    min             -0.883082          -0.683885        0.315548   
    25%              0.054944           0.040721        0.705755   
    50%              0.279902           0.192503        0.817423   
    75%              0.449573           0.318606        0.900105   
    max              0.827273           0.672727        0.990647   
    
           cd_ll_ref-spearman  cd_ll_ref-kendall  cd_ll_ref-ndcg  cd_ll-spearman  \
    count          110.000000         110.000000      110.000000      110.000000   
    mean             0.146944           0.101751        0.773887        0.145820   
    std              0.266984           0.190024        0.120110        0.291382   
    min             -0.615385          -0.538462        0.459281       -0.560432   
    25%             -0.026025          -0.028506        0.699419       -0.036346   
    50%              0.178384           0.113725        0.773616        0.199493   
    75%              0.333586           0.225445        0.861699        0.342401   
    max              0.709091           0.527273        0.971066        0.757576   
    
           cd_ll-kendall  cd_ll-ndcg           n  
    count     110.000000  110.000000  110.000000  
    mean        0.107840    0.772457   51.863636  
    std         0.204318    0.124682   67.183096  
    min        -0.405840    0.396974   10.000000  
    25%        -0.021595    0.686404   17.000000  
    50%         0.147653    0.781342   27.000000  
    75%         0.244612    0.877035   46.000000  
    max         0.555556    0.976662  295.000000  



```python
_res_file = Path("/home/zwxie/experiments/skempi_experiments") / "pifold" / "results.csv"
assert _res_file.exists()
pifold_df = pd.read_csv(_res_file)
pifold_df_g = agg_by_file_id(pifold_df)
print(pifold_df_g.describe())
```

           ll-spearman  ll-kendall     ll-ndcg  ll_mt-spearman  ll_mt-kendall  \
    count   110.000000  110.000000  110.000000      110.000000     110.000000   
    mean      0.173686    0.131715    0.783716       -0.025537      -0.015535   
    std       0.320458    0.235286    0.131212        0.249910       0.181750   
    min      -0.512130   -0.352941    0.401076       -0.776645      -0.613765   
    25%      -0.084179   -0.051907    0.713715       -0.153512      -0.106884   
    50%       0.198768    0.125950    0.802775       -0.018910      -0.017615   
    75%       0.399024    0.268795    0.883305        0.125675       0.095564   
    max       0.954545    0.854545    0.993387        0.596940       0.486453   
    
           ll_mt-ndcg  ll_ref-spearman  ll_ref-kendall  ll_ref-ndcg  \
    count  110.000000       110.000000      110.000000   110.000000   
    mean     0.715766        -0.123691       -0.089915     0.693694   
    std      0.133593         0.266121        0.190542     0.139162   
    min      0.390948        -0.718943       -0.511111     0.313250   
    25%      0.626856        -0.276421       -0.190607     0.591572   
    50%      0.734663        -0.147629       -0.110195     0.703634   
    75%      0.816478         0.035300        0.027561     0.792281   
    max      0.971611         0.680434        0.511478     0.974750   
    
           ll_global-spearman  ll_global-kendall  ll_global-ndcg  \
    count          110.000000         110.000000      110.000000   
    mean            -0.170426          -0.127519        0.666037   
    std              0.269569           0.199473        0.150063   
    min             -0.803302          -0.683885        0.279015   
    25%             -0.340900          -0.244382        0.542421   
    50%             -0.168708          -0.133318        0.684049   
    75%             -0.003644           0.017015        0.766844   
    max              0.602994           0.445904        0.953231   
    
           cd_ll_ref-spearman  cd_ll_ref-kendall  cd_ll_ref-ndcg  cd_ll-spearman  \
    count          110.000000         110.000000      110.000000      110.000000   
    mean            -0.009634          -0.000655        0.714150        0.147505   
    std              0.299695           0.212618        0.146676        0.340529   
    min             -0.588235          -0.416667        0.282799       -0.590909   
    25%             -0.209796          -0.137892        0.610885       -0.104554   
    50%             -0.042637          -0.020274        0.721680        0.151936   
    75%              0.163676           0.130542        0.822410        0.401088   
    max              0.811564           0.629512        0.974474        0.899588   
    
           cd_ll-kendall  cd_ll-ndcg           n  
    count     110.000000  110.000000  110.000000  
    mean        0.105655    0.753376   51.863636  
    std         0.250653    0.137578   67.183096  
    min        -0.490909    0.436409   10.000000  
    25%        -0.083794    0.670063   17.000000  
    50%         0.110024    0.755674   27.000000  
    75%         0.290569    0.867236   46.000000  
    max         0.812920    0.990171  295.000000  



```python

```


```python

```
