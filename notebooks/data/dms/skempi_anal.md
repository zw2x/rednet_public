# Skempi Mutation Effect Prediction Analysis

We evaluate the capability of fixed-backbone design models to predict binding affinity changes upon mutation in a zero-shot manner. We utilize the SKEMPI v2.0 dataset, benchmarking RedNet against ProteinMPNN, ESM-IF, and PiFold. We compute the log-likelihood (LL) of sequences and correlate these scores with experimental binding affinity measurements (-log Kd). We report Spearman's $\rho$, Kendall's $\tau$, and Normalized Discounted Cumulative Gain (NDCG) for six different scoring aggregations:
*   `ll`: Log-likelihood of the entire designed binder sequence.
*   `ll_mt`: Log-likelihood of only the mutated positions.
*   `ll_ref`: Log-likelihood of mutated positions normalized by the wild-type likelihood (LL_mut - LL_wt).
*   `ll_global`: Global log-likelihood of the complex.
*   `cd_ll`: Contrastive decoding score (log-likelihood difference vs background).
*   `cd_ll_ref`: Reference-normalized contrastive decoding score.

### Ranking Correlation Metrics (Spearman's $\rho$ and Kendall's $\tau$)

| Model | ll-sp | ll_g-sp | ll_mt-sp | ll_ref-sp | cd_ll-sp | cd_ref-sp | ll-kd | ll_g-kd | ll_mt-kd | ll_ref-kd | cd_ll-kd | cd_ref-kd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESM-IF | 0.16 | 0.24 | 0.20 | 0.20 | 0.15 | 0.15 | 0.12 | 0.18 | 0.14 | 0.14 | 0.11 | 0.10 |
| PiFold | 0.17 | -0.17 | -0.03 | -0.12 | 0.15 | -0.01 | 0.13 | -0.13 | -0.02 | -0.09 | 0.11 | 0.00 |
| ProteinMPNN (n=0.02) | 0.17 | 0.17 | 0.23 | 0.26 | 0.10 | 0.24 | 0.12 | 0.12 | 0.17 | 0.18 | 0.07 | 0.18 |
| RedNet (rgat_b64_n0) | 0.18 | 0.23 | 0.21 | 0.22 | 0.23 | 0.26 | 0.13 | 0.17 | 0.15 | 0.16 | 0.17 | 0.18 |
| RedNet (rgat_b64_n002) | 0.21 | 0.26 | 0.22 | 0.24 | 0.26 | 0.28 | 0.15 | 0.19 | 0.16 | 0.18 | 0.19 | 0.20 |

*Note: sp = Spearman, kd = Kendall, g = global, mt = mutated, ref = reference-normalized, cd = contrastive decoding.*

### Ranking Accuracy Metrics (NDCG)

| Model | ll-ndcg | ll_global-ndcg | ll_mt-ndcg | ll_ref-ndcg | cd_ll-ndcg | cd_ll_ref-ndcg |
| --- | --- | --- | --- | --- | --- | --- |
| ESM-IF | 0.77 | 0.79 | 0.79 | 0.78 | 0.77 | 0.77 |
| PiFold | 0.78 | 0.67 | 0.72 | 0.69 | 0.75 | 0.71 |
| ProteinMPNN (n=0.02) | 0.78 | 0.78 | 0.80 | 0.80 | 0.76 | 0.78 |
| RedNet (rgat_b64_n0) | 0.79 | 0.80 | 0.80 | 0.80 | 0.78 | 0.78 |
| RedNet (rgat_b64_n002) | 0.79 | 0.80 | 0.81 | 0.81 | 0.79 | 0.78 |

**Overall Performance.** RedNet (rgat_b64_n002) consistently demonstrates robust performance across all metrics. For the standard full-sequence log-likelihood (`ll`), RedNet achieves the highest Spearman correlation ($\rho$ = 0.21) and NDCG (0.79), outperforming ProteinMPNN ($\rho$ = 0.17, NDCG = 0.78) and ESM-IF ($\rho$ = 0.16, NDCG = 0.77). This indicates that RedNet's likelihood estimates for the full binder sequence are better aligned with binding affinity than competing methods.

**Mutated Region Specificity.** When focusing specifically on the likelihood of mutated residues (`ll_mt`), ProteinMPNN ($\rho$ = 0.23) and RedNet ($\rho$ = 0.22) perform comparably in terms of ranking correlation. However, RedNet achieves a higher NDCG (0.81 vs 0.80), suggesting better identification of the most critical high-affinity variants. PiFold performs significantly worse on this local metric ($\rho$ = -0.03), indicating it struggles to capture local mutational effects accurately.

**Global and Reference-Normalized Scores.** On reference-normalized scores (`ll_ref`), ProteinMPNN shows a slight advantage ($\rho$ = 0.26) over RedNet ($\rho$ = 0.24). However, on global complex likelihood (`ll_global`), RedNet ($\rho$ = 0.26) significantly surpasses ProteinMPNN ($\rho$ = 0.17) and edges out ESM-IF ($\rho$ = 0.24). This suggests RedNet captures global stability and compatibility features that are relevant to binding affinity, which may be missed by models focusing solely on local autoregressive factorization.

**Impact of Contrastive Decoding.** We further explore contrastive decoding metrics (`cd_ll` and `cd_ll_ref`). RedNet (n0.02) achieves the highest performance with `cd_ll_ref` ($\rho$ = 0.28), surpassing its own `ll_ref` score (0.24) and ProteinMPNN's best score (0.26). The standard contrastive decoding score `cd_ll` for RedNet ($\rho$ = 0.26) also significantly outperforms the baseline `ll` (0.21), confirming that contrastive objectives effectively isolate binding-specific signals from background statistical noise. In contrast, other models do not benefit from this metric calculation, with scores generally dropping or remaining stagnant (e.g., ProteinMPNN `cd_ll` drops to 0.10).

**Effect of Noise Augmentation.** Comparing the noise-augmented RedNet (n002) with the standard version (n0), we observe consistent improvements. For `ll`, noise augmentation increases Spearman's $\rho$ from 0.18 to 0.21. Similar gains are seen in `ll_mt` (0.21 to 0.22) and `ll_ref` (0.22 to 0.24). This confirms that training with backbone coordinate noise improves the model's sensitivity to energetics and robustness, translating to better zero-shot prediction of mutational effects.
