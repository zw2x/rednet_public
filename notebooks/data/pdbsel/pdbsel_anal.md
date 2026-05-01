# PDB Selectivity Analysis

This document summarizes the selectivity analysis results from `pdbsel_summary.md`. The analysis evaluates how well different models can design binders that are selective for an "on-target" over a closely related "off-target".

The following table summarizes the selectivity based on energy differences, specifically the **Binder Score**.

Binder Score is calculated as the sum of the **binding free energy** (`dG_binding`) and the **folding free energy** of the designed binder chain (`dG_binder`). Equivalently, it represents the energy of the complex minus the energy of the receptor (`dG_complex - dG_receptor`). This metric captures the total energetic contribution of adding the binder to the system. For a binder to be selective, it must exhibit a lower (more favorable) total energy state when interacting with the "on-target" compared to the "off-target". Optimizing this score encourages both high affinity binding and constitutive stability of the binder in the bound conformation.

*   **Binder Score Diff**: Calculated as `Score_on - Score_off`. A **negative** value indicates the binder prefers the on-target.
*   **Success Rate (Diff < X)**: The percentage of cases where the preference for the on-target is substantial (energy gap < X).

| Model | Success Rate (Diff < -10) | Success Rate (Diff < -5) | Success Rate (Diff < 0) | Avg Binder Score Diff |
| :--- | :--- | :--- | :--- | :--- |
| **Native** | 25.93% | 33.33% | 50.00% | 2.43 |
| **ProteinMPNN (n002)** | 31.48% | 44.44% | 53.70% | 0.62 |
| **ProteinMPNN (n010)** | 31.48% | 38.89% | 55.56% | -0.72 |
| **ProteinMPNN (n020)** | 31.48% | 44.44% | 53.70% | -2.32 |
| **ProteinMPNN (n030)** | 27.78% | 46.30% | 61.11% | -7.82 |
| **PiFold** | 27.78% | 40.74% | 55.56% | -5.69 |
| **ESM-IF** | 35.19% | 42.59% | 53.70% | -1.00 |
| **RGAT (n002)** | 22.22% | 27.78% | 33.33% | 8.90 |
| **RGAT (CD n002_a1_b09)** | 33.33% | 51.85% | 64.81% | -3.68 |

### Analysis of Energy-based Metrics

1.  **Selectivity Success Rates**:
    *   **ESM-IF** achieves the highest strict selectivity success rate (`Diff < -10`) at **35.19%**, followed by **RGAT (CD n002_a1_b09)** at **33.33%**.
    *   **RGAT (CD n002_a1_b09)** performs best at moderate thresholds (`Diff < -5` and `Diff < 0`), achieving **51.85%** and **64.81%** respectively.
    *   **ProteinMPNN (n030)** shows good performance at the lenient threshold (`Diff < 0`) with **61.11%**.
    *   **RGAT (n002)** performs poorly across all thresholds, significantly worse than the CD variant.

2.  **Average Binder Score Difference**:
    *   **ProteinMPNN (n030)** has the most favorable (most negative) average score difference (**-7.82**), indicating a strong general preference for the on-target.
    *   **PiFold** also shows a strong preference (**-5.69**).
    *   **RGAT (CD n002_a1_b09)** has a solid negative difference (**-3.68**), contrasting sharply with the standard **RGAT (n002)** which has a positive difference (**8.90**), suggesting the standard model often prefers the off-target or fails to discriminate.
    *   **Native** binders have a positive average difference (**2.43**), implying they are not optimized for this specific energetic discrimination metric.

## Summary Table

The following table summarizes the performance on 54 selected pairs (confident targets). Success rates are shown as percentages.

| Model | Selectivity Success | On-Target Success | Off-Target Success |
| :--- | :--- | :--- | :--- |
| **Native** | 3.70% | 74.07% | 74.07% |
| **ESM-IF** | 7.41% | 77.78% | 72.22% |
| **PiFold** | 3.70% | 72.22% | 75.93% |
| **ProteinMPNN (n002)** | 7.41% | 77.78% | 75.93% |
| **ProteinMPNN (n010)** | 1.85% | 79.63% | 79.63% |
| **ProteinMPNN (n020)** | 5.56% | 79.63% | 75.93% |
| **ProteinMPNN (n030)** | 7.41% | 81.48% | 75.93% |
| **RGAT (n002)** | 5.56% | 74.07% | 72.22% |
| **RGAT (CD n002_a1_b09)** | 9.26% | 72.22% | 70.37% |
| **RGAT (n002_a2_b09)** | 7.41% | 72.22% | 74.07% |

**Metrics Definitions**:
*   **Selectivity Success**: Proportion of failures where `iPTM_on > 0.55` and `iPTM_off < 0.55` (strict selectivity).
*   **On-Target Success**: Proportion where `iPTM_on > 0.55`.
*   **Off-Target Success**: Proportion where `iPTM_off > 0.55` (lower is generally better if aiming for specificity, but high means it binds both).

## Concise Analysis

1.  **ProteinMPNN Performance**:
    *   **ProteinMPNN (n030)** is a standout performer, achieving the highest **On-Target Success (81.48%)**.
    *   Increasing sampling/noise generally improves on-target success (n002: 77.78% -> n030: 81.48%), though its effect on selectivity success oscillates.

2.  **RGAT & Contrastive Decoding**:
    *   **RGAT (CD n002_a1_b09)** achieves the highest **Selectivity Success (9.26%)** among all models, suggesting that the contrastive decoding strategy is effective at enforcing specificity.
    *   Standard RGAT (n002) has moderate performance (Sel Success 5.56%), showing that parameter tuning (e.g., `a1_b09`) is crucial for optimal selectivity.

3.  **Baselines**:
    *   **Native** sequences show moderate success rates (74.07%).
    *   **ESM-IF** performs relatively well with high Selectivity Success (7.41%).
    *   **PiFold** shows minimal selectivity (3.70%).

4.  **Overall**:
    *   Designing for selectivity remains challenging, with "Selectivity Success" rates generally low (<10%).
    *   **ProteinMPNN (n030)** is best for maximizing binding affinity (success).
    *   **RGAT (CD)** variants are most promising for maximizing specificity/selectivity.
