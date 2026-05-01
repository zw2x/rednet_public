# H-dimer Energy Analysis Summary

This document summarizes the metrics from `hdimer_energy.md` comparing the native structure with various design models.

## Metrics Table

| Model | Binder Score | Surf Hydro | Int SC | Int Packstat | Int dG | Int dSASA | Int dG/SASA | Int Frac | Int Hydro | Int Nres | Int HBonds | Int HBond % | Int dUnsat HB | Int dUnsat HB % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Native** | -172.42 | 0.43 | 0.65 | 0.53 | -53.35 | 1918.33 | -2.68 | 105.04 | 47.50 | 20.10 | 6.97 | 46.15 | 2.98 | 17.90 |
| **RGAT (n002)** | -179.88 | 0.45 | 0.66 | 0.54 | -52.49 | 1868.29 | -2.77 | 111.30 | 48.93 | 19.42 | 6.21 | 44.80 | 2.44 | 15.40 |
| **RGAT (n002_a1_b09)** | -182.26 | 0.44 | 0.66 | 0.54 | -54.47 | 1894.69 | -2.78 | 112.55 | 49.23 | 19.70 | 7.01 | 48.23 | 2.63 | 14.77 |
| **PiFold** | -179.55 | 0.47 | 0.67 | 0.53 | -55.19 | 1870.66 | -2.91 | 193.63 | 52.55 | 18.98 | 6.07 | 43.49 | 2.91 | 19.32 |
| **ESM-IF** | -181.27 | 0.46 | 0.64 | 0.53 | -54.97 | 1947.75 | -2.68 | 115.88 | 48.49 | 19.92 | 6.51 | 43.19 | 2.78 | 17.26 |
| **ProteinMPNN (n002)** | -184.89 | 0.43 | 0.66 | 0.55 | -46.98 | 1682.12 | -2.73 | 107.30 | 51.19 | 17.34 | 5.44 | 40.30 | 2.52 | 17.62 |
| **SolMPNN** | -139.99 | 0.46 | 0.66 | 0.53 | -49.87 | 1803.24 | -2.66 | 124.57 | 50.40 | 17.11 | 6.15 | 44.37 | 2.36 | 15.50 |
| **Fused** | -188.04 | 0.44 | 0.67 | 0.55 | -56.66 | 1965.96 | -2.88 | 120.65 | 50.09 | 20.19 | 7.31 | 48.99 | 2.59 | 14.27 |

**Note**: Values are means from the `build_min_df` output (selecting best from replicates). `dG` refers to interface dG. `dSASA` is interface dSASA.

## Findings

1.  **Metric Comparison**: 
    *   **Native Performance**: The Native structure performs very well in this analysis, with a Binder Score of -172.42 and a markedly negative Interface dG of -53.35.
    *   **Models vs Native**: Several models outperform the Native structure. The **Fused** model achieves the best Binder Score (-188.04) among the remaining models and the best Interface dG (-56.66). **ProteinMPNN (n002)** also shows a very strong Binder Score (-184.89).

2.  **Top Performers**:
    *   **Binder Score**: The **Fused** model is the top performer (-188.04), followed closely by **ProteinMPNN (n002)** (-184.89) and **RGAT (n002_a1_b09)** (-182.26).
    *   **Interface dG**: The **Fused** model leads (-56.66), followed by **PiFold** (-55.19) and **ESM-IF** (-54.97).

3.  **ProteinMPNN Performance**:
    *   **ProteinMPNN (n002)** performs strongly with a Binder Score of -184.89, though its interface dG (-46.98) is slightly less favorable (less negative) compared to Native (-53.35) and other top models.

4.  **Interface Properties**:
    *   **Surface Area (dSASA)**: The **Fused** model generates the largest interface (1965.96), followed by **ESM-IF** (1947.75) and Native (1918.33). **ProteinMPNN (n002)** produces the smallest interfaces (1682.12).
    *   **Hydrogen Bonds**: The **Fused** model achieves the highest number of interface hydrogen bonds (7.31), surpassing Native (6.97). **RGAT (n002_a1_b09)** creates slightly more hydrogen bonds (7.01) than Native.
    *   **Shape Complementarity**: Most models achieve high SC (~0.66-0.67), with **Fused**, **PiFold**, and **RGAT** variants showing strong complementarity.

5.  **RGAT Performance**:
    *   **RGAT (n002_a1_b09)** is a balanced performer with excellent Binder Score (-182.26) and interface dG (-54.47).
    *   **RGAT (n002)** also performs well (-179.88), while the `b03` variant lags behind significantly (-133.24).
