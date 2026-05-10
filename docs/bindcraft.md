BindCraft Filter Thresholds Summary:

| Metric                 | Threshold  | Direction |
|------------------------|------------|-----------|
| pLDDT                  | > 0.8      | higher ✓  |
| pTM                    | > 0.55     | higher ✓  |
| i_pTM                  | > 0.5      | higher ✓  |
| Binder_pLDDT           | > 0.8      | higher ✓  |
| i_pAE                  | < 0.35     | lower ✓   | # this is ignored. definition of pAE in ColabDesign seems different

| Binder_Energy_Score    | < 0        | lower ✓   |
| dG                     | < 0        | lower ✓   |
| dSASA                  | > 1        | higher ✓  |
| ShapeComplementarity   | > 0.55-0.6 | higher ✓  |
| n_InterfaceResidues    | > 7        | higher ✓  |
| n_InterfaceHbonds      | ≥ 3        | higher ✓  |
| n_InterfaceUnsatHbonds | < 4        | lower ✓   |
| Surface_Hydrophobicity | < 0.35     | lower ✓   |

| Binder_Loop%           | < 90%      | lower ✓   |
| Binder_RMSD            | < 3.5 Å    | lower ✓   |
| Hotspot_RMSD           | < 6 Å      | lower ✓   |