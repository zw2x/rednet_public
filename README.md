# RedNet: inference and evaluation codes for rednet

## cli
* make_data: test data 
 ** select heteromers for folding test 
 ** select crosstalk db

* redsn: inference
 ** decode

* fold: folding evaluation
 ** config: configurate directory
 ** run: run alphafold3 with template
 ** eval: run usalign and dockq on predicted cif and gt cif
 ** check: check evaluated results

* rosetta: energetics evaluation