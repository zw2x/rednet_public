EXPR_NAME=$1
RESULT_DIR=$HOME/experiments/sel_experiments
BEST_RESULT=${RESULT_DIR}/results/${EXPR_NAME}/best_summary.csv

for i in {2..3}; do
    RELAX_DIR=${RESULT_DIR}/results/${EXPR_NAME}/r${i}
    if [ -d "${RELAX_DIR}" ]; then
        printf "Relaxation directory %s already exists. Skipping relaxation for run %d.\n" "$RELAX_DIR" "$i"
    else
        printf "Running relaxation for run %d: %s\n" "$i" "$RELAX_DIR"
        LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python cli/fold_pipeline.py run_relax ${BEST_RESULT} ${RELAX_DIR} \
            --cfg_file configs/filters/rosetta.yaml
    fi
done