EXPR_NAME=$1
CKPT_FILE=$2
RESULT_DIR=$HOME/experiments/sel_experiments
TPL_DIR=$HOME/db/pdb_sel_files
BEST_RESULT=${RESULT_DIR}/results/${EXPR_NAME}/best_summary.csv
RELAX_DIR=${RESULT_DIR}/results/${EXPR_NAME}/relaxed

printf "EXPR_NAME: %s\nCKPT_FILE: %s\nRESULT_DIR: %s\nTPL_DIR: %s\nBEST_RESULT: %s\nRELAX_DIR: %s\n" \
      "$EXPR_NAME" "$CKPT_FILE" "$RESULT_DIR" "$TPL_DIR" "$BEST_RESULT" "$RELAX_DIR"

CFG_FILE=$3
printf "CFG_FILE: %s\n" "$CFG_FILE"

if [ -f "${BEST_RESULT}" ]; then
    printf "Summary file %s already exists. Skipping experiment.\n" "$BEST_RESULT"
else
    printf "Running experiment: %s\n" "$EXPR_NAME"

    python cli/infer_pipeline.py run_sel --repr_file docs/pdb_sel.csv --config_file configs/test_exptl/pdb_sel.yaml \
        --ckpt_file ${CKPT_FILE} --output_dir ${RESULT_DIR}/models/${EXPR_NAME} --sampling_config_file ${CFG_FILE}

    python cli/fold_pipeline.py config_sel ${RESULT_DIR}/models/${EXPR_NAME} ${TPL_DIR}

    python cli/fold_pipeline.py run_fold ${RESULT_DIR}/models/${EXPR_NAME} configs/filters/default.yaml --recursive
fi


python cli/anal_pipeline.py pdb_sel ${RESULT_DIR}/models/${EXPR_NAME} ${RESULT_DIR}/results/${EXPR_NAME} 

LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python cli/fold_pipeline.py run_relax ${BEST_RESULT} ${RELAX_DIR} \
    --cfg_file configs/filters/rosetta.yaml
