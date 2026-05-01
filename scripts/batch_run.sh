SCRIPT_SH=$1
EXPR_PREFIX=$2
CKPT_FILE=$3

ALPHAS=("0.1" "0.5" "1" "2")
BETAS=("0.3" "0.5" "0.7" "0.9")
NUM_DEVICES=4

declare -A PID_TO_DEVICE  # Map PID to device ID
declare -a DEVICE_PIDS    # Track PID running on each device

printf "Running experiments with different alpha and beta values:\n"

TEMPERATURE=0.001

function generate_cfg {
    local alpha=$1
    local beta=$2
    local cfg_file=$3

    cat > "$cfg_file" <<EOF
alpha: ${alpha}
beta: ${beta}
temperature: ${TEMPERATURE}
EOF
}

for ALPHA in "${ALPHAS[@]}"; do
    for BETA in "${BETAS[@]}"; do
        NAME="a${ALPHA//.}_b${BETA//.}_t${TEMPERATURE//.}"
        EXPR_NAME="${EXPR_PREFIX}_${NAME}"
        CFG_NAME="cd_${NAME}"
        CFG_FILE="configs/sampling/${CFG_NAME}.yaml"

        # Generate config file if it does not exist
        if [[ ! -f "$CFG_FILE" ]]; then
            generate_cfg "$ALPHA" "$BETA" "$CFG_FILE"
        fi

        # Wait for a free device
        while true; do
            for (( dev=0; dev<NUM_DEVICES; dev++ )); do
                pid=${DEVICE_PIDS[$dev]}
                if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                    DEVICE_ID=$dev
                    break 2
                fi
            done
            sleep 1
        done

        printf "Starting experiment: %s on device %d\n" "$EXPR_NAME" "$DEVICE_ID"
        CUDA_VISIBLE_DEVICES=${DEVICE_ID} bash ${SCRIPT_SH} ${EXPR_NAME} ${CKPT_FILE} ${CFG_FILE} &
        DEVICE_PIDS[$DEVICE_ID]=$!
    done
done

wait  # Wait for remaining jobs
printf "All experiments completed.\n"