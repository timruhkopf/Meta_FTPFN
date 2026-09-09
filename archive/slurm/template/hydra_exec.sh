#!/bin/bash
#SBATCH --job-name=start_hydra
#SBATCH --output=start_hydra%j.out
#SBATCH --error=start_hydra%j.err

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

#SBATCH --mem=8GB

# Set up the environment variables
commit_hash=$(git log -1 --pretty=format:"%h")

REPONAME=ExperiencedFT-PFNs

# on kisski:
module load Miniforge3

if [[ $HOME == /mnt/home* ]]; then
    conda activate $HOME/.conda/envs/ft-pfn
    # if we are on kisski
    BIGWORK=~
else

  conda activate $BIGWORK/envs/ft-pfn-experimental
fi

export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH
export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_icml2024:$PYTHONPATH

#HYDRA_FULL_ERROR=1; python $REPONAME/main_ifbo.py benchmark=lcbench device=cuda +algorithm=ifbo ~fold
#python
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
#conda activate $BIGWORK/envs/eft-pfn2
#export PYTHONPATH=$BIGWORK/$REPONAME/src:$PYTHONPATH
#export PYTHONPATH=$BIGWORK/$REPONAME/ifBO_main:$PYTHONPATH


export CUBLAS_WORKSPACE_CONFIG=:4096:8


# Parsing the command line arguments --------------------
# Initialize an empty array to store Hydra overrides
HYDRA_OVERRIDES=()
MULTIRUN_FLAG=""
CFG_FLAG=""
RESOLVE_FLAG=""

# Parse all command-line arguments as Hydra overrides or check for --multirun flag
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--multirun" ]]; then
        MULTIRUN_FLAG="--multirun"
    elif [[ "$1" == "--cfg=all" ]]; then
        CFG_FLAG="--cfg=all"
    elif [[ "$1" == "--resolve" ]]; then
        RESOLVE_FLAG="--resolve"
    else
        HYDRA_OVERRIDES+=("$1")
    fi
    shift
done

# Construct the Hydra command
HYDRA_CMD="python main_ifbo.py"

# Add all Hydra overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    HYDRA_CMD+=" $override"
done

# Prepare final command with commit hash and multirun flag if specified
FINAL_CMD="$HYDRA_CMD commit_hash=$commit_hash slurm_id=$SLURM_JOB_ID"

if [[ -n "$MULTIRUN_FLAG" ]]; then
    FINAL_CMD+=" $MULTIRUN_FLAG"
fi

if [[ -n "$CFG_FLAG" ]]; then
    FINAL_CMD+=" $CFG_FLAG"
fi

if [[ -n "$RESOLVE_FLAG" ]]; then
    FINAL_CMD+=" $RESOLVE_FLAG"
fi

# Parsing the Directory from the overrides --------------------
# Defaults (optional)
EXPERIMENT_NAME=""
EXPERIMENT_GROUP=""

echo "Debugging output"
echo ${HYDRA_OVERRIDES[@]}

# Extract specific values from overrides
for override in "${HYDRA_OVERRIDES[@]}"; do
    if [[ "$override" =~ ^experiment_name= ]]; then
        EXPERIMENT_NAME="${override#experiment_name=}"
    elif [[ "$override" =~ ^experiment_group= ]]; then
        EXPERIMENT_GROUP="${override#experiment_group=}"
    fi
done


# LOGGING the execution: ----------------------------------
# Timestamp in readable format
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Escape any quotes in the final command
FINAL_CMD_ESCAPED=$(echo "$FINAL_CMD" | sed 's/"/""/g')

# Log file path
LOG_DIR="$BIGWORK/$REPONAME/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_log.csv"

# Ensure header is written only once
if [[ ! -f "$LOG_FILE" ]]; then
    echo "timestamp,commit_hash,slurm_job_id,experiment_group,experiment_name,sweep_dir,final_cmd" >> "$LOG_FILE"
fi

# Append CSV line
echo "\"$TIMESTAMP\",\"$commit_hash\",\"$SLURM_JOB_ID\",\"$EXPERIMENT_GROUP\",\"$EXPERIMENT_NAME\",\"$SWEEP_DIR_CLEAN\",\"$FINAL_CMD_ESCAPED\"" >> "$LOG_FILE"



# Execute the command and capture output -------------------
export HYDRA_FULL_ERROR=1

# Temporary file to capture the output
#HYDRA_LOG_OUTPUT_FILE=$(mktemp)

# Run the command, streaming live output and writing to file
eval "$FINAL_CMD" # 2>&1 | tee "$HYDRA_LOG_OUTPUT_FILE"