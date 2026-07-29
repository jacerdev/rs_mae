#!/usr/bin/env bash

# If not inside a tmux session, create a detached session and run this script inside it
if [ -z "$TMUX" ]; then
    SESSION_NAME="salloc_$(date +%Y%m%d_%H%M%S)"
    echo "📦 Starting detached tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" "bash '$0'"
    echo "💻 To attach: tmux attach -t $SESSION_NAME"
    exit 0
fi

# import send_telegram
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/functions.sh"

SESSION_NAME=$(tmux display-message -p '#S' 2>/dev/null)
SBATCH_SCRIPT="$SCRIPT_DIR/cluster_train.sh"

# Extract values from SBATCH comments
JOB_NAME=$(grep "#SBATCH --job-name=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
GRES=$(grep "#SBATCH --gres=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
MEM=$(grep "#SBATCH --mem=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
CPUS=$(grep "#SBATCH --cpus-per-gpu=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
PART=$(grep "#SBATCH --partition=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
TIME=$(grep "#SBATCH --time=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
MAIL_TYPE=$(grep "#SBATCH --mail-type=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)
MAIL_USER=$(grep "#SBATCH --mail-user=" "$SBATCH_SCRIPT" | awk -F'=' '{print $2}' | xargs)

# Allocate resources
SALLOC_CMD="salloc --job-name=$JOB_NAME --gres=$GRES --mem=$MEM --cpus-per-gpu=$CPUS --partition=$PART --time=$TIME --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p "$HOME/outputs/tmux/"
tmux pipe-pane -o "stdbuf -o0 cat >> $HOME/outputs/tmux/$SESSION_NAME.tmux.out"
echo "Parsed SLURM requirements from $SBATCH_SCRIPT..."
echo "⚙️ [$SESSION_NAME] executing: $SALLOC_CMD"

$SALLOC_CMD bash -c "
    source \"$SCRIPT_DIR/functions.sh\"
    send_telegram \"📍 $SESSION_NAME: Node allocated for $TIME starting at \$(date '+%Y-%m-%d %H:%M:%S')\"
    
    # Exec into an interactive bash shell
    exec bash -i
"

