#!/usr/bin/env bash
# Auto-resume training from the most recent checkpoint.

CKPT_DIR=${1:-checkpoints}
LATEST_CKPT=$(ls -t ${CKPT_DIR}/*/epoch_*.pt 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
  echo "No checkpoint found in ${CKPT_DIR}. Starting fresh training..."
  python main.py "$@"
else
  echo "Resuming from checkpoint: $LATEST_CKPT"
  python main.py checkpoint_path="$LATEST_CKPT" "$@"
fi 