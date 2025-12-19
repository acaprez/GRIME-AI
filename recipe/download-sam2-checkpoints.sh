#!/bin/bash

sam2init=`/opt/anaconda1anaconda2anaconda3/bin/python -c 'import sam2; print(sam2.__file__)'`

SAM2_HOME=$(dirname $sam2init)

CHKPOINT_HOME="$SAM2_HOME/checkpoints"

mkdir -p ${CHKPOINT_HOME}

if [ ! -d ]; then
    echo "Can't create ${CHKPOINT_HOME}!"
    echo "Please ensure '${CHKPOINT_HOME}' exists and is writeable"
    exit 1
fi

echo "Downloading SAM2 checkpoint files to '${CHKPOINT_HOME}'..."

cd ${CHKPOINT_HOME}

# The following is based https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Used under Apache 2.0 license

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints
CMD="curl --clobber -# -L -O"
echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
$CMD $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }

echo "Downloading sam2.1_hiera_small.pt checkpoint..."
$CMD $sam2p1_hiera_s_url || { echo "Failed to download checkpoint from $sam2p1_hiera_s_url"; exit 1; }

echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
$CMD $sam2p1_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2p1_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "Success! SAM2 checkpoint files are downloaded. You may now run GRIME-AI."

exit 0
