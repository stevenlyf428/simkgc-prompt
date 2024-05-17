#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

python3 -u _preprocess.py \
--task "${TASK}" \
--train-path "./data/${TASK}/train.txt" \
--valid-path "./data/${TASK}/valid.txt" \
--test-path "./data/${TASK}/test.txt"
# set -x
# set -e

# TASK="wiki5m_trans"
# if [[ $# -ge 1 ]]; then
#     TASK=$1
#     shift
# fi

# python3 -u preprocess.py \
# --task "${TASK}" \
# --train-path "./data/${TASK}/wikidata5m_transductive_train.txt" \
# --valid-path "./data/${TASK}/wikidata5m_transductive_valid.txt" \
# --test-path "./data/${TASK}/wikidata5m_transductive_test.txt"

# set -x
# set -e

# TASK="wiki5m_ind"
# if [[ $# -ge 1 ]]; then
#     TASK=$1
#     shift
# fi

# python3 -u preprocess.py \
# --task "${TASK}" \
# --train-path "./data/${TASK}/wikidata5m_inductive_train.txt" \
# --valid-path "./data/${TASK}/wikidata5m_inductive_valid.txt" \
# --test-path "./data/${TASK}/wikidata5m_inductive_test.txt"