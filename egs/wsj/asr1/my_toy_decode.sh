#!/bin/bash

set -eu

decode_config=./toy_decode.yaml
ngpu=0
backend="pytorch"
model="exp/train_si284_pytorch_train_no_preprocess/results/model.last10.avg.best"
rnnlm="exp/train_rnnlm_pytorch_lm/rnnlm.model.best"
nlsyms="./data/lang_1char/non_lang_syms.txt"
dict="./data/lang_1char/train_si284_units.txt"

outdir="toy_outputs"
mkdir -p $outdir

../../../espnet/bin/asr_recog.py \
    --config ${decode_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --recog-json ./toy_subset_test_eval92.json \
    --result-label ${outdir}/data.1.json \
    --model $model \
    --rnnlm $rnnlm

../../../../espnet/utils/score_sclite.sh --wer true --nlsyms ${nlsyms} ${outdir} ${dict}
