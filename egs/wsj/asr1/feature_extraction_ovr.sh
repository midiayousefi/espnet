#!/bin/bash

set -e
set -o pipefail
source path.sh
source cmd.sh

base_data_dir="/scratch2/mxy171630/espnet/egs/wsj/asr1/data"
orig_data_dirs="
test_mixed_datasets/SIR_0
test_mixed_datasets/SIR_5
test_mixed_datasets/SIR_10
test_mixed_datasets/SIR_15
test_mixed_datasets/SIR_20
test_mixed_datasets/SIR_25
"
dumpdir=dump
do_delta=false
train_set=train_si284
train_dev=test_dev93
recog_set="test_mixed_datasets/SIR_0 test_mixed_datasets/SIR_5 test_mixed_datasets/SIR_10 test_mixed_datasets/SIR_15 test_mixed_datasets/SIR_20 test_mixed_datasets/SIR_25"
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
nlsyms=data/lang_1char/non_lang_syms.txt
dict=data/lang_1char/${train_set}_units.txt

### Task dependent. You have to design training and dev sets by yourself.
### But you can utilize Kaldi recipes in most cases
echo "stage 1: Feature Generation"
fbankdir=fbank
# Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
for x in $orig_data_dirs; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
       ${base_data_dir}/${x} exp/make_fbank/${x} ${fbankdir}
    utils/fix_data_dir.sh ${base_data_dir}/${x}
done

# compute global CMVN

for rtask in ${recog_set}; do
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
        ${feat_recog_dir}
    data2json.sh --feat ${feat_recog_dir}/feats.scp \
        --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
done











