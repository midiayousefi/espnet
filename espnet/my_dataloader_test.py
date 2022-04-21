#!/usr/bin/env python3

import sys
import json
from argparse import Namespace

import torch

sys.path.append('/scratch2/mxy171630/espnet')
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr import CustomConverter


args = Namespace(
    valid_json = '/scratch2/mxy171630/espnet/egs/wsj/asr1/dump/test_dev93/deltafalse/data.json',
    num_encs = 1,
    batch_size = 8,
    maxlen_in = 600,
    maxlen_out = 60,
    minibatches = 0,
    batch_count = 'auto',
    batch_bins = 0,
    batch_frames_in = 0,
    batch_frames_out = 0,
    batch_frames_inout = 0,
    preprocess_conf = '/scratch2/mxy171630/espnet/egs/wsj/asr1/conf/no_preprocess.yaml',
    n_iter_processes = 1,
    ngpu=0,
    
)





# Setup a converter
if args.num_encs == 1:
    #converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)
    converter = CustomConverter(subsampling_factor=1, dtype=torch.float32)
else:
    pass
#    converter = CustomConverterMulEnc(
#        [i[0] for i in model.subsample_list], dtype=dtype
#    )

# read json data
with open(args.valid_json, "rb") as f:
    valid_json = json.load(f)["utts"]


#use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
use_sortagrad = False

# make minibatch list (variable length)
valid = make_batchset(
    valid_json,
    args.batch_size,
    args.maxlen_in,
    args.maxlen_out,
    args.minibatches,
    min_batch_size=args.ngpu if args.ngpu > 1 else 1,
    count=args.batch_count,
    batch_bins=args.batch_bins,
    batch_frames_in=args.batch_frames_in,
    batch_frames_out=args.batch_frames_out,
    batch_frames_inout=args.batch_frames_inout,
    iaxis=0,
    oaxis=0,
)

load_cv = LoadInputsAndTargets(
    mode="asr",
    load_output=True,
    preprocess_conf=args.preprocess_conf,
    preprocess_args={"train": False},  # Switch the mode of preprocessing
)

valid_iter = ChainerDataLoader(
    dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: x[0],
    num_workers=args.n_iter_processes,
)


for something in valid_iter:
    print(len(something))
    
