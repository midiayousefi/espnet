import sys
import json
import os
base_dir = "/scratch2/mxy171630/espnet/egs/wsj/asr1"



dataset_info = {
    "test_dev93": {
         "input_json_file": "dump/test_dev93/deltafalse/orig_data.json",
         "output_json_file": "dump/test_dev93/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
    "test_eval92": {
         "input_json_file": "dump/test_eval92/deltafalse/orig_data.json",
         "output_json_file": "dump/test_eval92/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_eval92",
    },
    "train_si284": {
         "input_json_file": "dump/train_si284/deltafalse/orig_data.json",
         "output_json_file": "dump/train_si284/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/train_si284",
    },
    "test_mixed_datasets_SIR_0": {
         "input_json_file": "dump/test_mixed_datasets/SIR_0/deltafalse/orig_data.json",
         "output_json_file": "dump/test_mixed_datasets/SIR_0/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
    "test_mixed_datasets_SIR_5": {
         "input_json_file": "dump/test_mixed_datasets/SIR_5/deltafalse/orig_data.json",
         "output_json_file": "dump/test_mixed_datasets/SIR_5/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
    "test_mixed_datasets_SIR_10": {
         "input_json_file": "dump/test_mixed_datasets/SIR_10/deltafalse/orig_data.json",
         "output_json_file": "dump/test_mixed_datasets/SIR_10/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
    "test_mixed_datasets_SIR_15": {
         "input_json_file": "dump/test_mixed_datasets/SIR_15/deltafalse/orig_data.json",
         "output_json_file": "dump/test_mixed_datasets/SIR_15/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
    "test_mixed_datasets_SIR_20": {
         "input_json_file": "dump/test_mixed_datasets/SIR_20/deltafalse/orig_data.json",
         "output_json_file": "dump/test_mixed_datasets/SIR_20/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
    "test_mixed_datasets_SIR_25": {
         "input_json_file": "dump/test_mixed_datasets/SIR_25/deltafalse/orig_data.json",
         "output_json_file": "dump/test_mixed_datasets/SIR_25/deltafalse/data.json",
         "xvector_dir": "exp/spk_xvectors/test_dev93",
    },
}



dumpdir = "dump"
expdir = "exp"

for dataset_name, dataset_info in dataset_info.items():
    with open(dataset_info["input_json_file"]) as f:
        data_info = json.load(f)

        for utt_id, utt_info in data_info["utts"].items():
            speaker_id = utt_info['utt2spk']
            xvec_file_name = "xvector_" +  speaker_id + ".npy"
            xvec_file = os.path.join(base_dir, dataset_info["xvector_dir"], xvec_file_name)

            #utt_info["xvector"] = xvec_file
            xvec_dict = {"feat": xvec_file,
                         "name": "input2",
                         "filetype": "npy",
                         "shape": [512,]}
            utt_info["input"].append(xvec_dict)
                        
            
        with open(dataset_info["output_json_file"], "w") as f_out:
            json.dump(data_info, f_out, indent=4)
             

