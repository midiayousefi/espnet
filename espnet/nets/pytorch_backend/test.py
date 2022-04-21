from customized_e2e_asr_transformer_xvector import E2E
import yaml
from argparse import Namespace



yaml_file = "/scratch2/mxy171630/espnet/egs/wsj/asr1/conf/tuning/train_pytorch_transformer.yaml"
with open(yaml_file) as f:
    config = yaml.safe_load(f)


new_config = dict()
for key, val in config.items():
    new_key = key.replace('-', '_')
    new_config[new_key] = val
config = new_config


args = Namespace(**config)
args.ctc_type = "builtin"


model = E2E(idim = 83,
            odim = 60,
            args = args)




