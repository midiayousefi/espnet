import json

data = {"utt1": {'feat': '/aaa/bbb', 'wav': '/ccc/ddd', 'category': 'train'},
         "utt2": {'feat': '/aaa/bbb', 'wav': '/ccc/ddd', 'category': 'dev'},
         "utt3": {'feat': '/aaa/bbb', 'wav': '/ccc/ddd', 'category': 'dev'}}

category2data = {}  # Dict[str, dict]
for k, v in data.items():
    category2data.setdefault(v.get("category"), {})[k] = v

print(json.dumps(category2data, indent=4))
