import json
import chardet




with open('ppl_dev.json') as f:
    ppls = json.load(f)



with open('example.json', 'r') as f:
    examples = json.load(f)

for i in range(len(ppls)):
    ppls[i]['example'] = examples[i]




with open('ppl_dev.json', 'w') as f:
    json.dump(ppls, f, indent=4,ensure_ascii=False)