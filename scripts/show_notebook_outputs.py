"""Print the text outputs of an executed notebook (for reviewing results quickly)."""
import sys, json
nb = json.load(open(sys.argv[1], encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    for o in c.get('outputs', []):
        if o['output_type'] == 'stream':
            print(f'[{i}]', ''.join(o['text'])[:3000])
        elif 'data' in o and 'text/plain' in o['data']:
            print(f'[{i}]', ''.join(o['data']['text/plain'])[:3000])
        elif o['output_type'] == 'error':
            print(f'[{i}] ERROR', o['ename'], o['evalue'])
