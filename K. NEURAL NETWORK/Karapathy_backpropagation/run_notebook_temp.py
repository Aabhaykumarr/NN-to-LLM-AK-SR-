import json, traceback, os
from pathlib import Path
path = Path(r'c:/Users/aabha/OneDrive/Desktop/project_micrograd/Karapathy_backpropagation/06_wavenet.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
ns = {}
os.chdir(path.parent)
for idx, cell in enumerate(nb['cells'], 1):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if not src.strip():
        continue
    print(f'===== Executing cell {idx} =====')
    try:
        exec(compile(src, f'{path.name}#cell{idx}', 'exec'), ns)
        print('OK')
    except Exception as e:
        print('ERROR:', repr(e))
        traceback.print_exc()
        break
