import os
import json
import pandas as pd

pwd = os.path.dirname(os.path.abspath(__file__))

def schrodinger_dock(smiles, grid, level='SP', ncpus=4, dmapfile=None):
    if 'SCHRODINGER' not in os.environ:
        raise Exception(f"environ SCHRODINGER not defined")
    
    if len([x for x in os.listdir(pwd+'/grids/') if x.find(grid)> -1]) ==0:
        raise Exception(f"grid {grid} not exist")
    
    if not dmapfile:
        dmapfile = f"{pwd}/schrodinger_{level}/dmap/{grid}.json"
        
    if not os.path.isfile(dmapfile):
        with open(dmapfile, 'w') as f:
            json.dump({}, f, indent=2)
    with open(dmapfile, 'r') as f:
        dscore_map = json.load(f)
    
    if type(smiles) == str:
        smiles = [smiles]
        
    dockmol = set(smiles)
    dockmol = [m for m in dockmol if m not in dscore_map]

    if len(dockmol) == 0:
        return [dscore_map[smi] for smi in smiles]
    else:
        score = [0]*len(dockmol)
        with open(f'{pwd}/schrodinger_{level}/ligand.smi', 'w') as file:
            for idx,smi in enumerate(dockmol):
                file.write(f'{smi} {idx}\n')
        with open(f'{pwd}/schrodinger_{level}/grid.info', 'w') as file:
            file.write(grid)
        sout = os.system(f'{pwd}/schrodinger_{level}/dock.sh')
        if sout ==0:
            df = pd.read_csv(f'{pwd}/schrodinger_{level}/output.csv').loc[1:]
            for i, r in df.iterrows():
                score[int(r.NAME)] = r['r_i_docking_score']
        else:
            raise Exception('Schrodinger Docking Wrong')
        dscore_map.update(dict((k,v) for k,v in zip(dockmol, score)))
        with open(dmapfile, 'r') as f:
            _dmap = json.load(f)
        dscore_map.update(_dmap)
        with open(dmapfile, 'w') as f:
            json.dump(dscore_map, f, indent=2)
        return [dscore_map[smi] for smi in smiles]