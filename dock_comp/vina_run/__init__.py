import os
from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
import re
import json

pwd = os.path.dirname(os.path.abspath(__file__))

def make_grid(grid_pdbqt):
#         grid_pdbqt = '7jnt.pdbqt'
    v = Vina(sf_name='vina', verbosity=0, cpu=8)
    v.set_receptor(grid_pdbqt)
    v.compute_vina_maps(center=[0, 0, 0], box_size=[25, 25, 25], force_even_voxels=True)
    v.write_maps(map_prefix_filename=f'{pwd}/'+grid_pdbqt.split('/')[-1].split('.')[0])

def smi2pdbqt(smiles):
    d = ['', '+','1', '2', '3', '4', '5', '6', '7', '8', '9', '>', '<', '[',']', '(', ')', '#', '-', '=', 'H', 'C', 'c', 'S', 's', 'N', 'n', 'O', 'o', 'F', 'Cl', 'Br', 'I', ':', '@','\\', '//', '/']
    reg = re.compile('(Br|Cl|.)')
    atoms = [atom for atom in reg.split(smiles)[1::2] if atom not in d]
    if atoms:
        print(atoms, " is invalid atom.")
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
        if not mol:
            return None
        m3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m3d, randomSeed=312)
        AllChem.MMFFOptimizeMolecule(m3d)
        preparator = MoleculePreparation()
        preparator.prepare(m3d)
        pdbqt_string = preparator.write_pdbqt_string()
        return pdbqt_string
    except:
        return None

def make_grid(grid_pdbqt):
    pdbqt_path = pwd +'/' + grid_pdbqt
    grid_path = pwd +'/vina_grid/'
    if not os.path.exists(grid_path ):
        os.makedirs(grid_path)
    v = Vina(sf_name='vina', verbosity=0, cpu=8)
    v.set_receptor(pdbqt_path)
    v.compute_vina_maps(center=[0, 0, 0], box_size=[25, 25, 25], force_even_voxels=True)
    v.write_maps(map_prefix_filename=grid_path+ grid_pdbqt.split('.')[0])

def vina_dock(smiles, grid, ncpus=4, dmapfile=None):
    if not dmapfile:
        dmapfile = f"{pwd}/dmap/{grid}.json"
        
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
        scores = []
        for smi in dockmol:
            v = Vina(sf_name='vina', verbosity=0, cpu=ncpus)
            if len([x for x in os.listdir(pwd+'/vina_grid/') if x.find(grid)> -1]) ==0:
                raise Exception(f"grid {grid} not exist")
            v.load_maps(pwd+'/vina_grid/'+grid)
            ligand = smi2pdbqt(smi)
            v.set_ligand_from_string(ligand)
            v.dock(exhaustiveness=8, n_poses=10)
            scores.append(v.score()[0])
        dscore_map.update(dict((k,v) for k,v in zip(dockmol, scores)))
        with open(dmapfile, 'r') as f:
            _dmap = json.load(f)
        dscore_map.update(_dmap)
        with open(dmapfile, 'w') as f:
            json.dump(dscore_map, f, indent=2)
        return [dscore_map[smi] for smi in smiles]
