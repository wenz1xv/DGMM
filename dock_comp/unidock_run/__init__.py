import numpy as np
from typing import List, Tuple, NewType, Union
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.Chem import Mol as RDMol
from pathlib import Path
import gzip
import json
import multiprocessing
import os
import tempfile
import warnings
from rdkit.Chem import SDWriter
from rdkit.Chem.rdDistGeom import EmbedMolecule, srETKDGv3
from unidock_tools.application.unidock_pipeline import UniDock

pwd = os.path.dirname(os.path.abspath(__file__))

def run_dock(
    grid: str,
    smiles_list: list[str],
    box_size: tuple[float, float, float],
    search_mode: str,
    num_workers: int = 4,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 1,
) -> list[tuple[None, float] | tuple[RDMol, float]]:
    num_mols = len(smiles_list)
    scores = [0.0] * num_mols

    # create pdbqt file
    out_dir = Path(pwd)
    protein_pdb_path = out_dir / "grids" / f"{grid}.pdb"
    protein_pdbqt_path = out_dir / "grids" / f"{grid}.pdbqt"
    if not protein_pdbqt_path.exists():
        print('pdbqt needed')
        raise FileNotFoundError(f"Please provide the {grid}.pdbqt file.")
    sdf_list = []

    # etkdg
    etkdg_dir = out_dir / "etkdg"
    etkdg_dir.mkdir(parents=True, exist_ok=True)
    args = [(smi, etkdg_dir / f"{i}.sdf") for i, smi in enumerate(smiles_list)]
    with multiprocessing.Pool(num_workers) as pool:
        sdf_list = pool.map(run_etkdg_func, args)
    sdf_list = [file for file in sdf_list if file is not None]

    # unidock
    if len(sdf_list) > 0:
        runner = UniDock(
            protein_pdb_path,
            sdf_list,
            center[0],
            center[1],
            center[2],
            box_size[0],
            box_size[1],
            box_size[2],
            out_dir / "workdir",
        )
        runner.docking(
            out_dir / "savedir",
            num_modes=1,
            search_mode=search_mode,
            seed=seed,
        )

    for i in range(num_mols):
        try:
            docked_file = out_dir / "savedir" / f"{i}.sdf"
            docked_rdmol = list(Chem.SDMolSupplier(str(docked_file)))[0]
            assert docked_rdmol is not None
            docking_score = float(docked_rdmol.GetProp("docking_score"))
        except Exception:
            docking_score = 0.0
        scores[i] = docking_score
    return scores

def run_etkdg_func(args: tuple[str, Path]) -> Path | None:
    # etkdg parameters
    param = srETKDGv3()
    param.randomSeed = 1
    param.timeout = 1  # prevent stucking

    smi, sdf_path = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol.GetNumAtoms() == 0 or mol is None:
            return None
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol, param)
        assert mol.GetNumConformers() > 0
        mol = Chem.RemoveHs(mol)
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return None
    else:
        return sdf_path

def uni_dock(smiles, grid, search_mode="fast", num_workers=4, box_size=(25, 25, 25), dmapfile=None):
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
        scores = run_dock(
            grid,
            dockmol,
            box_size = box_size,
            search_mode = search_mode, # fast, balance, detail
            num_workers = num_workers
        )
        dscore_map.update(dict((k,v) for k,v in zip(dockmol, scores)))
        with open(dmapfile, 'r') as f:
            _dmap = json.load(f)
        dscore_map.update(_dmap)
        with open(dmapfile, 'w') as f:
            json.dump(dscore_map, f, indent=2)
        return [dscore_map[smi] for smi in smiles]