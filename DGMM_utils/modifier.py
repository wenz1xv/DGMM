import json
import numpy as np
import pandas as pd
from functools import partial
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
RDLogger.DisableLog('rdApp.*')

pwd = os.path.dirname(os.path.abspath(__file__))

with open(f'{pwd}/modifier.json', 'r') as file:
    modifier_data = json.load(file)
    
class Modifier:
    def __init__(self):
        self.reactions = modifier_data['reactions_double']
        self.frags = set(modifier_data['frags'])
        for y in self.reactions:
            y['rdgroups'] = [Chem.MolFromSmarts(x) for x in y['group_smarts']]
            y['rdreactions'] = AllChem.ReactionFromSmarts(y['reaction_string'])
        self.reactions_single = modifier_data['reactions_single']
        for y in self.reactions_single:
            y['rdgroups'] = [Chem.MolFromSmarts(x) for x in y['group_smarts']]
            y['rdreactions'] = AllChem.ReactionFromSmarts(y['reaction_string'])
        self.df = None
        self.coreDict = {}
            
    def getRfp(self, smi, reactions=None):
        if not reactions:
            reactions = self.reactions
        mol = Chem.MolFromSmiles(smi)
        hasM = partial(self.hasSub, mol)
        rfp = np.array(list(map(hasM, reactions)))
        return rfp
        
    def hasSub(self, mol, reaction):
        cnt = 0
        for idx, g in enumerate(reaction['rdgroups']):
            if mol.HasSubstructMatch(g):
                return idx+1
        return 0
    
    def get_core(self, smi):
        if smi in self.coreDict:
            return self.coreDict[smi]
        mol = Chem.MolFromSmiles(smi)
        core_mol = MurckoScaffold.GetScaffoldForMol(mol)
        core_fp = Chem.MACCSkeys.GenMACCSKeys(core_mol)
        core = smi
        sim_max = 0
        flag = True
        while flag:
            flag = False
            frags = self.deComp([core])
            frags = list(frags)
            for frag in frags:
                fp = Chem.MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(frag))
                sim = DataStructs.TanimotoSimilarity(core_fp, fp)
                if sim > sim_max:
                    sim_max = sim
                    flag = True
                    core = frag
        self.coreDict[smi] = core
        print(f'max similarity : {sim_max}')
        return core
        
    def validMol(self, mol):
        try:
            inchi = Chem.MolToInchi(mol)
            mi = Chem.MolFromInchi(inchi, sanitize=True)
            smi = Chem.MolToSmiles(mi)
            return smi
        except Exception as e:
#             print('error')
            return ''
        print('None')
        return ''

    def getProd(self, rxn, reactants):
        res = []
        reactants_mol = [Chem.MolFromSmiles(x) for x in reactants]
        rxn_res = rxn.RunReactants(reactants_mol)
        for r in rxn_res:
            res += [ self.validMol(c) for c in r]
        if len(reactants) > 1:
            rxn_res = rxn.RunReactants(reactants_mol[::-1])
            for r in rxn_res:
                res += [ self.validMol(c) for c in r]
        res = [x for x in set(res) if len(x) > 0]
        return res
    
    def deComp(self, smis):
        deReactions = [AllChem.ReactionFromSmarts('>>'.join(x['reaction_string'].split('>>')[::-1])) for x in self.reactions_single ]
        dePatts = [Chem.MolFromSmarts(x['reaction_string'].split('>>')[-1]) for x in self.reactions_single ]
        res = []
        if type(smis) == str:
            smis = [smis]
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            for idx, patt in enumerate(dePatts):
                if mol.HasSubstructMatch(patt):
                    rxn = deReactions[idx]
                    rxn_res = rxn.RunReactants([mol])
                    for r in rxn_res:
                        res += [ self.validMol(c) for c in r]
        res = set([x for x in set(res) if len(x) > 1])
        return res
        
    def Renu(self, ligand, n=32, frags=[]):
        ligand = self.get_core(ligand)
        self.ligand = ligand
        self.ligand_rfp = self.getRfp(ligand)
        self.ligand_srfp = self.getRfp(ligand, reactions=self.reactions_single) 
        products = []
        reactions = []
        reactants = []
        rids = []
        frags = set(frags) | self.frags
        for frag in frags:
            reaction_id = np.where(self.ligand_rfp + self.getRfp(frag) ==3)[0]
            for rid in reaction_id:
                res = self.getProd(self.reactions[rid]['rdreactions'], [ligand, frag])
                for p in res:
                    rids.append(rid)
                    products.append(p)
                    reactions.append(self.reactions[rid]["reaction_name"])
                    reactants.append(frag)
        reaction_id = np.where(self.ligand_srfp>0)[0]
        for rid in reaction_id:
            res = self.getProd(self.reactions_single[rid]['rdreactions'], [ligand])
            for p in res:
                rids.append(rid)
                products.append(p)
                reactions.append(self.reactions_single[rid]["reaction_name"])
                reactants.append('')
        self.df = pd.DataFrame({
            'reactant': reactants,
            'reaction': reactions,
            'product': products,
            'rid': rids,
        })
        if len(self.df) > n:
            return set(self.df.sample(n=n)['product'].tolist())
        else:
            return set(products)