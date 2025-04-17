import re
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import functools
from functools import partial
from multiprocessing import Pool
from tensorflow.keras.models import load_model
# from keras.models import load_model
import selfies as sf
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, QED, MACCSkeys, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

DEBUG=False

from dock_comp import schrodinger_dock, vina_dock, uni_dock
from .model_layers import Sampling
from .score_utils import acceptance_score, SAscore
from .modifier import Modifier
from .dataprocess import alphabet, symbol_to_idx

def express(acceptance, pro):
    smiles = ''
    for p in pro:
        idx = np.random.choice(np.arange(len(alphabet)), p=p)
        token = alphabet[idx]
        if alphabet[idx].find("[nop]") > -1:
            break
        smiles+=token
    smiles = sf.decoder(smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        assert len(smiles) > 1
    except:
        return ''
    score = acceptance(smiles)
    if score > 0 and np.random.random() < score:
        return smiles
    return ''
        

class Filter:
    """
    Filter:
        filter molecule based on the constraint
        
    Method:
        run: run filter
    """
    def __init__(self, method,dock_grids,docklevel, simthd, qedthd, sathd,sim_increase=0):
        self.dscore_map = {}
        self.method = method
        if self.method not in ['schrodinger', 'vina', 'unidock']:
            raise Exception('Docking method should be vina or shrodinger!')
        self.sim = simthd
        self.qedthd = qedthd
        self.sathd = sathd
        self.sim_increase = sim_increase
        self.docklevel = docklevel
        
        if type(dock_grids) == str:
            dock_grids = [dock_grids]
        self.dock_grids = dock_grids
        self.config = {
            'sim': simthd,
            'qedthd': qedthd,
            'sathd': sathd,
            'dock_grids': dock_grids,
            'sim_increase': sim_increase
        }
    
    def load(self, config):
        self.config = config
        self.sim = config['sim']
        self.sim_increase = config['sim_increase']
        
    def extract_core(self, smis):
        core = Chem.MolFromSmiles(smis[0])
        core = MurckoScaffold.GetScaffoldForMol(core)
        if len(smis) > 1:
            mols = [Chem.MolFromSmiles(x) for x in smis]
            mcs = rdFMCS.FindMCS(mols,
                                 bondCompare=rdFMCS.BondCompare.CompareAny,
                                 atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                                 ringMatchesRingOnly=True
                                )
            if not core.HasSubstructMatch(mcs.queryMol):
                subcore = Chem.MolToSmarts(core)
                subcore = re.sub(r':\[#..?\]', ":*", subcore)
                subcore = Chem.MolFromSmarts(subcore)
                return subcore
            subcore = Chem.MolToSmarts(Chem.ReplaceSidechains(core, mcs.queryMol))
            subcore = re.sub(r'\[\d#0\]','*', subcore)
            subcore = re.sub(r'[-|=|#|:]\*', "~*", subcore)
            subcore = re.sub(r'\*[-|=|#|:]', "*~", subcore)
            subcore = re.sub(r':\[#..?\]', ":*", subcore)
            subcore = Chem.MolFromSmarts(subcore)
            return subcore
        else:
            subcore = Chem.MolToSmarts(core)
            subcore = re.sub(r':\[#..?\]', ":*", subcore)
            subcore = Chem.MolFromSmarts(subcore)
            return subcore
        
    def get_d_scores(self, smiles):
        scores = []
        for smi in smiles:
            if smi in self.dscore_map:
                scores.append(self.dscore_map[smi])
            else:
                scores.append(0)
        return scores
    
    def get_scffp(self, mol):
        try:
            if type(mol) == str:
                mol = Chem.MolFromSmiles(mol)
            core = MurckoScaffold.GetScaffoldForMol(mol)
            fp = Chem.MACCSkeys.GenMACCSKeys(core)
            return fp
        except:
            print('extract scf failed')
            return ''
    
    def get_p_scores(self, smiles):
        p_scores = np.ones(len(smiles))
        for idx in range(len(smiles)):
            mol = Chem.MolFromSmiles(smiles[idx])
            # QED penalty
            qed_score = QED.default(mol)
            if qed_score < self.qedthd:
                p_scores[idx] *= qed_score*3
            # SA penalty
            sa_score = SAscore(mol)
            if sa_score > self.sathd:
                p_scores[idx] *= 1./sa_score
            # scf similarity penalty
            fp = Chem.RDKFingerprint(mol)
            sim_score = DataStructs.TanimotoSimilarity(fp, self.afp)
            if sim_score == 1:
                p_scores[idx] = 1.
            elif len(self.scf.GetAtoms()) and mol.HasSubstructMatch(self.scf):
                pass
            elif sim_score < self.sim:
                p_scores[idx] *= sim_score*2
        return p_scores
        
    def run(self, pop, headers):
        ancestor = headers[0]
        def compare(lmol, rmol):
            if lmol == ancestor:
                return 1
            elif rmol == ancestor:
                return -1
            elif self.dscore_map[rmol] < self.dscore_map[lmol]:
                return 1
            elif self.dscore_map[lmol] < self.dscore_map[rmol] :
                return -1
            else:
                return 0
        self.afp = Chem.RDKFingerprint(Chem.MolFromSmiles(ancestor))
        self.scf = self.extract_core(headers)
        if type(pop) != list:
            pop = list(pop)
            
        p_scores = self.get_p_scores(pop)
        chooser = np.random.random(len(p_scores)) < p_scores
        pop = [m for f, m in zip(chooser, pop) if f]
        
        d_scores = np.ones(len(pop))
        for dock_grid in self.dock_grids:
            _scores = self.batchdock(pop, dock_grid)
            d_scores = np.min(np.array([d_scores, _scores]), axis=0)
        self.dscore_map.update(dict(zip(pop, d_scores)))
        pop.sort(key = functools.cmp_to_key(compare))
        
        # duplication penalty
        p_scores = np.ones(len(pop))
        fps = []
        for idx in range(len(pop)):
            dsim_score = 1
            fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles(pop[idx]))
            fps.append(fp1)
            if DataStructs.TanimotoSimilarity(fp1, self.afp)==1:
                continue
            for i in range(idx):
                _dsim_score = DataStructs.TanimotoSimilarity(fp1, fps[i])
                if _dsim_score > 0.9:
                    dsim_score = min(dsim_score, (1.-_dsim_score)*2.)
            p_scores[idx] *= dsim_score
        chooser = np.random.random(len(p_scores)) < p_scores
        pop = [m for f, m in zip(chooser, pop) if f]
        
        self.sim = min(0.8, self.sim+self.sim_increase)
        return pop
            
    def batchdock(self, smiles, dock_grid, thread=0):
        if self.method == 'vina':
            scores = vina_dock(smiles, dock_grid)
        elif self.method == 'schrodinger':
            if self.docklevel==0:
                scores = schrodinger_dock(smiles, dock_grid, level='HTVS')
            elif self.docklevel==1:
                scores = schrodinger_dock(smiles, dock_grid, level='SP')
            elif self.docklevel==2:
                scores = schrodinger_dock(smiles, dock_grid, level='XP')
            else:
                scores = schrodinger_dock(smiles, dock_grid, level='HTVS')
        elif self.method == 'unidock':
            if self.docklevel==0:
                scores = uni_dock(smiles, dock_grid,search_mode='fast')
            elif self.docklevel==1:
                scores = uni_dock(smiles, dock_grid,search_mode='balance')
            elif self.docklevel==2:
                scores = uni_dock(smiles, dock_grid,search_mode='detail')
            else:
                scores = uni_dock(smiles, dock_grid,search_mode='fast')
        return scores

class EDmodel:
    """
    EDmodel:
        the encoder/decoder model
    Method:
        encode: encode molecule to latent
        decode: decode latent to molecule n times
    """
    def __init__(self, model_path, childsize, simthd, qedthd, sathd, ex_constraint, thread, maxtrys=80, MAXLEN=80):
        self.encoder = None
        self.decoder = None
        if self.load_model(model_path):
            print('Successfully loading model.')
        else:
            print('Loading model Wrong!')
        self.maxtrys = maxtrys
        self.MAXLEN = MAXLEN
        self.childsize = childsize
        self.config = {
            'model': model_path,
            'maxtrys': maxtrys,
            'childsize': childsize,
            'MAXLEN': MAXLEN,
        }
        self.simthd = simthd
        self.qedthd = qedthd
        self.sathd = sathd
        self.thread = thread
        self.ex_constraint = ex_constraint
        
    def load_model(self, model_path):
        try:
            self.encoder = load_model(f'{model_path}/DGMM_sfi_encoder.keras', custom_objects={'Sampling': Sampling})
            self.decoder = load_model(f'{model_path}/DGMM_sfi_decoder.keras')
            return True
        except Exception as e:
            print(e, ' load model error')
            return False
        
    def load(self, config):
        self.config = config
        if self.load_model(config['model'],):
            print('Successfully loading model.')
        else:
            print('Loading model Wrong!')
        self.maxtrys = config['maxtrys']
        self.childsize = config['childsize']
        self.sim = config['sim']
        self.MAXLEN = config['MAXLEN']
    
    def regen_smi(self, smi, flag=0):
        m = Chem.MolFromSmiles(smi)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        nsmi = Chem.MolToSmiles(nm, canonical=False)
        try:
            inp = self.smiles2inp(nsmi)
            return nsmi
        except:
            if flag==0:
                return ''
            else:
                return self.regen_smi(smi, flag-1)
        
    def encodable(self, smiles, retry=3):
        pop = []
        if type(smiles) == str:
            smiles = [smiles]
        smiles = [smi for smi in smiles if len(smi) > 0]
        for smi in smiles:
            try:
                inp = self.smiles2inp(smi)
                assert len(smi) > 3
                pop.append(smi)
            except:
                nsmi = self.regen_smi(smi, retry)
        return pop
    
    def merge_fps(self, smis):
        mols = [Chem.MolFromSmiles(s) for s in smis]
        fps = [Chem.RDKFingerprint(m) for m in mols]
        fps_list = np.array([fp.ToList() for fp in fps])
        fps_flag = np.where(np.any(fps_list, axis=0))[0]
        fp_merge = DataStructs.ExplicitBitVect(len(fps[0]))
        fp_merge.SetBitsFromList(fps_flag.tolist())
        return fp_merge
        
    def extract_core(self, smis):
        core = Chem.MolFromSmiles(smis[0])
        core = MurckoScaffold.GetScaffoldForMol(core)
        if len(smis) > 1:
            mols = [Chem.MolFromSmiles(x) for x in smis]
            mcs = rdFMCS.FindMCS(mols,
                                 bondCompare=rdFMCS.BondCompare.CompareAny,
                                 atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                                 ringMatchesRingOnly=True
                                )
            if not core.HasSubstructMatch(mcs.queryMol):
                subcore = Chem.MolToSmarts(core)
                subcore = re.sub(r':\[#..?\]', ":*", subcore)
                subcore = Chem.MolFromSmarts(subcore)
                return subcore
            subcore = Chem.MolToSmarts(Chem.ReplaceSidechains(core, mcs.queryMol))
            subcore = re.sub(r'\[\d#0\]','*', subcore)
            subcore = re.sub(r'[-|=|#|:]\*', "~*", subcore)
            subcore = re.sub(r'\*[-|=|#|:]', "*~", subcore)
            subcore = re.sub(r':\[#..?\]', ":*", subcore)
            subcore = Chem.MolFromSmarts(subcore)
            return subcore
        else:
            subcore = Chem.MolToSmarts(core)
            subcore = re.sub(r':\[#..?\]', ":*", subcore)
            subcore = Chem.MolFromSmarts(subcore)
            return subcore
    
    def smiles2inp(self, smiles, mode='sfi'):
        if type(smiles) == str:
            smiles = [smiles]
        if mode == 'sfi':
            tokens = []
            for smi in smiles:
                label = sf.selfies_to_encoding(
                    selfies=sf.encoder(smi),
                    vocab_stoi=symbol_to_idx,
                    pad_to_len=self.MAXLEN,
                    enc_type="label"
                )
                tokens.append(np.array(label).astype('float'))
            return np.array(tokens)
        elif mode == 'smi':
            tokens = []
            for smi in smiles:
                reg = re.compile('(Br|Cl|.)')
                mols = reg.split(smi)[1::2]
                token = np.zeros((MAXLEN), dtype=int)
                for idx,atom in enumerate(mols):
                    if atom in symbol_to_idx:
                        token[idx] = symbol_to_idx[atom]
                tokens.append(token)
            return np.array(tokens)
        else:
            print('unknow mode')
            return None

    def get_code_indices(self, latent):
        # Calculate L2-normalized distance between the inputs and the codes.
        flattened_inputs = tf.reshape(latent, [-1, tf.shape(self.codebook)[0]])
        similarity = tf.matmul(flattened_inputs, self.codebook)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum( self.codebook ** 2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.reshape(tf.argmin(distances, axis=1), [tf.shape(latent)[0],-1])
        return encoding_indices
        
    def encode(self, smiles):
        inp = self.smiles2inp(smiles)
        z_mean, z_logvar, z = np.array(self.encoder.predict(np.array(inp), verbose=0))
        return (z_mean, z_logvar, z)
    
    def multisample(self, genes, headers, pop=[]):
        self.scf = self.extract_core(headers)
        self.afp = self.merge_fps(headers)
        self.visited = set(pop)
        children = set()
        pscore = partial(acceptance_score, self.ex_constraint, self.simthd, self.qedthd, self.sathd, self.scf, self.afp)
        batch = self.gene2pro(genes)
        for i in range(self.maxtrys):
            child = self.decode(pscore, batch)
            for c in child:
                children.add(c)
                if len(children) > self.childsize:
                    return children
                self.visited.add(c)
            if len(children) > self.childsize:
                break
        return children
        
    def gene2pro(self, genes):
        pro_pred = np.array(self.decoder.predict(genes, verbose=0)[0])
        return pro_pred
    
    def cleanUp(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            smi1 = Chem.MolToSmiles(mol, canonical=True)
            assert len(smi1) > 1
            return smi1
        except:
            return ''
    
    def decode(self, pscore, batch, pop=None):
        batchs = np.repeat(batch, self.maxtrys, axis=0)
        worker = Pool(self.thread)
        _express = partial(express, pscore)
        smiles = worker.map(_express, batchs)
        worker.close()
        worker.join()
        smiles = [x for x in set(smiles) if len(x) > 0]
        return smiles
    
class DGMMmodel:
    def __init__(self, model_path, popsize=100, genebatch=8, ex_constraint=None,
                 dock_grids = 'rock2_6ed6', method='schrodinger',thread=4, 
                 childsize=4, vfactor=0.3, enhance=0, simthd=0.47, qedthd=0.3, sathd=3,
                 path='result', compete=False, loglevel=0, docklevel=0):
        self.MAXLEN = 80
        
        self.EDmodel = EDmodel(model_path, childsize, simthd, qedthd, sathd, ex_constraint, thread)
        self.Filter = Filter(method, dock_grids, docklevel, simthd, qedthd, sathd)
        self.modifier = Modifier()
        self.popsize = popsize
        self.vfactor = vfactor
        self.childsize = childsize
        self.genebatch = genebatch
        self.compete = compete
        self.enhance = enhance
        self.egroup = []
        self.joined = []
        self.loglevel = loglevel
        self.target = dock_grids

        self.log = []
        self.stat = {
            'init': False,
            'epoch': 0,
            "header":'',
            "best":'',
            "popsize": 1,
        }
        self.config = {
            "target": self.target,
            "original":"",
            "header":"",
            "joined":[],
            'egroup':self.egroup,
            "popsize":popsize,
            "childsize":childsize,
            "genebatch":genebatch,
            "vfactor":vfactor,
            "compete": compete,
            "enhance": enhance,
            "EDmodel": self.EDmodel.config,
            "Filter": self.Filter.config,
            "path": path,
            "loglevel": loglevel,
        }
    
    def getmol(self, smiles):
        """
        getmol:
            Validate the smiles string and return mol using rdkit.

        Parameters:
          smiles - mol in SMILES format

        Returns:
            mol in rdkit format
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        try:
            Chem.Kekulize(mol)
        except Exception as e:
            return None
        return mol
    
        
    def reset(self, ancestor):
        """
        Reset:
            initialize the population
        Parameters:
          ancestor - the ancestor of the population
        """
        if self.stat['init']: # save and start a new session
            self.save()
            self.log = []
            self.stat = {
                'init': True,
                'epoch': 0,
                "header":ancestor,
                "best":ancestor,
                "popsize": 1,
                "score": self.Filter.get_d_scores(self.pop[:10])
            }
        self.config['savepath'] = f"{self.config['path']}/{self.target}_{time.strftime('%Y_%m%d_%H%M%S',time.localtime())}.json"
        ancestor = self.EDmodel.cleanUp(ancestor)
        mol = Chem.MolFromSmiles(ancestor)
        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
        self.EDmodel.encode(ancestor)
        assert len(Chem.MolToSmiles(mol)) > 0
        self.config['original'] = ancestor
        self.header = ancestor
        self.uheader = ancestor
        self.headers = [ancestor]
        self.pop = [ancestor]
        self.epoch = 0
        self.update(new=True)
        print('init success')
    
    def test(self):
        genes = self.EDmodel.encode(self.headers[0])
        child = self.EDmodel.multisample(genes[0], self.headers)
        return genes, child
        
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.config = data['config']
        self.pop = data['pop']
        self.header = data['header']
        self.headers = data['headers']
        self.epoch = data['epoch']
        self.log = data['log']
        
        self.EDmodel.load(self.config['EDmodel'])
        self.Filter.load(self.config['Filter'])
        self.popsize = self.config['popsize']
        self.vfactor = self.config['vfactor']
        self.childsize = self.config['childsize']
        self.genebatch = self.config['genebatch']
        self.compete = self.config['compete']
        self.enhance = self.config['enhance']
        self.update(new=True)
    
    def save(self, savepath=None):
        data = {
            "date": time.strftime("%Y-%m-%d %H:%M",time.localtime()),
            "config": self.config,
            "pop": self.pop,
            "header": self.header,
            "headers": self.headers,
            "epoch": self.epoch,
            "log":self.log,
        }
        if not savepath:
            try:
                savepath = self.config['savepath']
            except:
                self.config['savepath'] = f"{self.config['path']}/{self.target}_{time.strftime('%Y_%m%d_%H%M%S',time.localtime())}.json"
                savepath = self.config['savepath']
        with open(savepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def showconfig(self):
        print("GA config:\n", json.dumps(self.config, indent=2))
            
    def update(self, new=False):
        now = time.strftime("%Y-%m-%d %H:%M",time.localtime())
        if new:
            print("GA config:\n", json.dumps(self.config, indent=2))
            self.log.append(f"{now}: new mol {self.config['header']} start")
        else:
            stat = {
                'init': True,
                "epoch": self.epoch,
                "header":self.header,
                "best":self.pop[0],
                "popsize": len(self.pop),
                "pop10":self.pop[:10],
                "score": self.Filter.get_d_scores(self.pop),
                "spent": int(time.time() - time.mktime(time.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')))
            }
            if self.loglevel ==1:
                stat = {
                    'init': True,
                    "epoch": self.epoch,
                    "header":self.header,
                    "best":self.pop[0],
                    "popsize": len(self.pop),
                    "score": self.Filter.get_d_scores(self.pop[:20]),
                    "spent": int(time.time() - time.mktime(time.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')))
                }
            self.stat = stat
            log = json.dumps(stat, indent=2)
            if self.loglevel>0:
                print(log)
            self.log.append(log)
                
    def join(self, smiles):
        if type(smiles) == str:
            smiles = [smiles]
        joined = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                Chem.Kekulize(mol)
                Chem.SanitizeMol(mol)
                joined.append(smi)
            except Exception as e:
                print(e, ' error')
        if len(joined) ==0:
            print('None valid')
            return
        print(joined) #         self.egroup = list(set(self.egroup) | self.modifier.deComp(joined))
        self.joined += joined
        self.config['joined'] = self.joined
        
    def showmol(self, smiles):
        """
        Show of the headers and population.
        """
        nums = len(smiles)
        colum = 5
        if nums>colum:
            plt.figure(figsize=(colum*8, (nums//colum)*10))
        else:
            plt.figure(figsize=(colum*8, 10))
        for i in range(nums):
            ax = plt.subplot(nums//colum+1, colum, i+1)
            mol = Chem.MolFromSmiles(smiles[i])
            ax.set_title(smiles[i])
            pic = Draw.MolToImage(mol, kekulize=True)
            plt.imshow(pic)
            plt.xticks([])
            plt.yticks([])
        plt.show()
            
    def mutation(self, _smi, npoint=0):
        smi = [_smi] + [self.EDmodel.regen_smi(_smi) for i in range(self.genebatch)]
        smi = self.EDmodel.encodable(smi)
        z_mean, z_var, z = self.EDmodel.encode(smi)
        z1 = z_mean + np.exp(0.5 * z_var) * np.random.random(size=z.shape) * self.vfactor
        return z1
    
    def crossover(self, _smi1, _smi2, npoint=0):
        smi1 = [_smi1] + [self.EDmodel.regen_smi(_smi1) for i in range(self.genebatch)]
        smi2 = [_smi2] + [self.EDmodel.regen_smi(_smi2) for i in range(self.genebatch)]
        smi1 = self.EDmodel.encodable(smi1)
        smi2 = self.EDmodel.encodable(smi2)
        z_mean1, z_var1, z1 = self.EDmodel.encode(smi1)
        z_mean2, z_var2, z2 = self.EDmodel.encode(smi2)
        z_mean = z_mean1.copy()
        z_var = z_var1.copy()
        npoint = int(z_mean.shape[1]*self.vfactor)
        for i in range(z_mean.shape[0]):
            idx = np.random.choice(np.arange(z_mean.shape[1]),np.random.randint(0, npoint), replace=False)
            z_mean[i,idx] += (z_mean2[i, idx] - z_mean[i,idx] )*self.vfactor
            idx = np.random.choice(np.arange(z_mean.shape[1]),np.random.randint(0, npoint), replace=False)
            z_var[i,idx] += (z_var2[i, idx] - z_var[i,idx] )*self.vfactor
        z = z_mean + np.exp(0.5 * z_var) * np.random.random(size=z_mean.shape) * self.vfactor
        return z
    
    def multiply(self):
        """
        Multiple of the population
        """
        def compare(lmol, rmol):
            if self.Filter.dscore_map[rmol] < self.Filter.dscore_map[lmol]:
                return 1
            elif self.Filter.dscore_map[lmol] < self.Filter.dscore_map[rmol] :
                return -1
            else:
                return 0
        if self.enhance==1:
            pop += [self.EDmodel.regen_smi(x) for x in pop]
        if self.enhance==2:
            try:
                frags = pop[:2]
                if len(self.joined)>0:
                    frags += self.joined
                for i in range(2):
                    frags = self.modifier.deComp(frags)
                frags |= set(self.egroup)
                self.egroup = [x for x in frags if len(x) < 10]
                chose = np.random.randint(min(len(pop), 5))
                pop = list(set(pop) | self.modifier.Renu(pop[chose], frags=self.egroup, n=self.childsize) )
                pop = self.EDmodel.encodable(pop)
                pop = self.Filter.run(pop, self.headers)
            except Exception as e:
                print(e,' enhance failed')
                
        pop = list(self.pop)
        mgenes = np.array(list(map(self.mutation, pop)))
        
        x = np.arange(len(pop))/len(pop)
        x = np.exp(-x**2/2) / (np.sqrt(2*np.pi))+0.3 
        parent1 = x > np.random.random(len(pop))
        parent1 = np.where(parent1)[0][:self.genebatch]
        parent2 = x > np.random.random(len(pop))
        parent2 = np.where(parent2)[0][:self.genebatch]
        cgenes = []
        for i in parent1:
            for j in parent2:
                if j > i:
                    cgenes.append(self.crossover(pop[i], pop[j]))
                if len(cgenes) > self.genebatch:
                    break
            if len(cgenes) > self.genebatch:
                break
        for j in self.joined:
            for i in pop:
                cgenes.append(self.crossover(i, j))
        cgenes = np.array(cgenes)
        
        nextpop = set(pop)
        
        for gene in mgenes:
            children = self.EDmodel.multisample(gene, self.headers)
            nextpop |= set(children)
        
        for gene in cgenes:
            children = self.EDmodel.multisample(gene, self.headers)
            nextpop |= set(children)
        
        pop = self.EDmodel.encodable(nextpop)
        pop = self.Filter.run(pop, self.headers)
        pop = [m for m in pop if self.Filter.dscore_map[m] < 0]
        if len(pop) == 0:
            pop = self.headers
        pop.sort(key = functools.cmp_to_key(compare))
        self.pop = pop[:self.popsize]
        if self.pop[0] != self.headers[-1] and self.pop[0] not in self.headers:
            if self.uheader == self.pop[0]:
                self.headers.append(self.pop[0])
                if self.compete:
                    self.header = self.pop[0]
            else:
                self.uheader = self.pop[0]
        
    def envolve(self, epoch=10, er=0):
        """
        Envolve of the population.

        Parameters:
          epoch - envolve epoch of the population
        """
        epoch += self.epoch
        while self.epoch < epoch:
            self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f'Epoch {self.epoch} start: ', self.start_time)
            self.multiply()
            self.update()
            self.epoch += 1
            self.save()
            if self.epoch>5 and self.epoch > epoch*2 //3 and len(self.pop) == 1:
                break
                
        if len(self.pop) == 1:
            print('failed')
        else:
            self.save()
            if self.loglevel >-1:
                print('Headers:')
                self.showmol(self.headers)
                print('Pop:')
                self.showmol(self.pop)