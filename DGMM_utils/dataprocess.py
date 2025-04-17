import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from functools import partial
from multiprocessing import Pool
RDLogger.DisableLog("rdApp.*")
MAXLEN=80
chars = [
    ' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
    '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\',
    ']', 'c', 'i', 'l', 'n', 'o', 'p', 'r', 's']
char_index = dict([(x,y) for y,x in list(enumerate(chars))])

alphabet = [
    '[#Branch1]', '[#Branch2]', '[#C]', '[#N]', '[#Ring1]', '[#Ring2]', '[#S]',
    '[-/Ring1]', '[-/Ring2]', '[-\\Ring1]', '[-\\Ring2]', '[/B]', '[/Br]', '[/C@@H1]',
    '[/C@@]', '[/C@H1]', '[/C@]', '[/CH0]', '[/CH2]', '[/C]', '[/Cl]', '[/F]', '[/I]',
    '[/NH1]', '[/N]', '[/OH0]', '[/O]', '[/P@@]', '[/P@H1]', '[/P@]', '[/P]', '[/S@@]',
    '[/S@]', '[/SH1]', '[/S]', '[/Si]', '[17O]', '[=17O]', '[=B]', '[=Branch1]', '[=Branch2]',
    '[=CH0]', '[=C]', '[=NH0]', '[=N]', '[=O]', '[=P@@H1]', '[=P@@]', '[=P@H1]', '[=P@]',
    '[=PH1]', '[=P]', '[=Ring1]', '[=Ring2]', '[=S@@]', '[=S@]', '[=SH1]', '[=S]', '[=Si]',
    '[B]', '[Br]', '[Branch1]', '[Branch2]', '[C@@H1]', '[C@@]', '[C@H1]', '[C@]', '[CH0]',
    '[CH1]', '[CH2]', '[C]', '[Cl]', '[F]', '[I+3]', '[I]', '[N@@]', '[N@]', '[NH0]', '[NH1]',
    '[N]', '[OH0]', '[O]', '[P@@H1]', '[P@@]', '[P@H1]', '[P@]', '[PH1]', '[P]', '[Ring1]', '[Ring2]',
    '[S@@H1]', '[S@@]', '[S@H1]', '[S@]', '[SH1]', '[SH3]', '[S]', '[Si@@]', '[Si@]', '[SiH1]',
    '[SiH2]', '[SiH3]', '[Si]', '[\\B]', '[\\Br]', '[\\C@@H1]', '[\\C@@]', '[\\C@H1]', '[\\C@]',
    '[\\CH0]', '[\\CH2]', '[\\C]', '[\\Cl]', '[\\F]', '[\\I]', '[\\NH1]', '[\\N]', '[\\OH0]', '[\\O]',
    '[\\P@@]', '[\\P@H1]', '[\\P@]', '[\\P]', '[\\S@@]', '[\\S@]', '[\\S]', '[\\Si]', '[nop]']
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

def smi2sfi(smi):
    label = sf.selfies_to_encoding(
        selfies=sf.encoder(smi),
        vocab_stoi=symbol_to_idx,
        pad_to_len=MAXLEN,
        enc_type="label"
    )
    return np.array(label).astype('float')

def smiles2inp(smiles, thread=10):
    if type(smiles) == str:
        smiles = [smiles]
    worker = Pool(thread)
    tokens = worker.map(smi2sfi, smiles)
    worker.close()
    worker.join()
    return np.array(tokens)

def getlabels(sfis, pad_to_len=80):
    labels = []
    ohs = []
    for sfi in sfis:
        label, oh = sf.selfies_to_encoding(
            selfies=sfi,
            vocab_stoi=symbol_to_idx,
            pad_to_len=pad_to_len,
#             enc_type="label"
        )
        labels.append(label)
    return np.array(labels).astype('float')

def get_fp(smiles):
    res = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = MACCSkeys.GenMACCSKeys(mol)
            res.append(fp.ToList())
        except:
            print(smi, ' error')
            res.append([])
    return np.array(res)

def smiles2label(smiles, canonize_smiles=False, maxlen=80):
    if len(smiles) >= maxlen:
        return None
    if canonize_smiles:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)
    if len(smiles) >= maxlen:
        return None
    encode = [0]*maxlen
    for idx,s in enumerate(smiles):
        if s in char_index:
            encode[idx] = char_index[s]+1
        else:
            encode[idx] = len(chars)+2
    return encode

def getslabels(smis):
    labels = []
#     indices = np.eye(80)
    for smi in smis:        
#         labels.append(indices[smiles2label(smi)])
        labels.append(smiles2label(smi))
    return np.array(labels).astype('float')
