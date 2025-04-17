import math
import pickle
import gzip
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, rdMolDescriptors, Descriptors
import os

pwd = os.path.dirname(os.path.abspath(__file__))
RDLogger.DisableLog('rdApp.*')

DEBUG=False

def get_SA():
    # generate the full path filename:
    data = pickle.load(gzip.open(pwd + '/fpscores.pkl.gz'))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    return outDict
        
SAdb = get_SA()

def SAscore(m, _fscores=None):
    # fragment score
    if not _fscores:
        _fscores = SAdb
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf
    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
#     nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(m)
    nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)
    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5
    sascore = score1 + score2 + score3
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    return sascore

def get_QED(smi):
    mol = Chem.MolFromSmiles(smi)
    return QED.default(mol)


def acceptance_score(ex_constraint, simthd, qedthd, sathd, scf, afp, smiles):
    mol = Chem.MolFromSmiles(smiles)
    score = 1
    try:
        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
        rings = Chem.GetSymmSSSR(mol)
        molwt = Descriptors.MolWt(mol)
        if ex_constraint != None:
            score *= ex_constraint(mol)
        if score == 0:
            return 0
        for i in range(len(rings)):
            if len(list(rings[i])) > 6 or len(list(rings[i])) < 4:
                if DEBUG:
                    print('\tBad Ring ')
                return 0
            for j in range(i):
                if len(set(list(rings[i])) & set(list(rings[j]))) > 3:
                    if DEBUG:
                        print('\tBad complex Ring')
                    return 0
        if molwt < 200:
            score *= 0.1
        elif molwt < 500:
            score *= 0.5
        else:
            score *= 2

        qed_score = QED.default(mol)
        if qed_score < 0.2:
            if DEBUG:
                print('\tQED too Low')
            return 0
        elif qed_score < qedthd:
            score *= qed_score*3
        else:
            score *= 2

        N_num = sum([1 for x in mol.GetAtoms() if x.GetSymbol()=='N'])
        if N_num > 0.35* len(mol.GetAtoms()):
            if DEBUG:
                print('\ttoo Many Nitrogen ')
            return 0
        if N_num > 0.25* len(mol.GetAtoms()):
            score *= 0.5
        multiBond = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds() if b.GetBondTypeAsDouble() >=2]
        if len(sum(multiBond , [])) != len(set(sum(multiBond , []))):
            if DEBUG:
                print('\tMulti Bond Share')
            return 0
        sa_score = SAscore(mol)
        if sa_score > 5:
            if DEBUG:
                print('\tSA too high')
            return 0
        elif sa_score > sathd:
            score *= 0.3/sa_score
        else:
            score *= 2

        try:
            if mol.HasSubstructMatch(scf):
                return score
            else:
                fp = Chem.RDKFingerprint(mol)
                sim_score = DataStructs.TanimotoSimilarity(fp, afp)
                if sim_score < simthd/2:
                    if DEBUG:
                        print('\tSIM too Low')
                    return score*0.1
                if sim_score > 0.7:
                    return score*2
                if sim_score < simthd:
                    return score*0.5
        except Exception as e:
            if DEBUG:
                print(e,' or ancestor not init.')
    except Exception as e:
        print(smiles, e)
        return 0
    return score