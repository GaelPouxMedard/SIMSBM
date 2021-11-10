import numpy as np
import sparse
import itertools
import pickle
import TF
import os

def normalized(a):
    newCoords, vals = [], []
    dicMax={}

    nnz = list(zip(*a.nonzero()[:-1]))
    lg=len(nnz)
    for ind, c in enumerate(nnz):
        if ind%(lg//10)==0: print("Normalization", ind*100./lg, "%")

        if c not in dicMax: dicMax[c] = np.sum(a[c])

        vals.append(dicMax[c])

    vals = np.array(a.data / np.array(vals), dtype=float)
    aNew = sparse.COO(a.nonzero(), vals, shape=a.shape)
    return aNew

def readMatrix(filename):
    try:
        return sparse.load_npz(filename.replace(".txt", ".npz"))
    except:
        with open(filename, 'r') as outfile:
            dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
            for i in range(len(dims)):
                dims[i] = int(dims[i])

        new_data = np.loadtxt(filename).reshape(dims)
        return new_data

    # return sparse.csr_matrix(new_data)

def getName(DS, folder, nbClus, nbInterp, features):
    codeSave = ""
    for i in range(len(DS)):
        for j in range(DS[i]):
            codeSave += str(features[i]) + "-"
    codeSave = codeSave[:-1]

    codeSaveTF = ""
    for i in range(len(DS)):
        for _ in range(nbInterp[i]):
            codeSaveTF += str(features[i]) + "-"
    codeSaveTF = codeSaveTF[:-1]

    featToClus = []
    nbClus = np.array(nbClus)
    for iter, interp in enumerate(nbInterp):
        for i in range(interp):
            featToClus.append(iter)
    featToClus = np.array(featToClus, dtype=int)


    codeT = ""
    for i in range(len(DS)):
        for _ in range(nbInterp[i]):
            codeT += str(nbClus[i]) + "-"
    codeT = codeT[:-1]

    return "Output/" + folder + "/" + codeSave, codeT, "Output/" + folder + "/" + codeSaveTF


def writeAlphaTF(fname, alphaTr):
    alphaNorm = normalized(alphaTr)
    d = alphaNorm.data
    print("LONGUEUR TRAINING DATA", len(d))
    print(alphaTr)
    with open(fname+"_MatrixTF.txt", "a") as f:
        f.truncate(0)
        for i, c in enumerate(zip(*alphaNorm.nonzero())):
            for c_i in c:
                f.write(str(c_i)+",")
            f.write(str(d[i])+"\n")

def moveRes(fname, params, codeClus):
    cmdCoreRen = f"ren ./{fname}_TF/{codeClus}_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_core.npy {fname[fname.rfind('/')+1:]}_TF_core.npy"
    cmdCoreRen = cmdCoreRen.replace("/", "\\")
    cmdCoreMv = f"move /Y ./{fname}_TF/{fname[fname.rfind('/')+1:]}_TF_core.npy ./{fname[:fname.rfind('/')]}/"
    cmdCoreMv = cmdCoreMv.replace("/", "\\").replace("\Y", "/Y")

    os.system(cmdCoreRen)
    os.system(cmdCoreMv)

    cmdURen = f"ren ./{fname}_TF/{codeClus}_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[4]}_U.npy {fname[fname.rfind('/')+1:]}_TF_U.npy"
    cmdURen = cmdURen.replace("/", "\\")
    cmdUMv = f"move /Y ./{fname}_TF/{fname[fname.rfind('/')+1:]}_TF_U.npy ./{fname[:fname.rfind('/')]}/"
    cmdUMv = cmdUMv.replace("/", "\\").replace("\Y", "/Y")

    os.system(cmdURen)
    os.system(cmdUMv)


def run(DS, folder, nbClus, nbInterp, features, norm, step, N):
    fnameAlpha, codeClus, fname = getName(DS, folder, nbClus, nbInterp, features)
    print(fname)

    alphaTr, alphaTe = readMatrix(fnameAlpha.replace("Output", "Data")+"_AlphaTr.npz"), readMatrix(fnameAlpha.replace("Output", "Data")+"_AlphaTe.npz")
    print(alphaTr)

    toRem, ind = [], 0
    for i in range(len(DS)):
        if DS[i] != nbInterp[i]:
            for t in range(ind, ind+DS[i]-nbInterp[i]):
                toRem.append(t)
        ind += DS[i]
    if len(toRem)!=0:
        alphaTr = alphaTr.sum(toRem)
        alphaTe = alphaTe.sum(toRem)

    codeClus += f"-{np.min([20, int(list(alphaTr.shape)[-1])])}"  # Clusters pour l'output

    print(codeClus)
    
    params = [norm, norm, step, step, N]
    writeAlphaTF(fname, alphaTr)
    import platform
    strBS = "\\"
    if platform.system()=="Windows":
        os.system(f"python TF.py --train {fname+'_MatrixTF.txt'} --test {fname+'_MatrixTF.txt'} --model {fname.replace('/', strBS)+'_TF'} --k {codeClus} --reg {params[0]} --regS {params[1]} --lr {params[2]} --lrS {params[3]} --maxEpo {params[4]} --fname {fname}")
    else:
        os.system(f"python3 TF.py --train {fname+'_MatrixTF.txt'} --test {fname+'_MatrixTF.txt'} --model {fname+'_TF'} --k {codeClus} --reg {params[0]} --regS {params[1]} --lr {params[2]} --lrS {params[3]} --maxEpo {params[4]} --fname {fname}")
    print(fname+'_TF')
    #moveRes(fname, params, codeClus)


'''
folder = "Spotify"
if "Spotify" in folder:
    features = [0]
    DS = [3]
    nbInterp = [3]
    nbClus = [20]

fname, codeClus = getName(DS, folder, nbClus, nbInterp)

modU = np.load(f"{fname}_TF_U.npy", allow_pickle=True)
modCore = np.load(f"{fname}_TF_core.npy", allow_pickle=True)

for m in modU:
    print(np.shape(m))
print(np.shape(modU), np.shape(modCore))
'''