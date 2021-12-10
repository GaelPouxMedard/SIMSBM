import os
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'
import numpy as np
import random
# from scipy import sparse
import sparse
import datetime
from copy import deepcopy as copy
from scipy.special import binom
import time
import itertools
import sys
from itertools import product

'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
pause()
'''

'''
from memory_profiler import profile
import gc
fp = open("memory_profiler_Norm.log", "a")
@profile(stream=fp, precision=5)
'''

'''  Why we cannot set the initial diagonals equals to frequency
popFeat = [200]
nbClus = [20]
nbOutputs = 100
nbInterp = [2]
A = np.random.random((popFeat[0], nbClus[0]))

shape = []
for i, p in enumerate(popFeat):
    for _ in range(nbInterp[i]):
        shape.append(p)
shape.append(nbOutputs)

Z = np.random.random(shape)
Z_mask = np.zeros(shape)
indices_diag = None
for i, dim in enumerate(nbInterp):
    n = popFeat[i]
    if dim == 1:
        x = np.ones((n))
    else:
        x = np.zeros([n for _ in range(dim)])
        L = np.ones((n))
        x[np.diag_indices(n, ndim=dim)] = L
    if indices_diag is None:
        indices_diag = x
    else:
        indices_diag = np.tensordot(indices_diag, x, axe=0)
indices_diag = np.tensordot(indices_diag, np.ones((nbOutputs)), axes=0)
Z_mask[indices_diag.astype(bool)] = 1
Z *= Z_mask
Z_mask = Z_mask.astype(bool)
# AX = B

X_inf = np.linalg.pinv(A).dot(np.linalg.pinv(A).dot(Z))
print(np.allclose(A.dot(A.dot(X_inf))[Z_mask], Z[Z_mask]))
print(list(A.dot(A.dot(X_inf))[Z_mask].flatten()))
print(list(Z[Z_mask].flatten()))
print(np.abs(np.mean(A.dot(A.dot(X_inf))[Z_mask]-Z[Z_mask])))
sys.exit()
'''


seed = 111
np.random.seed(seed)
random.seed(seed)

sparseMatrices = True

#// region Manipulates the data files


def startAtTime(string="1970-01-01 00:00:00"):
    while True:
        if datetime.datetime.now() > datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S"):
            print("STARTED")
            break
        else:
            time.sleep(60)
            
startAtTime(string="2020-10-07 21:00:00")

# Generic function to save matrices of dim=2 or 3
def writeMatrix(arr, filename):
    try:
        sparse.save_npz(filename.replace(".txt", ""), arr)
    except:
        try:
            np.save(filename, arr)
        except:
            with open(filename, 'a') as outfile:
                outfile.truncate(0)
                outfile.write('# Array shape: {0}\n'.format(arr.shape))
                for slice_2d in arr:
                    np.savetxt(outfile, slice_2d)
                    outfile.write("# New slice\n")

    # np.savetxt(filename, arr)

# Generic function to read matrices of dim=2 or 3
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

# Saves the model's parameters theta, p, corresponding likelihood and held-out likelihood
def writeToFile_params(folder, thetas, p, maxL, HOL, featToClus, popFeat, nbClus, run=-1):
    while True:
        try:
            s=""
            folderParams = "Output/" + folder + "/"
            pass
            codeT=""
            for i in featToClus:
                codeT += str(nbClus[i])+"-"
            codeT = codeT[:-1]

            for i in range(len(thetas)):
                writeMatrix(thetas[i], folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"theta_"+str(i)+"_Inter_theta")

            writeMatrix(p, folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_p")

            f = open(folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_L.txt", "w")
            f.write(str(maxL) + "\n")
            f.close()

            f = open(folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_HOL.txt", "w")
            f.write(str(HOL) + "\n")
            f.close()

            f = open(folderParams + "/T="+codeT+"_%.0f_" % (run)+s+"Inter_FeatToClus.txt", "w")
            for i in range(len(featToClus)):
                f.write(str(i)+"\t" + str(featToClus[i]) + "\t" + str(popFeat[i]) + "\n")
            f.close()

            break

        except Exception as e:
            print("Retrying to write file -", e)

# Synthetic data
def normalized(a, axis=-1):
    l2 = np.sum(a, axis=axis)
    l2[l2==0]=1
    return a / np.expand_dims(l2, axis)

def getSynthData(featToClus, nbOutputs, popFeat, nbClus):
    nbFeat = len(featToClus)
    nbLayers = len(nbClus)

    thetas = []  # theta(f/g/...), f/g/..., k/l/...
    for i in range(nbLayers):
        pop = 0
        for j in range(len(featToClus)):
            if featToClus[j]==i:
                pop = popFeat[j]
                break

        t = np.random.random((pop, nbClus[i]))
        for p in range(pop):
            pass
            trous = np.zeros((nbClus[i]))
            trous[np.random.randint(0, nbClus[i])] = 1
            t[p] *= trous
        t = t / np.sum(t, axis=1)[:, None]
        thetas.append(t)

    shape = [nbClus[featToClus[i]] for i in range(nbFeat)]
    shape.append(nbOutputs)
    p = np.random.random(tuple(shape))  # K, K, L, O
    p *= (p>0.5).astype(int)
    for l in range(nbLayers):
        indTemp = []
        for i in range(len(featToClus)):
            if featToClus[i]==l:
                indTemp.append(i)

        a = list(range(nbFeat+1))
        for i in indTemp:
            for j in indTemp:
                aTemp = copy(a)
                t = aTemp[i]
                aTemp[i]=aTemp[j]
                aTemp[j]=t
                p = (p + np.transpose(p, aTemp))/2

    p = normalized(p, axis=-1)

    if sparseMatrices:
        probs = p
        numEntries = int(min(min(popFeat), nbOutputs)**(len(popFeat)+1))
        print("Simulation de "+str(numEntries/100000)+" entries")
        coords = [tuple(np.random.randint(0, min(min(popFeat), nbOutputs), nbFeat+1)) for _ in range(int(numEntries/100000))]
        vals=[]
        for f in coords:
            prob = p
            for i in range(nbFeat):
                prob = thetas[featToClus[nbFeat-i-1]][f[nbFeat-i-1]].dot(prob)
            v = prob[f[-1]]
            vals.append(v)

        shape = [popFeat[i] for i in range(nbFeat)]
        shape.append(nbOutputs)
        probs = sparse.COO(list(zip(*coords)), np.array(vals), shape=shape)

        alphaTraining = probs*1000
        alphaTest = probs*100

    else:
        probs = p
        for i in range(nbFeat):
            probs = thetas[featToClus[nbFeat-i-1]].dot(probs)
        alphaTraining = np.array(probs*1000, dtype=float)
        alphaTest = np.array(probs*100, dtype=float)

    codeT = ""
    for i in featToClus:
        codeT += str(nbClus[i]) + "-"
    codeT = codeT[:-1]
    writeMatrix(alphaTraining, "Data/Synth/T="+codeT+"_Inter_alphaTr.txt")
    writeMatrix(alphaTest, "Data/Synth/T="+codeT+"_Inter_alphaTe.txt")

    return alphaTraining, alphaTest, thetas, p

def weightedBinomCoeff(a, arrFeat, nbFeat):
    inds = list(range(nbFeat+1))
    for i in arrFeat:
        inds.remove(i)

    inds = tuple(inds)
    #asum = np.sum(a, inds)
    #nnz = asum.nonzero()

    nnz = a.nonzero()
    unzippednnz = np.array(list(zip(*nnz)))
    n = len(arrFeat)
    #weights = np.zeros(asum.shape)
    coords, vals = nnz, []
    lg = len(unzippednnz)
    for j, c in enumerate(unzippednnz):
        order = n-len(set(c[arrFeat]))+1
        w = 0
        for o in range(1, order+1):
            #weights[c] += binom(n, o)*o
            w += binom(n, o)*o
        vals.append(w)

    weights = sparse.COO(coords, vals, shape=a.shape)

    #print(weights)
    #for i in range(len(inds)):
    #    weights = np.expand_dims(weights, inds[i])
    return weights

#// endregion


#// region Fit tools

# Computes the likelihood of the model
def likelihood_old(thetas, p, alpha, featToClus):
    nbFeat = len(featToClus)
    print("prout")

    if sparseMatrices:
        coords = alpha.nonzero()
        vals=[]
        p = np.moveaxis(p, -1, 0)
        for f in zip(*coords):
            prob = p[f[-1]]
            for i in range(nbFeat):
                tet = thetas[featToClus[nbFeat-i-1]][f[nbFeat-i-1]]  # k
                prob = prob.dot(tet)
            v = prob
            if v==0: v=1e-20
            vals.append(v)
        L = alpha.data.dot(np.log(np.array(vals)))


    else:
        probs = p
        for i in range(nbFeat):
            probs = thetas[featToClus[nbFeat-i-1]].dot(probs)
        L = np.sum(alpha * (np.log(probs+1e-20)))

    return L

def likelihood(alpha, Pfo):
    return np.sum(alpha*np.log(Pfo+1e-20))

def getPfo(alpha, thetas, p, featToClus):
    nbFeat = len(featToClus)

    coords = alpha.nonzero()
    vals = []
    p = np.moveaxis(p, -1, 0)
    for f in zip(*coords):
        prob = p[f[-1]]
        for i in range(nbFeat):
            tet = thetas[featToClus[nbFeat - i - 1]][f[nbFeat - i - 1]]  # k
            prob = prob.dot(tet)
        v = prob
        if v == 0: v = 1e-20
        vals.append(v)
    p = np.moveaxis(p, 0, -1)

    Pfo = sparse.COO(coords, np.array(vals), shape=alpha.shape)

    if not sparseMatrices:
        Pfo=Pfo.todense()

    return Pfo

# Recursive function to build the dict whose keys are nonzero values of observations
def buildDicCoords(f, dic):
    if len(f)==1:
        dic[f[0]]=1
    else:
        if f[0] not in dic: dic[f[0]]={}
        dic[f[0]] = buildDicCoords(f[1:], dic[f[0]])

    return dic

#Recursive function to build P_{f,o} (the denominator of omega, see main paper)
def getAllProbs(dic, vals, prob, thetas, featToClus, feat, nbFeat):
    if feat == nbFeat:
        for k in dic:
            vals.append(prob[k])
    else:
        for k in dic:
            tet = thetas[featToClus[feat]][k]
            probk = np.moveaxis(prob, 1, -1).dot(tet)  # Move the axis in 1 to the end bc with recurrency the indices "slide" back to 1
            vals = getAllProbs(dic[k], vals, probk, thetas, featToClus, feat+1, nbFeat)

    return vals

def prod(a):
    x = 1
    for a_i in a:
        x *= a_i
    return x

# EM steps for p
def maximization_p(alpha, featToClus, popFeat, nbClus, theta, pPrev, Pfo):
    nbFeat = len(featToClus)

    terme1 = alpha / (Pfo + 1e-20)  # features, o
    for t in range(nbFeat):
        sizeAfterOperation = prod(terme1.shape)*theta[featToClus[nbFeat - t - 1]].shape[1]*8/terme1.shape[-2]
        if sizeAfterOperation < 2e9:  # As soon as we can use non-sparse we do it (2Gb)
            #print("P dense")
            tet = theta[featToClus[nbFeat - t - 1]].T  # K F1
            terme1 = np.dot(tet, terme1)
        else:
            #print("P sparse")
            tet = sparse.COO(theta[featToClus[nbFeat - t - 1]].T)  # K F1
            terme1 = sparse.COO(np.dot(tet, terme1))

    # print(terme1.shape)  # clusters, o
    if prod(terme1.shape)*8 < 2e9 and type(terme1) is not type(np.array([])):
        terme1 = terme1.todense()

    grandDiv = np.sum(pPrev*terme1, -1)
    grandDiv = np.expand_dims(grandDiv, -1)
    p = pPrev*terme1 / (grandDiv+1e-20)

    '''  Explicit computation for 2 layers with 2 interactions each
    omegatop = np.moveaxis(pPrev, -1, 0)
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, None]*theta[0].T[None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, :, None]*theta[0].T[None, None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, :, :, None]*theta[1].T[None, None, None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = omegatop[:, :, :, :, :, :, :, :, None]*theta[1].T[None, None, None, None, :, None, None, None, :]
    print(omegatop.shape)
    omegatop = np.moveaxis(omegatop, 0, -1)
    print(omegatop.shape)

    omega = omegatop/omegatop.sum(axis=(0, 1, 2, 3))[None, None, None, None, :, :, :, :, :]
    p2 = omega[:, :, :, :, :, :, :, :, :]*alpha[None, None, None, None, :, :, :, :, :]
    p2 = p2.sum(axis=(4, 5, 6, 7))
    p2 = p2/p2.sum(axis=-1)[:, :, :, :, None]

    '''

    return p

# EM steps for theta
def maximization_Theta(alpha, featToClus, nbClus, thetaPrev, p, Cm, Pfo):
    nbFeat = len(featToClus)
    nbNatures = len(nbClus)
    thetas = []

    for nature in range(nbNatures):
        theta_base = thetaPrev[nature]

        alphadivided = alpha / (Pfo + 1e-20)  # f1 f2 g r

        arrFeat = []
        for feat in range(len(featToClus)):
            if nature==featToClus[feat]: arrFeat.append(feat)
        nbInter = len(arrFeat)

        if len(arrFeat) == 0:
            continue

        # Sum over all other natures' permutations (they are identical for this nature)
        omega = np.moveaxis(p, arrFeat, range(len(arrFeat)))
        for t in reversed(range(nbFeat)):
            if t not in arrFeat:
                omega = thetaPrev[featToClus[t]].dot(omega)

        for i in range(1, nbInter):  #Keep theta_mn out from omega
            omega = np.dot(theta_base, omega)

        omega = omega * nbInter

        idxalpha = tuple(arrFeat[1:]+[i for i in range(nbFeat) if i not in arrFeat]+[-1])
        idxomega = tuple([i for i in range(len(arrFeat)-1)]+[i for i in range(len(arrFeat)-1, nbFeat-1)]+[-1])
        omegalpha = np.tensordot(omega, alphadivided, axes=(idxomega, idxalpha)).T

        thetaNatureNew = omegalpha * theta_base / (Cm[nature][:, None]+1e-20)
        thetas.append(thetaNatureNew)

        '''  Explicit summation
        for m in range(I):
            for n in range(K):
                # Explicit
                for ia, ialpha in enumerate(list(zip(*alpha.nonzero()))):
                    indAlpha = tuple(ialpha)[:-1]
                    o = ialpha[-1]
                    im = np.where(np.array(indAlpha)==m)[0]
        
                    cm = list(np.array(indAlpha)[im]).count(m)
        
                    permutOmega = list(itertools.product(list(range(K)), repeat=nbInter))
                    for indOmega in permutOmega:
        
                        if m not in np.array(indAlpha)[im]: continue
                        if n not in np.array(indOmega)[im]: continue
        
                        indOmega = tuple(indOmega)
                        cn = list(np.array(indOmega)[im]).count(n)
        
                        tupInd = list(indAlpha)+list(indOmega)+[o]
                        tupInd = tuple(tupInd)
        
                        theta[m, n] += omega[tupInd]*dataa[ia]*cn
        
            theta[m] = theta[m]/Cm[m]
        '''


    return thetas

# Reduce number of clusters
def distClus(K1, K2):
    return np.sum(abs(K1-K2))/np.sum(K1+K2)

def BIC(thetas, p, alpha, featToClus, Pfo):
    k = np.prod(p.shape)
    for i in range(len(thetas)):
        k += np.prod(thetas[i].shape)
    n = np.sum(alpha)
    #L = likelihood(thetas, p, alpha, featToClus)
    L = likelihood(alpha, Pfo)

    return k * np.log(n) - 2*L

def reduceSetK(alpha, featToClus, popFeat, nbClus, thetas, p, Pfo):
    dists = []
    valMin, ind1min, ind2min, thetaMin = 1, 0, 0, 0
    for theta in list(set(featToClus)):
        axis = np.where(featToClus == theta)[0]
        tet = copy(p)
        tet = np.moveaxis(tet, axis, range(len(axis)))
        nbC = nbClus[theta]
        dists.append(np.zeros((nbC, nbC)))
        for i1, c1 in enumerate(tet):
            for i2, c2 in enumerate(tet):
                dists[theta][i1, i2] = distClus(c1, c2)
                if dists[theta][i1, i2]<valMin and dists[theta][i1, i2]!=0:
                    valMin, ind1min, ind2min, thetaMin = dists[theta][i1, i2], i1, i2, theta

    thetaNew = copy(thetas[thetaMin])
    thetaNew[:, ind1min] += thetaNew[:, ind2min]
    thetaNew = np.delete(thetaNew, ind2min, axis=1)
    thetasNew = copy(thetas)
    thetasNew[thetaMin] = thetaNew

    pNew = copy(p)
    axis = np.where(featToClus == thetaMin)[0]
    for i in axis:
        pNew = np.delete(pNew, ind2min, axis=i)

    nbClusNew = copy(nbClus)
    nbClusNew[thetaMin] -= 1

    print(valMin, ind1min, ind2min, thetaMin)
    print(p.shape)
    print(pNew.shape)

    BICOld = BIC(thetas, p, alpha, featToClus, Pfo)
    BICNew = BIC(thetasNew, pNew, alpha, featToClus, Pfo)

    print(BICOld, BICNew)

    if BICOld>BICNew and valMin<0.3:
        return np.array(nbClusNew), thetasNew, pNew
    else:
        return np.array(nbClus), thetas, p

# Random initialisation of p, theta
def initVars(featToClus, popFeat, nbOutputs, nbLayers, nbClus):
    nbFeat = len(featToClus)
    thetas = []  # theta(f/g/...), f/g/..., k/l/...
    for i in range(nbLayers):
        pop = 0
        for j in range(len(featToClus)):
            if featToClus[j]==i:
                pop = popFeat[j]
                break

        t = np.random.random((pop, nbClus[i]))
        t = t / np.sum(t, axis=1)[:, None]
        thetas.append(t)

    shape = [nbClus[featToClus[i]] for i in range(nbFeat)]
    shape.append(nbOutputs)
    p = np.random.random(tuple(shape))  # K, K, L, O

    # Important to make symmetric initialization, otherwise assumptions made in the algorithm do not hold (maximizationp).
    prev = 0
    for num, i in enumerate(nbInterp):
        permuts = list(itertools.permutations(list(range(prev, prev+int(i))), int(i)))
        p2 = p.copy()
        for per in permuts[1:]:
            arrTot = np.array(list(range(len(p.shape))))
            arrTot[prev:prev+i] = np.array(per)
            p2 = p2 + p.transpose(arrTot)
        p = p2 / len(permuts)  # somme permutations = 1 obs
        prev += i


    p = normalized(p, axis=-1)

    return thetas, p

# Main loop of the EM algorithm, for 1 run
def EMLoop(alpha, featToClus, popFeat, nbOutputs, nbLayers, nbClus, maxCnt, prec, folder, run, Cm, reductionK, dicnnz, nbInterp, features):
    nbFeat = len(featToClus)
    thetas, p = initVars(featToClus, popFeat, nbOutputs, nbLayers, nbClus)
    maskedProbs = getAllProbs(dicnnz, [], np.moveaxis(p, -1, 0), thetas, featToClus, 0, nbFeat)
    Pfo = sparse.COO(alpha.nonzero(), np.array(maskedProbs), shape=alpha.shape)
    maxThetas, maxP = initVars(featToClus, popFeat, nbOutputs, nbLayers, nbClus)
    prevL, L, maxL = -1e20, 0.1, -1e20
    cnt = 0
    i = 0
    iPrev=0
    changeNbClus = False
    nbClusOld = nbClus
    while i < 1000000:  # 1000000 iterations top ; prevents infinite loops but never reached in practice
        #print(i)
        if i%10==0:  # Compute the likelihood and possibly save the results every 10 iterations
            #L = likelihood(thetas, p, alpha, featToClus)
            L = likelihood(alpha, Pfo)
            print(f"Run {run} - Iter {i} - Feat {features} - Interps {nbInterp} - L={L}")

            if ((L - prevL) / abs(L)) < prec:
                cnt += i-iPrev
                if reductionK:
                    if cnt > maxCnt//2:
                        nbClusNew, thetasNew, pNew = reduceSetK(alpha, featToClus, popFeat, nbClus, thetas, p, Pfo)
                        if list(nbClus) != list(nbClusNew):
                            changeNbClus = True
                            nbClus, thetas, p = nbClusNew, thetasNew, pNew
                            cnt = 0

                if cnt > maxCnt:
                        break
            else:
                cnt = 0

            iPrev=i

            if (L > prevL and L > maxL) or changeNbClus:
                changeNbClus = False
                maxThetas, maxP = thetas, p
                maxL = L
                #HOL = likelihood(thetas, p, alpha_Te, featToClus)
                HOL = 0.
                writeToFile_params(folder, maxThetas, maxP, maxL, HOL, featToClus, popFeat, nbClusOld, run)
                print("Saved")
            prevL = L

        maskedProbs = getAllProbs(dicnnz, [], np.moveaxis(p, -1, 0), thetas, featToClus, 0, nbFeat)
        Pfo = sparse.COO(alpha.nonzero(), np.array(maskedProbs), shape=alpha.shape) # Sparse matrix of every probability for observed entries (normalization term of omega)

        pNew = maximization_p(alpha, featToClus, popFeat, nbClus, thetas, p, Pfo)
        thetasNew = maximization_Theta(alpha, featToClus, nbClus, thetas, p, Cm, Pfo)
        p = pNew
        thetas = thetasNew

        i += 1

    return maxThetas, maxP, maxL, nbClus


#// endregion

def runFit(alpha_Tr, nbClus, nbInterp, prec, nbRuns, maxCnt, reductionK, features):
    print(alpha_Tr.shape)
    nbFeat = alpha_Tr.ndim - 1
    nbOutputs = alpha_Tr.shape[-1]
    popFeat = [l for l in alpha_Tr.shape[:-1]]
    print(nbClus)
    nbNatures = len(nbClus)
    featToClus = []
    nbClus = np.array(nbClus)
    for iter, interp in enumerate(nbInterp):
        for i in range(interp):
            featToClus.append(iter)
    featToClus = np.array(featToClus, dtype=int)
    
    print("Pop feats")
    print(popFeat)
    print("Feat to clus")
    print(featToClus)
    print("Nb clus")
    print(nbClus)
    print("Nb interp")
    print(nbInterp)

    dicnnz = {}
    coords = alpha_Tr.nonzero()
    for f in zip(*coords):
        dicnnz = buildDicCoords(f, dicnnz)

    Cm = []
    for nature in range(nbNatures):
        arrFeat = []
        for feat in range(len(featToClus)):
            if nature==featToClus[feat]: arrFeat.append(feat)

        Cm.append(np.zeros((popFeat[arrFeat[-1]])))
        dataa = alpha_Tr.data
        for i, ialpha in enumerate(list(zip(*alpha_Tr.nonzero()))):
            indAlpha = tuple(list(np.array(ialpha)[arrFeat]))
            for m in set(indAlpha):
                im = np.where(np.array(indAlpha)==m)[0]
                cm = list(np.array(indAlpha)[im]).count(m)
                Cm[nature][m] += cm*dataa[i]

    maxL = -1e100
    for i in range(nbRuns):
        print("RUN", i)
        theta, p, L, nbClusNew = EMLoop(alpha_Tr, featToClus, popFeat, nbOutputs, nbNatures, nbClus, maxCnt, prec, folder, i, Cm, reductionK, dicnnz, nbInterp, features)
        #HOL = likelihood(theta, p, alpha_Te, featToClus)
        HOL=0
        if L > maxL:
            maxL = L
            writeToFile_params(folder + "/Final/", theta, p, L, HOL, featToClus, popFeat, nbClus, -1)
            print("######saved####### MAX L =", L)
        print("=============================== END EM ==========================")

def runForOneDS(folder, DS, features, nbInterp, nbClus, buildData, seuil, lim, propTrainingSet, prec, nbRuns, maxCnt, reductionK, sparseMatrices, onlyBuildDS=False):
    print("Reduction K", reductionK)
    print("Features", features)
    print("Structure", nbInterp)
    print("DS", DS)

    if buildData:
        print("Build alphas")
        import BuildAlpha
        alpha_Tr, alpha_Te = BuildAlpha.run(folder, DS, features, propTrainingSet, lim, seuil=seuil)
    else:
        print("Get alphas")
        codeSave = ""
        for i in range(len(features)):
            for j in range(DS[i]):
                codeSave += str(features[i]) + "-"
        codeSave = codeSave[:-1]
        fname = "Data/"+folder+"/"+codeSave
        alpha_Tr, alpha_Te = readMatrix(fname+"_AlphaTr.npz"), readMatrix(fname+"_AlphaTe.npz")

    if onlyBuildDS:
        return 0



    toRem, ind = [], 0
    for i in range(len(DS)):
        if DS[i] != nbInterp[i]:
            for t in range(ind, ind+DS[i]-nbInterp[i]):
                toRem.append(t)
        ind += DS[i]
    if len(toRem)!=0:
        alpha_Tr = alpha_Tr.sum(toRem)
        alpha_Te = alpha_Te.sum(toRem)

    print("Alpha:", len(alpha_Tr.data), alpha_Tr)
    print("Number triplets training:", alpha_Tr.sum())
    
    if not sparseMatrices:
        try:
            alpha_Tr=alpha_Tr.todense()
            alpha_Te=alpha_Te.todense()
        except:
            sparseMatrices=True
            print("=============== SWITCH TO SPARSE =================")

    runFit(alpha_Tr, nbClus, nbInterp, prec, nbRuns, maxCnt, reductionK, features)



# Run the algorithm with command line parameters or not
# TreatData = do you want to redo the entire corpus from raw data
# RetreatEverything = do you want to compute alpha again
# folder = has to do with the project structure

prec = 1e-4  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
maxCnt = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
saveToFile = True
propTrainingSet = 0.9
nbRuns = 10
reductionK = True
lim = -1

# Juste pour que ce soit défini
features = []  # Which features to consider (see key)
DS = []  # Which dataset use (some are already built for interactions)
nbInterp = []  # How many interactions consider for each dataset (reduces DS to this number by summing)
nbClus = []
buildData = bool

seuil=0  # If retreatEverything=True : choose the threshold for the number of apparitions of an nplet.
# If an nplet appears stricly less than "seuil" times, it's not included in the dataset


folder = "PubMed"
if "PubMed" in folder:
    # 0 = symptoms  ;  o = disease
    features = [0]
    DS = [3]
    nbInterp = [3]
    nbClus = [20]
    buildData = True
    seuil = 500
if "Spotify" in folder:
    # 0 = artists  ;  o = next artist
    features = [0]
    DS = [3]
    nbInterp = [1]
    nbClus = [20]
    buildData = False
if "Dota" in folder:
    # 0 = characters team 1, 1 = characters team 2  ;  o = victory/defeat
    features = [0, 1]
    DS = [1, 1]
    nbInterp = [1, 1]
    nbClus = [10, 10]
    buildData = True
if "Imdb" in folder:
    # 0 = movie, 1 = user, 2 = director, 3 = cast  ;  o = rating
    features = [2, 3]
    DS = [1, 2]
    nbInterp = [1, 1]
    nbClus = [10, 10]
    buildData = False
if "Drugs" in folder:
    # 0 = drugs, 1 = age, 2 = gender, 3 = education  ;  o = attitude (NotSensationSeeking, Introvert, Closed, Calm, Unpleasant, Unconcious, NonNeurotics)
    features = [0, 1]
    DS = [3, 1]
    nbInterp = [1, 1]
    nbClus = [4, 4]
    buildData = True
if "MrBanks" in folder:
    # 0 = usr, 1 = situation, 2 = gender, 3 = age, 4=key  ;  o = decision (up/down)
    features = [0, 1, 2, 3]
    DS = [1, 3, 1, 1]
    nbInterp = [1, 3, 1, 1]
    nbClus = [10, 10, 3, 3]
    buildData = False

if False:  # If we want to specify precisely what to do ; UI
    try:
        folder=sys.argv[1]
        features = np.array(sys.argv[2].split(","), dtype=int)
        DS=np.array(sys.argv[3].split(","), dtype=int)
        nbInterp=np.array(sys.argv[4].split(","), dtype=int)
        nbClus=np.array(sys.argv[5].split(","), dtype=int)
        buildData = bool(int(sys.argv[6]))
    except Exception as e:
        print(e)
        pass

else:  # EXPERIMENTAL SETUP
    try:
        folder=sys.argv[1]
        prec = 1e-4  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
        maxCnt = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
        saveToFile = True
        propTrainingSet = 0.9
        nbRuns = 100
        reductionK = False
        lim = -1
        #folder = "Drugs"
        # Features, DS, nbInterp, nbClus, buildData, seuil
        if "pubmed" in folder.lower():
            # 0 = symptoms  ;  o = disease
            list_params = []
            list_params.append(([0], [3], [1], [20], False, 0))
            list_params.append(([0], [3], [2], [20], False, 0))
            list_params.append(([0], [3], [3], [20], False, 0))
        if "spotify" in folder.lower():
            # 0 = artists  ;  o = next artist
            list_params = []
            list_params.append(([0], [3], [1], [20], False, 1))
            list_params.append(([0], [3], [2], [20], False, 1))
            list_params.append(([0], [3], [3], [20], False, 1))
        if "dota" in folder.lower():
            # 0 = characters team 1, 1 = characters team 2  ;  o = victory/defeat
            list_params = []
            list_params.append(([0, 1], [3, 3], [1, 1], [5, 5], False, 0))
            list_params.append(([0, 1], [3, 3], [2, 2], [5, 5], False, 0))
            list_params.append(([0, 1], [3, 3], [3, 3], [5, 5], False, 0))
        if "imdb" in folder.lower():
            # 0 = movie, 1 = user, 2 = director, 3 = cast  ;  o = rating
            nbRuns = 10
            list_params = []
            list_params.append(([0, 1], [1, 1], [1, 1], [10, 10], False, 0))  # Antonia

            #  Attention, le nombre de clusters pour 2 modèles avec le même nombre de permutations doit être différent sinon l'un écrase l'autre (voir codeT pour les sauvegardes)
            list_params.append(([2, 3], [1, 2], [1, 1], [8, 8], False, 0))
            list_params.append(([2, 3], [1, 2], [1, 2], [8, 8], False, 0))

            list_params.append(([1, 3], [1, 2], [1, 1], [10, 8], False, 0))  # Maybe too large
            list_params.append(([1, 3], [1, 2], [1, 2], [10, 8], False, 0))

            list_params.append(([1, 2, 3], [1, 1, 1], [1, 1, 1], [10, 10, 10], False, 0))
        if "drugs" in folder.lower():
            # 0 = drugs, 1 = age, 2 = gender, 3 = education  ;  o = attitude (NotSensationSeeking, Introvert, Closed, Calm, Unpleasant, Unconcious, NonNeurotics)
            list_params = []
            list_params.append(([0], [3], [1], [7], False, 0))
            list_params.append(([0], [3], [2], [7], False, 0))
            list_params.append(([0], [3], [3], [7], False, 0))

            list_params.append(([0, 3], [3, 1], [1, 1], [7, 5], False, 0))
            list_params.append(([0, 3], [3, 1], [2, 1], [7, 5], False, 0))
            list_params.append(([0, 3], [3, 1], [3, 1], [7, 5], False, 0))

            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [1, 1, 1, 1], [7, 3, 3, 5], False, 0))
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [2, 1, 1, 1], [7, 3, 3, 5], False, 0))
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [3, 1, 1, 1], [7, 3, 3, 5], False, 0))
        if "mrbanks" in folder.lower():
            # 0 = usr, 1 = situation, 2 = gender, 3 = age, 4=key  ;  o = decision (up/down)
            list_params = []
            list_params.append(([0, 4], [1, 1], [1, 1], [4, 8], False, 0))  # Complex decision making...

            list_params.append(([0, 1], [1, 3], [1, 1], [5, 5], False, 0))
            list_params.append(([0, 1], [1, 3], [1, 2], [5, 5], False, 0))
            list_params.append(([0, 1], [1, 3], [1, 3], [5, 5], False, 0))

            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 1, 1, 1], [5, 5, 3, 3], False, 0))
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 2, 1, 1], [5, 5, 3, 3], False, 0))
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 3, 1, 1], [5, 5, 3, 3], False, 0))
        if "twitter" in folder.lower():
            # 0 = history tweets ;  o = retweet
            list_params = []
            list_params.append(([0], [3], [1], [10], False, 0))
            list_params.append(([0], [3], [2], [10], False, 0))
            list_params.append(([0], [3], [3], [10], False, 0))

    except Exception as e:
        print(e)
        pass

for features, DS, nbInterp, nbClus, buildData, seuil in list_params:
    runForOneDS(folder, DS, features, nbInterp, nbClus, buildData, seuil, lim, propTrainingSet, prec, nbRuns, maxCnt, reductionK, sparseMatrices, onlyBuildDS=False)



sys.exit(0)



