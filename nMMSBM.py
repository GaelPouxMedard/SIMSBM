import numpy as np
import random
# from scipy import sparse
import sparse
import datetime
from copy import deepcopy as copy
from scipy.special import binom
import time

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


from memory_profiler import profile
fp = open("memory_profiler_Norm.log", "w+")

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
def likelihood(thetas, p, alpha, featToClus):
    nbFeat = len(featToClus)

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

def likelihood(alpha, divrm):
    return np.sum(alpha*np.log(divrm+1e-20))

def getDivrm(alpha, thetas, p, featToClus):
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

    divrm = sparse.COO(coords, np.array(vals), shape=alpha.shape)

    if not sparseMatrices:
        divrm=divrm.todense()

    return divrm

# Recursive function to build the dict whose keys are nonzero values of observations
def buildDicCoords(f, dic):
    if len(f)==1:
        dic[f[0]]=1
    else:
        if f[0] not in dic: dic[f[0]]={}
        dic[f[0]] = buildDicCoords(f[1:], dic[f[0]])

    return dic

#Recursive function to build the denominator of omega (see main paper)
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
def maximization_p(alpha, featToClus, popFeat, nbClus, theta, pPrev, divrm):
    nbFeat = len(featToClus)

    if sparseMatrices:

        terme1 = alpha / (divrm + 1e-20)  # f1 f2 g r
        for t in range(nbFeat):
            sizeAfterOperation = prod(terme1.shape)*theta[featToClus[nbFeat - t - 1]].shape[1]*8/terme1.shape[-2]
            #print(terme1.shape, sizeAfterOperation < 2e9)
            if sizeAfterOperation < 2e9:  # As soon as we can use non-sparse we do it (1Gb)
                #print("P dense")
                tet = theta[featToClus[nbFeat - t - 1]].T  # K f1
                terme1 = np.dot(tet, terme1)
            else:
                #print("P sparse")
                tet = sparse.COO(theta[featToClus[nbFeat - t - 1]].T)  # K f1
                terme1 = sparse.COO(np.dot(tet, terme1))


        # print(terme1.shape)  # K1 K2 L r
        if prod(terme1.shape)*8 < 2e9 and type(terme1) is not type(np.array([])):
            terme1 = terme1.todense()

        grandDiv = np.sum(pPrev*terme1, -1)
        grandDiv = np.expand_dims(grandDiv, -1)
        p = pPrev*terme1 / (grandDiv+1e-20)

    else:
        divrm = pPrev
        for i in range(nbFeat):
            divrm = theta[featToClus[nbFeat - i - 1]].dot(divrm)

        terme1 = alpha / (divrm+1e-20)  # f1 f2 g r
        for t in range(nbFeat):
            terme1 = theta[featToClus[nbFeat - t - 1]].T.dot(terme1)

        #print(terme1.shape)  # K1 K2 L r

        grandDiv = np.sum(terme1, axis=-1)
        grandDiv = np.expand_dims(grandDiv, -1)
        p = terme1/grandDiv

    return p

# EM steps for theta
def maximization_Theta(alpha, featToClus, popFeat, nbClus, thetaPrev, p, phim, coeffBin, divrm):
    nbFeat = len(featToClus)
    nbLayers = len(nbClus)

    if sparseMatrices:
        terme1 = alpha / (divrm + 1e-20)  # f1 f2 g r

    else:
        divrm = p
        for i in range(nbFeat):
            divrm = thetaPrev[featToClus[nbFeat - i - 1]].dot(divrm)

        terme1 = alpha / (divrm + 1e-20)  # f1 f2 g r

    thetas = copy(thetaPrev)
    for i in range(len(thetas)):
        thetas[i]*=0
    for i in range(nbLayers):
        arrFeat = []
        for feat in range(len(featToClus)):
            if i==featToClus[feat]: arrFeat.append(feat)

        if len(arrFeat) == 0:
            continue

        prob2 = np.moveaxis(p, arrFeat, range(len(arrFeat)))
        for t in range(nbFeat):
            if nbFeat - t - 1 not in arrFeat:
                prob2 = thetaPrev[featToClus[nbFeat - t - 1]].dot(prob2)

        otherArr = None
        for feat2 in range(len(arrFeat)):  # f2 g r K1
            if feat2!=0:
                if otherArr is None:
                    otherArr = thetaPrev[i]
                else:
                    sizeAfterOperation = prod(otherArr.shape)*prod(thetaPrev[i].shape)*8
                    if sizeAfterOperation < 2e9:
                        print("dense")
                        otherArr = np.tensordot(otherArr, thetaPrev[i], axes=0)
                    else:
                        print("sparse")
                        otherArr = sparse.tensordot(otherArr, thetaPrev[i], axes=0)

        if otherArr is not None:
            inds = [i for i in range(len(otherArr.shape)) if i%2==1]

            prob2 = np.tensordot(otherArr, prob2, axes = (inds, -len(arrFeat)+np.array(arrFeat[:-1])-1))

        prob = sparse.moveaxis(prob2, -1, -2)

        ''' Old/Understandable way
        # === 2 ===
        prob = np.moveaxis(p, arrFeat, range(len(arrFeat)))
        for t in range(nbFeat):
            if nbFeat - t - 1 not in arrFeat:
                prob = thetaPrev[featToClus[nbFeat - t - 1]].dot(prob)

        prob = np.moveaxis(prob, -1, -1-len(arrFeat))  # g r K1 K2

        for feat2 in range(len(arrFeat)):  # f2 g r K1
            if feat2!=0:
                #probOld = thetaPrev[i].dot(np.moveaxis(prob, -len(arrFeat)+feat2, -2))
                sizeAfterOperation = prod(prob.shape)*thetaPrev[i].shape[0]*8/prob.shape[-len(arrFeat)+feat2]
                print(prob.shape, sizeAfterOperation < 2e9)
                if sizeAfterOperation < 2e9:
                    print("T dense")
                    prob = np.tensordot(thetaPrev[i], np.moveaxis(prob, -len(arrFeat)+feat2, 0), axes=1)
                else:
                    print("T sparse")
                    if sizeAfterOperation < 2e9:
                        prob = thetaPrev[i].dot(np.moveaxis(prob, -len(arrFeat) + feat2, -2))
                    else:
                        print("Tenosor1")
                        prob = sparse.tensordot(thetaPrev[i], sparse.moveaxis(sparse.COO(prob), -len(arrFeat) + feat2, 0), axes=1)
                        print("Tenosor2")



        print(prob2.shape, prob.shape)
        d = prob2 - prob
        print("==================================", d[d>1e-5])
        '''

        b = sparse.moveaxis(terme1*coeffBin[i], arrFeat, range(len(arrFeat)))  # f1 f2 g r


        temp = sparse.tensordot(b, prob, axes=b.ndim - 1)  # f1 K1

        thetas[i] += temp*len(arrFeat)  # En fait si on le fait n fois pour n interactions on aura tjrs le meme resultat

        thetas[i] /= phim[i][:, None]+1e-20
        thetas[i] *= thetaPrev[i]

    return thetas

# Reduce number of clusters
def distClus(K1, K2):
    return np.sum(abs(K1-K2))/np.sum(K1+K2)

def BIC(thetas, p, alpha, featToClus, divrm):
    k = np.prod(p.shape)
    for i in range(len(thetas)):
        k += np.prod(thetas[i].shape)
    n = np.sum(alpha)
    #L = likelihood(thetas, p, alpha, featToClus)
    L = likelihood(alpha, divrm)

    return k * np.log(n) - 2*L

def reduceSetK(alpha, featToClus, popFeat, nbClus, thetas, p, divrm):
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

    BICOld = BIC(thetas, p, alpha, featToClus, divrm)
    BICNew = BIC(thetasNew, pNew, alpha, featToClus, divrm)

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

    return thetas, p

# Main loop of the EM algorithm, for 1 run
def EMLoop(alpha, featToClus, popFeat, nbOutputs, nbLayers, nbClus, maxCnt, prec, alpha_Te, folder, run, phim, coeffBin, reductionK, dicnnz):
    nbFeat = len(featToClus)
    thetas, p = initVars(featToClus, popFeat, nbOutputs, nbLayers, nbClus)
    maskedProbs = getAllProbs(dicnnz, [], np.moveaxis(p, -1, 0), thetas, featToClus, 0, nbFeat)
    divrm = sparse.COO(alpha.nonzero(), np.array(maskedProbs), shape=alpha_Tr.shape)
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
            L = likelihood(alpha, divrm)
            print("Run "+str(run)+" - Iter "+str(i)+" - L=" + str(L))

            if ((L - prevL) / abs(L)) < prec:
                cnt += i-iPrev
                if reductionK:
                    if cnt > maxCnt//2:
                        nbClusNew, thetasNew, pNew = reduceSetK(alpha, featToClus, popFeat, nbClus, thetas, p, divrm)
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

        #divrm = getDivrm(alpha, thetas, p, featToClus)
        maskedProbs = getAllProbs(dicnnz, [], np.moveaxis(p, -1, 0), thetas, featToClus, 0, nbFeat)
        divrm = sparse.COO(alpha.nonzero(), np.array(maskedProbs), shape=alpha_Tr.shape)

        pNew = maximization_p(alpha, featToClus, popFeat, nbClus, thetas, p, divrm)
        thetasNew = maximization_Theta(alpha, featToClus, popFeat, nbClus, thetas, p, phim, coeffBin, divrm)
        p = pNew
        thetas = thetasNew

        i += 1

    return maxThetas, maxP, maxL, nbClus


#// endregion

def runFit(alpha_Tr, alpha_Te, nbClus, nbInterp, prec, nbRuns, maxCnt, reductionK):
    print(alpha_Tr.shape)
    nbFeat = alpha_Tr.ndim - 1
    nbOutputs = alpha_Tr.shape[-1]
    popFeat = [l for l in alpha_Tr.shape[:-1]]
    nbLayers = len(nbClus)
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

    phim, coeffBin = [], []
    for i in range(nbLayers):
        arrFeat = []
        for feat in range(len(featToClus)):
            if i==featToClus[feat]: arrFeat.append(feat)

        p = popFeat[arrFeat[-1]]
        phim.append(np.zeros((p)))
        coeffBin.append(weightedBinomCoeff(alpha_Tr, arrFeat, nbFeat))
        #coeffBin.append(alpha_Tr**0)

        for feat in range(len(arrFeat)):
            arrSum = list(range(len(featToClus)+1))
            arrSum.remove(arrFeat[feat])
            arrSum = tuple(arrSum)
            if sparseMatrices:
                phim[i] += (alpha_Tr * coeffBin[i]).sum(arrSum).todense()
            else:
                phim[i] += np.sum(alpha_Tr * coeffBin[i].todense(), arrSum)

    maxL = -1e100
    for i in range(nbRuns):
        print("RUN", i)
        theta, p, L, nbClusNew = EMLoop(alpha_Tr, featToClus, popFeat, nbOutputs, nbLayers, nbClus, maxCnt, prec, alpha_Te, folder, i, phim, coeffBin, reductionK, dicnnz)
        #HOL = likelihood(theta, p, alpha_Te, featToClus)
        HOL=0
        if L > maxL:
            maxL = L
            writeToFile_params(folder + "/Final/", theta, p, L, HOL, featToClus, popFeat, nbClus, -1)
            print("######saved####### MAX L =", L)
        print("=============================== END EM ==========================")


# Run the algorithm with command line parameters or not
# TreatData = do you want to redo the entire corpus from raw data
# RetreatEverything = do you want to compute alpha again
# folder = has to do with the project structure

prec = 1e-4  # Stopping threshold : when relative variation of the likelihood over 10 steps is < to prec
maxCnt = 30  # Number of consecutive times the relative variation is lesser than prec for the algorithm to stop
saveToFile = True
propTrainingSet = 0.9
nbRuns = 100
reductionK = True

# Juste pour que ce soit défini
features = [0]  # Which features to consider (see key)
DS = [3]  # Which dataset use (some are already built for interactions)
nbInterp = [2]  # How many interactions consider for each dataset (reduces DS to this number by summing)
nbClus = [20]
buildData = False

seuil=0  # If retreatEverything=True : choose the threshold for the number of apparitions of an nplet.
# If an nplet appears less than "seuil" times, it's not included in the dataset

'''
nbFeat = 2
nbOutputs = 100
nbClus = [5]

featToClus = np.zeros((nbFeat), dtype=int)
popFeat = np.zeros((nbFeat), dtype=int)

featToClus[0] = featToClus[1] = 0
#featToClus[2] = featToClus[3] = 1
popFeat[0] = popFeat[1] = 200
#popFeat[2] = popFeat[3] = 4
nbLayers=len(nbClus)

alpha_Tr, alpha_Te, thetasSynth, pSynth = getSynthData(featToClus, nbOutputs, popFeat, nbClus)
'''


folder = "PubMed"

lim = -1
if "PubMed" in folder:
    features = [0]
    DS = [3]
    nbInterp = [3]
    nbClus = [20]
    buildData = False
    seuil = 500
if "Spotify" in folder:
    features = [0]
    DS = [3]
    nbInterp = [1]
    nbClus = [20]
    buildData = False
if "Dota" in folder:
    features = [0, 1]
    DS = [1, 1]
    nbInterp = [1, 1]
    nbClus = [10, 10]
    buildData = True
if "Imdb" in folder:
    # 0 = movie, 1 = user, 2 = director, 3 = cast
    features = [2, 3]
    DS = [1, 2]
    nbInterp = [1, 1]
    nbClus = [10, 10]
    buildData = False
if "Drugs" in folder:
    # 0 = drugs, 1 = age, 2 = gender, 3 = education
    features = [0, 1]
    DS = [3, 1]
    nbInterp = [1, 1]
    nbClus = [4, 4]
    buildData = True
if "MrBanks" in folder:
    # 0 = usr, 1 = situation, 2 = gender, 3 = age, 4=key
    features = [0, 1, 2, 3]
    DS = [1, 3, 1, 1]
    nbInterp = [1, 3, 1, 1]
    nbClus = [10, 10, 3, 3]
    buildData = False

import sys
try:
    folder=sys.argv[1]
    features = np.array(sys.argv[2].split(","), dtype=int)
    DS=np.array(sys.argv[3].split(","), dtype=int)
    nbInterp=np.array(sys.argv[4].split(","), dtype=int)
    nbClus=np.array(sys.argv[5].split(","), dtype=int)
    buildData = bool(int(sys.argv[6]))
    reductionK = bool(int(sys.argv[7]))
except Exception as e:
    print(e)
    pass

print("RedK", reductionK)
print("Features", features)
print("DS", DS)


if buildData:
    print("Build alphas")
    import BuildAlpha
    alpha_Tr, alpha_Te = BuildAlpha.run(folder, DS, features, propTrainingSet, lim, seuil=seuil)
else:
    print("Get alphas")
    codeSave = ""
    for i in range(len(DS)):
        for j in range(DS[i]):
            codeSave += str(i) + "-"
    codeSave = codeSave[:-1]
    fname = "Data/"+folder+"/"+codeSave
    alpha_Tr, alpha_Te = readMatrix(fname+"_AlphaTr.npz"), readMatrix(fname+"_AlphaTe.npz")

print("Number of different observations (before thres):", len(alpha_Tr.data), alpha_Tr)
mask = alpha_Tr.data>seuil
alpha_Tr = sparse.COO([a[mask] for a in alpha_Tr.coords], alpha_Tr.data[mask])
print("Number of different observations (after thres):", len(alpha_Tr.data), alpha_Tr)



toRem, ind = [], 0
for i in range(len(DS)):
    if DS[i] != nbInterp[i]:
        for t in range(ind, ind+DS[i]-nbInterp[i]):
            toRem.append(t)
    ind += DS[i]
if len(toRem)!=0:
    alpha_Tr = alpha_Tr.sum(toRem)
    alpha_Te = alpha_Te.sum(toRem)

if not sparseMatrices:
    try:
        alpha_Tr=alpha_Tr.todense()
        alpha_Te=alpha_Te.todense()
    except:
        sparseMatrices=True
        print("=============== SWITCH TO SPARSE =================")

runFit(alpha_Tr, alpha_Te, nbClus, nbInterp, prec, nbRuns, maxCnt, reductionK)
pause()



import pprofile
profiler = pprofile.Profile()
with profiler:
    runFit(alpha_Tr, alpha_Te, nbClus, nbInterp, prec, nbRuns, maxCnt, reductionK)
profiler.dump_stats("BenchmarkSparse.txt")
#profiler.print_stats()
pause()










