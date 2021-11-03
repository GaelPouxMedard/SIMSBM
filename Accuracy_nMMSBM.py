import os
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'
import numpy as np
import matplotlib.pyplot as plt
import sparse
from copy import copy as copy
import itertools
from sklearn import metrics
import pickle
import sys
import sktensor

#// region Tools

def normalized(a, dicForm=False):
    newCoords, vals = [], []
    dicMax={}
    dic={}

    nnz = set(zip(*a.nonzero()[:-1]))
    lg=len(nnz)
    for ind, c in enumerate(nnz):
        if ind%(lg//10)==0: print("Normalizing", ind*100./lg, "%")

        if c not in dicMax: dicMax[c] = np.sum(a[c])

        if dicForm:
            dic[c]=a[c].todense()/dicMax[c]
        else:
            for i in range(len(a[c].nonzero())):
                vals.append(dicMax[c])

    if dicForm:
        return dic
    else:
        vals = np.array(a.data / np.array(vals), dtype=float)
        aNew = sparse.COO(a.nonzero(), vals, shape=a.shape)
        return aNew

def dicsToList(tabK, *a):
    nba = len(a)
    lists = [[] for _ in range(nba)]
    for k in tabK:
        for i in range(nba):
            lists[i].append(a[i][k])

    for i in range(nba):
        lists[i]=np.array(lists[i])

    return lists

def rescale(a):
    newCoords, vals, sub = [], [], []
    dicMax={}
    dicMin={}
    for f in zip(*a.nonzero()):
        c = f[:-1]
        if c not in dicMax: dicMax[c] = np.max(a[c])
        if c not in dicMin: dicMin[c] = np.min(a[c])
        vals.append(dicMax[c])
        sub.append(dicMin[c])
    vals, sub = np.array(vals), np.array(sub)
    vals = np.array((a.data-sub) / (vals-sub), dtype=float)
    aNew = sparse.COO(a.nonzero(), vals, shape=a.shape)
    return aNew

#// endregion

#// region Manipulates the data files

def readMatrix(filename):
    try:
        return sparse.load_npz(filename.replace(".txt", ".npz"))
    except:
        try:
            return np.load(filename.replace(".txt", ".npy"))
        except:
            with open(filename, 'r') as outfile:
                dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
                for i in range(len(dims)):
                    dims[i]=int(dims[i])

            new_data = np.loadtxt(filename).reshape(dims)
            return new_data


    #return sparse.csr_matrix(new_data)


def recoverData(folder, DS):

    strT = ""
    for f, interp in enumerate(DS):
        for i in range(interp):
            strT+=str(f)+"-"
    strT = strT[:-1]+"_"
    print(strT)
    alpha_Tr, alpha_Te = readMatrix("Data/" + folder + "/"+strT+"AlphaTr.npz"), readMatrix("Data/" + folder + "/"+strT+"AlphaTe.npz")

    return alpha_Tr, alpha_Te


def recoverParams(folder, nbClus, nbInterp, final = True, run=-1):
    strT = ""
    for f, [interp, clus] in enumerate(zip(nbInterp, nbClus)):
        for i in range(interp):
            strT += str(clus) + "-"
    strT = strT[:-1] + "_"
    if final:
        run=-1
        featToClus = []
        popFeat = []
        with open("Output/" + folder+"/Final/T="+strT+"%.0f_Inter_FeatToClus.txt" %run, encoding="utf-8") as f:
            for line in f:
                feat, clus, pop = line.replace("\n", "").split("\t")
                featToClus.append(int(clus))
                popFeat.append(pop)
        p = readMatrix("Output/" + folder+"/Final/T="+strT+"%.0f_Inter_p.npy" %run)
        thetas = []
        for i in range(len(set(featToClus))):
            theta = readMatrix("Output/" + folder+"/Final/T="+strT+"%.0f_theta_%.0f_Inter_theta.npy" %(run, i))
            thetas.append(theta)
    else:
        featToClus = []
        popFeat = []
        with open("Output/" + folder+"/T="+strT+"%.0f_Inter_FeatToClus.txt" %run, encoding="utf-8") as f:
            for line in f:
                feat, clus, pop = line.replace("\n", "").split("\t")
                featToClus.append(int(clus))
                popFeat.append(pop)
        p = readMatrix("Output/" + folder+"/T="+strT+"%.0f_Inter_p.npy" %run)
        thetas = []
        for i in range(len(set(featToClus))):
            theta = readMatrix("Output/" + folder+"/T="+strT+"%.0f_theta_%.0f_Inter_theta.npy" %(run, i))
            thetas.append(theta)

    return thetas, p, featToClus, popFeat


def saveResults(tabMetricsAll, folder, features, DS, printRes=True, final=False):
    try:
        if final:
            txtFin = "_Final_"
        else:
            txtFin = ""

        if not os.path.exists("Results/" + folder + "/"):
            os.makedirs("Results/" + folder + "/")
        with open("Results/" + folder + f"/_{txtFin}{features}_{DS}_Results.txt", "w+") as f:
            firstPassage = True
            for label in sorted(list(tabMetricsAll.keys()), key=lambda x: "".join(list(reversed(x)))):
                if firstPassage:
                    f.write("\t")
                    for metric in tabMetricsAll[label]:
                        f.write(metric+"\t")
                    f.write("\n")
                    firstPassage = False
                f.write(label+"\t")
                for metric in tabMetricsAll[label]:
                    f.write("%.4f, " % (tabMetricsAll[label][metric]))
                f.write("\n")
                if printRes:
                    print(label + " " + str(tabMetricsAll[label]))
    except Exception as e:
        print(e)
        pass


def loadModel(folder, DS, model="NB"):
    strT = ""
    for f, interp in enumerate(DS):
        for i in range(interp):
            strT += str(f) + "-"
    strT = strT[:-1]
    filename = f"Output/{folder}/" + strT + f"_{model}.sav"

    return pickle.load(open(filename, 'rb'))


def loadMF(folder, DS, nbInterp, model="NMF"):
    strT = ""
    for i in range(len(DS)):
        for _ in range(nbInterp[i]):
            strT += str(DS[i]) + "-"
    strT = strT[:-1]

    filename = f"Output/{folder}/" + strT + f"_{model}_"
    W = np.load(filename+"W.npy")
    H = np.load(filename+"H.npy")
    coordToInt = {}
    with open(filename+"coordToInt.txt", "r") as f:
        for line in f:
            l = line.replace("\n", "").split("\t")
            coordToInt[l[0]] = int(l[1])

    return W, H, coordToInt

def loadTF(folder, DS, nbInterp, model="TF"):
    strT = ""
    for i in range(len(DS)):
        for _ in range(nbInterp[i]):
            strT += str(DS[i]) + "-"
    strT = strT[:-1]

    filename = f"Output/{folder}/" + strT + f"_{model}_"
    modU = np.load(f"{filename}U.npy", allow_pickle=True)
    modCore = np.load(f"{filename}core.npy", allow_pickle=True)

    return modU, modCore

#// endregion

#// region Build probs
def getDataTe(folder, featuresData, DS, lim=1e20):
    folderName = "Data/" + folder

    outToInt = {}
    featToInt = [{} for i in range(len(featuresData))]

    strDS = ""
    for f, interp in enumerate(DS):
        for i in range(interp):
            strDS+=str(f)+"-"
    strDS = strDS[:-1]+"_"
    with open(folderName + "/"+strDS+"IDTe.txt") as f:
        IDsTe = f.read().replace("[", "").replace("]", "").split(", ")
        IDsTe = np.array(IDsTe, dtype=int)

    with open(folderName + "/"+strDS+"outToInt.txt", encoding="utf-8") as f:
        for line in f:
            lab, num = line.replace("\n", "").split("\t")
            num=int(num)
            outToInt[lab]=num

    for i in range(len(featuresData)):
        with open(folderName + "/"+strDS+"featToInt_%.0f.txt" % featuresData[i], "r", encoding="utf-8") as f:
            for line in f:
                lab, num = line.replace("\n", "").split("\t")
                num = int(num)
                featToInt[i][lab] = num

    outcome = {}
    listOuts = set()
    ind=0
    lg=len(IDsTe)
    #lg = open(folderName + "/outcome.txt", "r", encoding="utf-8").read().count("\n")
    with open(folderName + "/outcome.txt", "r", encoding="utf-8") as f:
        j=0
        for line in f:
            num, out = line.replace("\n", "").split("\t")
            num = int(num)
            if num not in IDsTe: continue
            if j%(lg//10)==0: print("Outcomes:", j*100/lg, "%")
            j+=1
            if j==len(IDsTe): break
            out = out.split(" ")
            outcome[num]=[]
            for o in out:
                listOuts.add(o)
                if o not in outToInt:
                    continue
                outcome[num].append(outToInt[o])

            if j>lim: break

    features = []
    listFeatures = []
    for i in range(len(featuresData)):
        features.append({})
        listFeatures.append(set())
        lg=len(IDsTe)
        #lg = open(folderName + "/feature_%.0f.txt" %i, "r", encoding="utf-8").read().count("\n")
        with open(folderName + "/feature_%.0f.txt" % featuresData[i], "r", encoding="utf-8") as f:
            j=0
            for line in f:
                num, feat = line.replace("\n", "").split("\t")
                num = int(num)
                if num not in IDsTe: continue
                if j%(lg//10)==0: print(f"Features {featuresData[i]}:", j*100/lg, "%")
                j+=1
                if j==len(IDsTe): break
                feat = feat.split(" ")
                features[i][num] = []
                for f in feat:
                    listFeatures[i].add(f)
                    if f not in featToInt[i]:
                        continue
                    features[i][num].append(featToInt[i][f])

                if j > lim: break

    return features, outcome, featToInt, outToInt, IDsTe


def getProbs(alpha, thetas, p, featToClus):
    nbFeat = len(featToClus)
    coords = alpha.nonzero()
    vals = []
    p = np.moveaxis(p, -1, 0)
    for f in zip(*coords):
        probs = p[f[-1]]
        for i in range(nbFeat):
            tet = thetas[featToClus[nbFeat - i - 1]][f[nbFeat - i - 1]]  # k
            probs = probs.dot(tet)
        v = probs
        if v == 0: v = 1e-20
        vals.append(v)
    probs = sparse.COO(coords, vals, shape=alpha_Te.shape)
    return probs


def getIndsMod(DS, nbInterp):
    indsMod = []
    ind = 0
    for i in range(len(DS)):
        for j in range(nbInterp[i]):
            indsMod.append(ind+j)
        ind += DS[i]

    return np.array(indsMod)


def getProbTF(k, U, core):
    from TF import tensor_ttv

    ndims = len(k)
    ind = k

    '''
    Ui_list = [U[c][ind[c]] for c in range(ndims)]

    prediction = []  # Predict every output

    Ui_list_temp = Ui_list + [U[-1]]
    pred_Y = tensor_ttv(core, Ui_list_temp)
    prediction.append(pred_Y)

    prediction = np.array(prediction)
    print(prediction.shape)
    '''

    pr = core
    for c in range(ndims):
        pr = np.tensordot(U[c][ind[c]], pr, axes=1)

    pr = U[-1].dot(pr)
    pr /= np.sum(pr)

    return pr


def getElemProb(c, thetas, p, featToClus):
    nbFeat = len(featToClus)

    probs = p
    for i in range(nbFeat):
        tet = thetas[featToClus[i]][c[i]]  # k
        probs = np.tensordot(tet, probs, axes=1)
    v = probs

    return v


def buildArraysProbs(folder, features, DS, alpha, alphaTe, thetasMod, pMod, featToClus, nbInterp):
    features, outcome, featToInt, outToInt, IDsTe = getDataTe(folder, features, DS, lim=1e20)

    inds = getIndsMod(DS, nbInterp)

    toRem, ind = [], 0
    for i in range(len(DS)):
        if DS[i] != nbInterp[i]:
            for t in range(ind, ind+DS[i]-nbInterp[i]):
                toRem.append(t)
        ind += DS[i]
    if len(toRem)!=0:
        alpha_BL_Tr = alpha.sum(toRem)
        alpha_BL_Te = alphaTe.sum(toRem)
    else:
        alpha_BL_Tr = alpha
        alpha_BL_Te = alphaTe

    print("Build BL")
    pBL = alpha_BL_Tr.sum(list(range(len(alpha_BL_Tr.shape)-1))).todense()
    pBL = pBL/sum(pBL)

    print("Build PF")
    pPF = normalized(alpha_BL_Te, dicForm=True)

    modKNN = loadModel(folder, DS, model="KNN")
    modNB = loadModel(folder, DS, model="NB")
    WNMF, HNMF, coordToInt = loadMF(folder, DS, nbInterp, model="NMF")
    modU, modCore = loadTF(folder, DS, nbInterp, model="TF")

    nbOut = alpha_Te.shape[-1]

    lg = len(IDsTe)
    nb=0
    dicTrue, dicProbMod, dicProbBL, dicProbPF, dicProbNMF, dicProbTF, dicProbKNN, dicProbNB, dicProbRand, dicWeights = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for j, id in enumerate(IDsTe):
        if j % (lg//10) == 0: print("Build list probs", j * 100. / lg, "%")
        if j*100./lg>0.2 and False:
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            break

        if id not in outcome: continue

        """
        toProd = []
        for i in range(len(features)):
            for interp in range(DS[i]):
                toProd.append(features[i][id])
        listKeys = list(itertools.product(*toProd))
        """

        toProd = []
        for i in range(len(features)):
            toProd.append(list(itertools.combinations(features[i][id], r=DS[i])))
        #toProd.append(list(itertools.combinations(outcome[id], r=1)))
        listKeys = list(itertools.product(*toProd))
        if listKeys==[]: continue

        ''' Biased ; repeat evaluation several times for one document
        for k in []:
            karray = np.array(k)
            nb+=1
            if k not in dicTrue:
                a=np.zeros((nbOut))
                for o in outcome[id]:
                    a[o] = 1
                dicTrue[k] = a

            if k not in dicProbBL: dicProbBL[k]=pBL
            if k not in dicProbPF:
                try:dicProbPF[k] = pPF[k]
                except:dicProbPF[k] = np.zeros((nbOut))

            if k not in dicProbMod1: dicProbMod1[k]=getElemProb(karray[indsMod1], thetasMod1, pMod1, featToClusMod1)
            if k not in dicProbMod2: dicProbMod2[k]=getElemProb(karray[indsMod2], thetasMod2, pMod2, featToClusMod2)
            if k not in dicProbMod3: dicProbMod3[k]=getElemProb(karray[indsMod3], thetasMod3, pMod3, featToClusMod3)
            if k not in dicProbRand: dicProbRand[k]=np.random.random((nbOut))

            if k not in dicWeights: dicWeights[k]=0
            dicWeights[k]+=1
        '''

        tempProbMod, tempProbBL, tempProbPF, tempProbNMF, tempProbTF, tempProbKNN, tempProbNB, tempProbRand = [], [], [], [], [], [], [], []
        for ktup in listKeys:
            k = sum(ktup, ())
            karray = np.array(k)
            nb+=1

            tempProbBL.append(pBL)
            try:tempProbPF.append(pPF[k])
            except:tempProbPF.append(np.zeros((nbOut)))

            # [inds] important car réduit le DS au modèle considéré
            tempProbMod.append(getElemProb(karray[inds], thetasMod, pMod, featToClus))

            try: tempProbNMF.append(WNMF[coordToInt[str(k)]].dot(HNMF))
            except: tempProbNMF.append(np.zeros((nbOut)))

            tempProbTF.append(getProbTF(karray[inds], modU, modCore))

            tempProbKNN.append(modKNN.predict_proba([karray])[0])
            tempProbNB.append(modNB.predict_proba([karray])[0])

            rnd = np.random.random((nbOut))
            rnd/=np.sum(rnd)
            tempProbRand.append(rnd)


        a = np.zeros((nbOut))
        for o in outcome[id]:
            a[o] = 1

        dicTrue[j] = a
        dicProbMod[j]=np.mean(tempProbMod, axis=0)
        dicProbPF[j]=np.mean(tempProbPF, axis=0)
        dicProbBL[j]=np.mean(tempProbBL, axis=0)
        dicProbNMF[j]=np.mean(tempProbNMF, axis=0)
        dicProbTF[j]=np.mean(tempProbTF, axis=0)
        dicProbKNN[j]=np.mean(tempProbKNN, axis=0)
        dicProbNB[j]=np.mean(tempProbNB, axis=0)
        dicProbRand[j]=np.mean(tempProbRand, axis=0)

        if j not in dicWeights: dicWeights[j]=0
        dicWeights[j]+=1



    tabK = list(dicTrue.keys())
    listTrue, listProbMod, listProbBL, listProbPF, listProbNMF, listProbTF, listProbKNN, listProbNB, listProbRand, listWeights = \
        dicsToList(tabK, dicTrue, dicProbMod, dicProbBL, dicProbPF, dicProbNMF, dicProbTF, dicProbKNN, dicProbNB, dicProbRand, dicWeights)
    print(nb, len(listWeights))
    print("Min coverage error:", np.average(np.sum(listTrue, axis=1), weights=listWeights)-1)
    return listTrue, listProbMod, listProbBL, listProbPF, listProbNMF, listProbTF, listProbKNN, listProbNB, listProbRand, listWeights

#// endregion

#// region Metrics

def scores(listTrue, listProbs, listWeights, label, tabMetricsAll, nbOut):
    listTrue = np.vstack((listTrue, np.ones((nbOut))))  # Pour eviter qu'une classe n'ait aucun ex negatif ; prendre la moyenne weighted si on utilise ca !
    listProbs = np.vstack((listProbs, np.ones((nbOut))))
    listWeights = np.append(listWeights, 1e-10)
    if label not in tabMetricsAll: tabMetricsAll[label]={}
    print(f"Scores {label}")
    tabMetricsAll[label]["F1"], tabMetricsAll[label]["Acc"] = 0, 0
    for thres in np.linspace(0, 1, 101):
        F1 = metrics.f1_score(listTrue, (listProbs>thres).astype(int), average="weighted", sample_weight=listWeights)
        acc = metrics.accuracy_score(listTrue, (listProbs>thres).astype(int), sample_weight=listWeights)
        if F1 > tabMetricsAll[label]["F1"]:
            tabMetricsAll[label]["F1"] = F1
        if acc > tabMetricsAll[label]["Acc"]:
            tabMetricsAll[label]["Acc"] = acc

    k = 1  # Si k=1, sklearn considère les 0 et 1 comme des classes, mais de fait on prédit jamais 0 dans un P@k...
    topk = np.argpartition(listProbs, -k, axis=1)[:, -k:]
    trueTopK = np.array([listTrue[i][topk[i]] for i in range(len(listTrue))])
    probsTopK = np.array([np.ones((len(topk[i]))) for i in range(len(listProbs))])
    if k>=2:
        tabMetricsAll[label][f"P@{k}"] = metrics.precision_score(trueTopK, probsTopK, average="weighted", sample_weight=listWeights)
    else:
        tabMetricsAll[label][f"P@{k}"] = np.average(trueTopK, weights=listWeights, axis=0)[0]

    tabMetricsAll[label]["AUCROC"] = metrics.roc_auc_score(listTrue, listProbs, average="weighted", sample_weight=listWeights)
    tabMetricsAll[label]["AUCPR"] = metrics.average_precision_score(listTrue, listProbs, average="weighted", sample_weight=listWeights)
    tabMetricsAll[label]["RankAvgPrec"] = metrics.label_ranking_average_precision_score(listTrue, listProbs, sample_weight=listWeights)
    c=metrics.coverage_error(listTrue, listProbs, sample_weight=listWeights)
    tabMetricsAll[label]["CovErr"] = c-1
    tabMetricsAll[label]["CovErrNorm"] = (c-1)/nbOut

    print(tabMetricsAll[label])

    return tabMetricsAll

#// endregion


folder = "MrBanks"
features = [0, 1, 2, 3]
DS = [1, 2, 1, 1]
nbClus = [10, 10, 3, 3]
nbInterp = [1, 2, 1, 1]

'''
folder = "Imdb"
features = [2, 3]
DS = [1, 1]
nbClusMod1 = nbClusMod2 = nbClusMod3 = [10, 10]
nbInterpMod1 = [1, 1]
nbInterpMod2 = [1, 1]
nbInterpMod3 = [1, 1]
'''
'''
folder = "Drugs"
features = [0, 1, 2, 3]
DS = [3, 1, 1, 1]
nbClusMod1 = nbClusMod2 = nbClusMod3 = [10, 5, 2, 5]
nbInterpMod1 = [1, 1, 1, 1]
nbInterpMod2 = [1, 1, 1, 1]
nbInterpMod3 = [1, 1, 1, 1]
'''

final = True
redoBL = True
run=0

if False:  # "UI"
    try:
        folder=sys.argv[1]
        features = np.array(sys.argv[2].split(","), dtype=int)
        DS=np.array(sys.argv[3].split(","), dtype=int)
        nbInterp=np.array(sys.argv[4].split(","), dtype=int)
        nbClu=np.array(sys.argv[7].split(","), dtype=int)
        final = int(sys.argv[10])
        redoBL = int(sys.argv[11])
    except Exception as e:
        print(e)
        pass

else:  # Experimental evaluation
    try:
        folder=sys.argv[1]
        #folder="Drugs"
        # Features, DS, nbInterp, nbClus, buildData, seuil
        paramsDS = []
        if "pubmed" in folder.lower():
            # 0 = symptoms  ;  o = disease
            list_params = []
            list_params.append(([0], [3], [1], [20], False, 500))
            list_params.append(([0], [3], [2], [20], False, 500))
            list_params.append(([0], [3], [3], [20], False, 500))
            paramsDS.append(list_params)
        if "spotify" in folder.lower():
            # 0 = artists  ;  o = next artist
            list_params = []
            list_params.append(([0], [3], [1], [20], False, 2))
            list_params.append(([0], [3], [2], [20], False, 2))
            list_params.append(([0], [3], [3], [20], False, 2))
            paramsDS.append(list_params)
        if "dota" in folder.lower():
            # 0 = characters team 1, 1 = characters team 2  ;  o = victory/defeat
            list_params = []
            list_params.append(([0, 1], [3, 3], [1, 1], [5, 5], False, 0))
            list_params.append(([0, 1], [3, 3], [2, 2], [5, 5], False, 0))
            list_params.append(([0, 1], [3, 3], [3, 3], [5, 5], False, 0))
            paramsDS.append(list_params)
        if "imdb" in folder.lower():
            # 0 = movie, 1 = user, 2 = director, 3 = cast  ;  o = rating
            list_params = []
            list_params.append(([0, 1], [1, 1], [1, 1], [10, 10], False, 0))  # Antonia
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([2, 3], [1, 2], [1, 1], [10, 10], False, 0))
            list_params.append(([2, 3], [1, 2], [1, 2], [10, 10], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([1, 2, 3], [1, 1, 1], [1, 1, 1], [10, 10, 10], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([1, 3], [1, 2], [1, 1], [10, 10], False, 0))  # Maybe too large
            list_params.append(([1, 3], [1, 2], [1, 2], [10, 10], False, 0))
            paramsDS.append(list_params)
        if "drugs" in folder.lower():
            # 0 = drugs, 1 = age, 2 = gender, 3 = education  ;  o = attitude (NotSensationSeeking, Introvert, Closed, Calm, Unpleasant, Unconcious, NonNeurotics)
            list_params = []
            list_params.append(([0], [3], [1], [7], False, 0))
            list_params.append(([0], [3], [2], [7], False, 0))
            list_params.append(([0], [3], [3], [7], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [1, 1, 1, 1], [7, 3, 3, 5], False, 0))
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [2, 1, 1, 1], [7, 3, 3, 5], False, 0))
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [3, 1, 1, 1], [7, 3, 3, 5], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 3], [3, 1], [1, 1], [7, 5], False, 0))
            list_params.append(([0, 3], [3, 1], [2, 1], [7, 5], False, 0))
            list_params.append(([0, 3], [3, 1], [3, 1], [7, 5], False, 0))
            paramsDS.append(list_params)
        if "mrbanks" in folder.lower():
            # 0 = usr, 1 = situation, 2 = gender, 3 = age, 4=key  ;  o = decision (up/down)
            list_params = []
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 1, 1, 1], [5, 5, 3, 3], False, 0))
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 2, 1, 1], [5, 5, 3, 3], False, 0))
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 3, 1, 1], [5, 5, 3, 3], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 1], [1, 3], [1, 1], [5, 5], False, 0))
            list_params.append(([0, 1], [1, 3], [1, 2], [5, 5], False, 0))
            list_params.append(([0, 1], [1, 3], [1, 3], [5, 5], False, 0))
            paramsDS.append(list_params)
    except Exception as e:
        print(e)
        pass

print(folder)
for index_params, list_params in enumerate(paramsDS):
    tabMetricsAll = {}
    for features, DS, nbInterp, nbClus, buildData, seuil in list_params:
        print("Compute BL :", redoBL)
        if redoBL:
            import Baselines
            Baselines.run(folder, DS, features, nbClus, nbInterp)

        print("Import params")
        alpha_Tr, alpha_Te = recoverData(folder, DS)
        nbOut = alpha_Tr.shape[-1]

        probsMod = 0.
        thetasMod, pMod, featToClus, popFeat = recoverParams(folder, nbClus, nbInterp, final=final, run=run)


        print("Build probs")
        listTrue, listProbMod, listProbBL, listProbPF, listProbNMF, listProbTF, listProbKNN, listProbNB, listProbRand, listWeights = \
            buildArraysProbs(folder, features, DS, alpha_Tr, alpha_Te, thetasMod, pMod, featToClus, nbInterp)

        print("Compute metrics")
        tabMetricsAll = scores(listTrue, listProbTF, listWeights, f"TF_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbNMF, listWeights, f"NMF_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbKNN, listWeights, f"KNN_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbNB, listWeights, f"NB_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbMod, listWeights, f"nMMSBM_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbBL, listWeights, f"BL_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbPF, listWeights, f"PF_{nbInterp}", tabMetricsAll, nbOut)
        tabMetricsAll = scores(listTrue, listProbRand, listWeights, f"Rand_{nbInterp}", tabMetricsAll, nbOut)
        print("\n\n")
        saveResults(tabMetricsAll, folder, features, DS, printRes=True, final=final)



pause()






