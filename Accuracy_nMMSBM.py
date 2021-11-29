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

        if c not in dicMax: dicMax[c] = np.sum(a[c])+1e-20

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


def recoverData(folder, DS, features):
    strT = ""
    for f, interp in enumerate(DS):
        for i in range(interp):
            strT+=str(features[f])+"-"
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


def loadModel(folder, DS, nbInterp, features, model="NB"):
    strT = ""
    for i in range(len(features)):
        for _ in range(nbInterp[i]):
            strT += str(features[i]) + "-"
    strT = strT[:-1]
    filename = f"Output/{folder}/" + strT + f"_{model}.sav"

    return pickle.load(open(filename, 'rb'))


def loadMF(folder, DS, nbInterp, features, model="NMF"):
    strT = ""
    for i in range(len(features)):
        for _ in range(nbInterp[i]):
            strT += str(features[i]) + "-"
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

def loadTF(folder, DS, nbInterp, features, model="TF"):
    strT = ""
    for i in range(len(features)):
        for _ in range(nbInterp[i]):
            strT += str(features[i]) + "-"
    strT = strT[:-1]

    filename = f"Output/{folder}/" + strT + f"_{model}_"
    print(filename)
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
            strDS+=str(featuresData[f])+"-"
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


def buildArraysProbs(folder, featuresCons, DS, alpha, alphaTe, thetasMod, pMod, featToClus, nbInterp):
    features, outcome, featToInt, outToInt, IDsTe = getDataTe(folder, featuresCons, DS, lim=1e20)

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

    modKNN = loadModel(folder, DS, nbInterp, featuresCons, model="KNN")
    modNB = loadModel(folder, DS, nbInterp, featuresCons, model="NB")
    WNMF, HNMF, coordToInt = loadMF(folder, DS, nbInterp, featuresCons, model="NMF")
    modU, modCore = loadTF(folder, DS, nbInterp, featuresCons, model="TF")

    nbOut = alpha_Te.shape[-1]

    lg = len(IDsTe)
    nb=0
    index_obs = 0
    dicTrue, dicProbMod, dicProbBL, dicProbPF, dicProbNMF, dicProbTF, dicProbKNN, dicProbNB, dicProbRand, dicWeights = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    tempStoreProbKNN, tempStoreProbNB = {}, {}
    for j, id in enumerate(IDsTe):
        if j % (lg//10) == 0: print("Build list probs", j * 100. / lg, f"% ({j}/{lg})")
        if j*100./lg>0.2 and False:
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            break


        if j>20000 and "PubMed" in folder and False:
            print("ATTENTION CA S'EST ARRETE EXPRES (PUBMED) ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES (PUBMED) ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES (PUBMED) ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            print("ATTENTION CA S'EST ARRETE EXPRES ==============================================")
            break

        if id not in outcome: continue

        toProd = []
        for i in range(len(features)):
            toProd.append(list(itertools.combinations(features[i][id], r=DS[i])))
        #toProd.append(list(itertools.combinations(outcome[id], r=1)))
        listKeys = list(itertools.product(*toProd))
        if len(listKeys)==0 or len(listKeys)>500: continue

        a = np.zeros((nbOut))
        for o in outcome[id]:
            a[o] = 1

        tempProbMod, tempProbBL, tempProbPF, tempProbNMF, tempProbTF, tempProbKNN, tempProbNB, tempProbRand = [], [], [], [], [], [], [], []
        for ktup in listKeys:
            k = sum(ktup, ())
            karray = np.array(k)
            nb+=1

            tempProbBL.append(pBL)
            try:
                tempProbPF.append(pPF[tuple(karray[inds])])
            except Exception as e:
                tempProbPF.append(np.zeros((nbOut)));print("PF failure", e)

            # [inds] important car réduit le DS au modèle considéré
            tempProbMod.append(getElemProb(karray[inds], thetasMod, pMod, featToClus))

            try:
                parr = WNMF[coordToInt[str(tuple(karray[inds]))]].dot(HNMF)
                tempProbNMF.append(parr/(sum(parr)+1e-20))
            except Exception as e:
                tempProbNMF.append(np.zeros((nbOut)));print("NMF failure", e)

            tempProbTF.append(getProbTF(karray[inds], modU, modCore))

            if tuple(karray[inds]) not in tempStoreProbKNN: tempStoreProbKNN[tuple(karray[inds])] = modKNN.predict_proba([karray[inds]])[0]
            if tuple(karray[inds]) not in tempStoreProbNB: tempStoreProbNB[tuple(karray[inds])] = modNB.predict_proba([karray[inds]])[0]
            tempProbKNN.append(tempStoreProbKNN[tuple(karray[inds])])
            tempProbNB.append(tempStoreProbNB[tuple(karray[inds])])

            rnd = np.random.random((nbOut))
            rnd/=np.sum(rnd)
            tempProbRand.append(rnd)

            dicTrue[index_obs] = a
            dicProbMod[index_obs]=tempProbMod[-1]
            dicProbPF[index_obs]=tempProbPF[-1]
            dicProbBL[index_obs]=tempProbBL[-1]
            dicProbNMF[index_obs]=tempProbNMF[-1]
            dicProbTF[index_obs]=tempProbTF[-1]
            dicProbKNN[index_obs]=tempProbKNN[-1]
            dicProbNB[index_obs]=tempProbNB[-1]
            dicProbRand[index_obs]=tempProbRand[-1]

            if index_obs not in dicWeights: dicWeights[index_obs]=0
            dicWeights[index_obs]+=1

            index_obs += 1

    tabK = list(dicTrue.keys())
    listTrue, listProbMod, listProbBL, listProbPF, listProbNMF, listProbTF, listProbKNN, listProbNB, listProbRand, listWeights = \
        dicsToList(tabK, dicTrue, dicProbMod, dicProbBL, dicProbPF, dicProbNMF, dicProbTF, dicProbKNN, dicProbNB, dicProbRand, dicWeights)
    #print("Min coverage error:", np.average(np.sum(listTrue, axis=1), weights=listWeights)-1)
    print("#eval entries:", len(listTrue))
    return listTrue, listProbMod, listProbBL, listProbPF, listProbNMF, listProbTF, listProbKNN, listProbNB, listProbRand, listWeights

#// endregion

#// region Metrics

def scores(listTrue, listProbs, listWeights, label, tabMetricsAll, nbOut):
    print(f"Scores {label}")
    listTrue = np.vstack((listTrue, np.ones((nbOut))))  # Pour eviter qu'une classe n'ait aucun ex negatif ; prendre la moyenne weighted si on utilise ca !
    listProbs = np.vstack((listProbs, np.ones((nbOut))))
    nanmask = np.isnan(listProbs)
    if np.any(nanmask):
        print(f"CAREFUL !!!!! {np.sum(nanmask.astype(int))} NANs IN PROBA !!!")
        listProbs[nanmask] = 0
    listWeights = np.append(listWeights, 1e-10)
    if label not in tabMetricsAll: tabMetricsAll[label]={}

    tabMetricsAll[label]["F1"], tabMetricsAll[label]["Acc"] = 0, 0
    for thres in np.linspace(0, 1, 1001):
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
do_TF = True
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
        #folder="Drugs";print("REMOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOVE ME")
        # Features, DS, nbInterp, nbClus, buildData, seuil
        paramsDS = []
        if "pubmed" in folder.lower():
            # 0 = symptoms  ;  o = disease
            list_params = []
            list_params.append(([0], [3], [1], [20], False, 0))
            list_params.append(([0], [3], [2], [20], False, 0))
            list_params.append(([0], [3], [3], [20], False, 0))
            paramsDS.append(list_params)
        if "spotify" in folder.lower():
            # 0 = artists  ;  o = next artist
            do_TF = True
            list_params = []
            list_params.append(([0], [3], [1], [20], False, 1))
            list_params.append(([0], [3], [2], [20], False, 1))
            list_params.append(([0], [3], [3], [20], False, 1))
            paramsDS.append(list_params)
        if "dota" in folder.lower():
            # 0 = characters team 1, 1 = characters team 2  ;  o = victory/defeat
            do_TF = False
            list_params = []
            list_params.append(([0, 1], [3, 3], [1, 1], [5, 5], False, 0))
            list_params.append(([0, 1], [3, 3], [2, 2], [5, 5], False, 0))
            list_params.append(([0, 1], [3, 3], [3, 3], [5, 5], False, 0))
            paramsDS.append(list_params)
        if "imdb" in folder.lower():
            # 0 = movie, 1 = user, 2 = director, 3 = cast  ;  o = rating
            do_TF = False
            list_params = []
            list_params.append(([0, 1], [1, 1], [1, 1], [10, 10], False, 0))  # Antonia
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([2, 3], [1, 2], [1, 1], [8, 8], False, 0))
            list_params.append(([2, 3], [1, 2], [1, 2], [8, 8], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([1, 3], [1, 2], [1, 1], [10, 8], False, 0))  # Maybe too large
            list_params.append(([1, 3], [1, 2], [1, 2], [10, 8], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([1, 2, 3], [1, 1, 1], [1, 1, 1], [10, 10, 10], False, 0))
            paramsDS.append(list_params)
        if "drugs" in folder.lower():
            # 0 = drugs, 1 = age, 2 = gender, 3 = education  ;  o = attitude (NotSensationSeeking, Introvert, Closed, Calm, Unpleasant, Unconcious, NonNeurotics)
            do_TF = False
            list_params = []
            list_params.append(([0], [3], [1], [7], False, 0))
            list_params.append(([0], [3], [2], [7], False, 0))
            list_params.append(([0], [3], [3], [7], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 3], [3, 1], [1, 1], [7, 5], False, 0))
            list_params.append(([0, 3], [3, 1], [2, 1], [7, 5], False, 0))
            list_params.append(([0, 3], [3, 1], [3, 1], [7, 5], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [1, 1, 1, 1], [7, 3, 3, 5], False, 0))
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [2, 1, 1, 1], [7, 3, 3, 5], False, 0))
            list_params.append(([0, 1, 2, 3], [3, 1, 1, 1], [3, 1, 1, 1], [7, 3, 3, 5], False, 0))
            paramsDS.append(list_params)
        if "mrbanks" in folder.lower():
            # 0 = usr, 1 = situation, 2 = gender, 3 = age, 4=key  ;  o = decision (up/down)
            do_TF = False
            list_params = []
            list_params.append(([0, 4], [1, 1], [1, 1], [4, 8], False, 0))  # Complex decision making...
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 1], [1, 3], [1, 1], [5, 5], False, 0))
            list_params.append(([0, 1], [1, 3], [1, 2], [5, 5], False, 0))
            list_params.append(([0, 1], [1, 3], [1, 3], [5, 5], False, 0))
            paramsDS.append(list_params)

            list_params = []
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 1, 1, 1], [5, 5, 3, 3], False, 0))
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 2, 1, 1], [5, 5, 3, 3], False, 0))
            list_params.append(([0, 1, 2, 3], [1, 3, 1, 1], [1, 3, 1, 1], [5, 5, 3, 3], False, 0))
            paramsDS.append(list_params)

        if "twitter" in folder.lower():
            # 0 = history tweets ;  o = retweet
            do_TF = False
            list_params = []
            list_params.append(([0], [3], [1], [10], False, 0))
            list_params.append(([0], [3], [2], [10], False, 0))
            list_params.append(([0], [3], [3], [10], False, 0))
            paramsDS.append(list_params)


    except Exception as e:
        print(e)
        pass

print(folder)
allRes = []
for index_params, list_params in enumerate(paramsDS):
    tabMetricsAll = {}
    for features, DS, nbInterp, nbClus, buildData, seuil in list_params:
        print("Compute BL :", redoBL)
        if redoBL:
            import Baselines
            Baselines.run(folder, DS, features, nbClus, nbInterp, do_TF=do_TF)

        print("Import params")
        alpha_Tr, alpha_Te = recoverData(folder, DS, features)
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

    allRes.append(tabMetricsAll)

    for tabMetricsAll in allRes:
        strRes = ""
        firstPassage = True
        for label in sorted(list(tabMetricsAll.keys()), key=lambda x: "".join(list(reversed(x)))):
            if firstPassage:
                strRes += "\t"
                for metric in tabMetricsAll[label]:
                    strRes += metric+"\t"
                strRes += "\n"
                firstPassage = False
            strRes += label+"\t"
            for metric in tabMetricsAll[label]:
                strRes += "%.4f\t" % (tabMetricsAll[label][metric])
            strRes += "\n"

        print(strRes.expandtabs(20))

pause()






