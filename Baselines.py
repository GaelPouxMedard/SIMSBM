import numpy as np
import sparse
import itertools
import pickle

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

def getName(DS, nbInterp, folder, features):
    codeSave = ""
    for i in range(len(features)):
        for _ in range(nbInterp[i]):
            codeSave += str(features[i]) + "-"
    codeSave = codeSave[:-1]

    return "Output/" + folder + "/" + codeSave

def getDataTr(folder, featuresData, DS, lim=1e20):
    folderName = "Data/" + folder

    outToInt = {}
    featToInt = [{} for i in range(len(featuresData))]

    strDS = ""
    for f, interp in enumerate(DS):
        for i in range(interp):
            strDS+=str(featuresData[f])+"-"
    strDS = strDS[:-1]+"_"

    with open(folderName + "/"+strDS+"IDTr.txt") as f:
        IDsTr = f.read().replace("[", "").replace("]", "").split(", ")
        IDsTr = np.array(IDsTr, dtype=int)

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
    lg=len(IDsTr)
    #lg = open(folderName + "/outcome.txt", "r", encoding="utf-8").read().count("\n")
    with open(folderName + "/outcome.txt", "r", encoding="utf-8") as f:
        j=0
        for line in f:
            num, out = line.replace("\n", "").split("\t")
            num = int(num)
            if num not in IDsTr: continue
            if j%(lg//10)==0: print("Outcomes:", j*100/lg, "%")
            j+=1
            if j==len(IDsTr): break
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
        lg=len(IDsTr)
        #lg = open(folderName + "/feature_%.0f.txt" %i, "r", encoding="utf-8").read().count("\n")
        with open(folderName + "/feature_%.0f.txt" % featuresData[i], "r", encoding="utf-8") as f:
            j=0
            for line in f:
                num, feat = line.replace("\n", "").split("\t")
                num = int(num)
                if num not in IDsTr: continue
                if j%(lg//10)==0: print(f"Features {featuresData[i]}:", j*100/lg, "%")
                j+=1
                if j==len(IDsTr): break
                feat = feat.split(" ")
                features[i][num] = []
                for f in feat:
                    listFeatures[i].add(f)
                    if f not in featToInt[i]:
                        continue
                    features[i][num].append(featToInt[i][f])

                if j > lim: break

    return features, outcome, featToInt, outToInt, IDsTr

def getIndsMod(DS, nbInterp):
    indsMod = []
    ind = 0
    for i in range(len(DS)):
        for j in range(nbInterp[i]):
            indsMod.append(ind+j)
        ind += DS[i]

    return np.array(indsMod)

def buildArraysProbs(folder, features, DS, nbInterp):
    features, outcome, featToInt, outToInt, IDsTr = getDataTr(folder, features, DS, lim=1e20)

    nbOut = len(outToInt)

    lg = len(IDsTr)
    nb=0
    inds = getIndsMod(DS, nbInterp)

    X, y = [], []
    for i in range(nbOut):  # Pour pas qu'il reclassifie
        X.append(np.zeros((sum(DS)))[inds]-1)
        y.append(i)

    coords = {}
    for j, id in enumerate(IDsTr):
        if j % (lg//10) == 0: print("Build list probs", j * 100. / lg, "%")
        if j*100./lg>101: break

        if id not in outcome: continue

        toProd = []
        for i in range(len(features)):
            toProd.append(list(itertools.combinations(features[i][id], r=DS[i])))
        #toProd.append(list(itertools.combinations(outcome[id], r=1)))
        listKeys = list(itertools.product(*toProd))
        if listKeys==[]: continue

        for ktup in listKeys:

            k = sum(ktup, ())
            karray = np.array(k)[inds]
            nb+=1

            for o in outcome[id]:
                X.append(karray)
                y.append(o)

                ctot = tuple(list(karray)+[o])
                if ctot not in coords: coords[ctot]=0
                coords[ctot] += 1


    coordsFin, vals = [], []
    coordsToInt, intToCoords = {}, {}
    num = 0
    for k in coords:
        c = tuple(list(k)[:-1])
        o = k[-1]
        if c not in coordsToInt:
            coordsToInt[c] = num
            intToCoords[num] = c
            num += 1

        coordsFin.append((coordsToInt[c], o))
        vals.append(coords[k])
    coordsFin = np.array(coordsFin)

    D = sparse.COO(coordsFin.T, vals, shape=(num, nbOut))

    return X, y, D, coordsToInt

def classifiers(fname, X, y):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression

    for m in ["NB", "KNN"]:

        filename = fname+f"_{m}.sav"
        print(filename)

        if m=="NB":
            model = GaussianNB()
            model.fit(X, y)
        elif m=="LogReg":
            model = LogisticRegression(multi_class="multinomial", max_iter=10000)
            model.fit(X, y)
        elif m=="KNN":
            model = KNeighborsClassifier()
            model.fit(X, y)

        pickle.dump(model, open(filename, 'wb'))

def MF(fname, D, coordsToInt):
    from sklearn.decomposition import NMF
    alphaNorm = normalized(D)
    alphaNorm = alphaNorm.to_scipy_sparse()

    nbComp = len(alphaNorm.shape)

    nmf = NMF(n_components=nbComp, random_state=0)
    W = nmf.fit_transform(alphaNorm)
    H = nmf.components_

    np.save(fname + "_NMF_W.npy", W)
    np.save(fname + "_NMF_H.npy", H)
    with open(fname + "_NMF_coordToInt.txt", "a") as f:
        f.truncate(0)
        for c in coordsToInt:
            f.write(str(c) + "\t" + str(coordsToInt[c]) + "\n")

def TF(DS, folder, nbClus, nbInterp, features):
    from TensorDecomposition import run as runTD

    norm = 0.001
    step = 0.0001
    N = 10000
    #N = 2
    #print("================= REMOVE ME REMOVE ME REMOVE ME f° TF !!! =====================")

    runTD(DS, folder, nbClus, nbInterp, features, norm, step, N)

def run(folder, DS, features, nbClusMod1, nbInterpMod1, do_TF=True):
    fname = getName(DS, nbInterpMod1, folder, features)
    print(fname)

    X, y, D, coordsToInt = buildArraysProbs(folder, features, DS, nbInterpMod1)

    MF(fname, D, coordsToInt)
    classifiers(fname, X, y)
    if do_TF:
        TF(DS, folder, nbClusMod1, nbInterpMod1, features)

'''
folder = "Spotify"
if "Spotify" in folder:
    features = [0]
    DS = [3]
run(folder, DS, features)
'''


