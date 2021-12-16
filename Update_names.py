folder = "MrBanks"
import numpy as np

import os

os.curdir = "./"

paramsDS = []
if "pubmed" in folder.lower():
    do_TF = False
    # 0 = symptoms  ;  o = disease
    list_params = []
    list_params.append(([0], [3], [1], [20], False, 0))
    list_params.append(([0], [3], [2], [20], False, 0))
    list_params.append(([0], [3], [3], [20], False, 0))
    paramsDS.append(list_params)
if "spotify" in folder.lower():
    # 0 = artists  ;  o = next artist
    do_TF = False
    list_params = []
    list_params.append(([0], [3], [1], [20], False, 1))
    list_params.append(([0], [3], [2], [20], False, 1))
    list_params.append(([0], [3], [3], [20], False, 1))

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

print(folder)
allRes = []
for index_params, list_params in enumerate(paramsDS):
    tabMetricsAll = {}
    for features, DS, nbInterp, nbClus, buildData, seuil in list_params:

        featToClus = []
        for iter, interp in enumerate(nbInterp):
            for i in range(interp):
                featToClus.append(iter)
        featToClus = np.array(featToClus, dtype=int)

        codeTnew = ""
        for i in featToClus:
            codeTnew += f"{features[i]}({nbClus[i]})-"
        codeTnew = codeTnew[:-1] + "_"

        # ===== TO REMOVE
        codeTold = ""
        for i in featToClus:
            codeTold += f"{nbClus[i]}-"
        codeTold = codeTold[:-1] + "_"

        codeT = ""
        for i in featToClus:
            codeT += f"{features[i]}({nbClus[i]})-"
        codeT = codeT[:-1]

        codeTnewBL = ""
        for i in featToClus:
            codeTnewBL += f"{features[i]}({nbClus[i]})-"
        codeTnewBL = codeTnewBL[:-1] + "_"

        # ===== TO REMOVE
        codeToldBL = ""
        for i in range(len(nbInterp)):
            for _ in range(nbInterp[i]):
                codeToldBL += f"{features[i]}-"
        codeToldBL = codeToldBL[:-1] + "_"

        print(codeToldBL)

        files = os.listdir(os.curdir + "Output/" + folder)
        for file in files:
            if codeToldBL in file and codeToldBL == file[:len(codeToldBL)]:
                print(codeToldBL, file)
                os.rename(os.curdir + "Output/" + folder + "/" + file,
                          os.curdir + "Output/" + folder + "/" + file.replace(codeToldBL, codeTnewBL))

            if codeTold in file and "T=" + codeTold == file[:len(codeTold) + 2]:
                pass
                print(codeTold, file)
                os.rename(os.curdir + "Output/" + folder + "/" + file,
                          os.curdir + "Output/" + folder + "/" + file.replace(codeTold, codeTnew))


        files = os.listdir(os.curdir + "Output/" + folder + "/Final")
        for file in files:
            if codeToldBL in file and codeToldBL == file[:len(codeToldBL)]:
                print(codeToldBL, file)
                os.rename(os.curdir + "Output/" + folder + "/Final" + "/" + file,
                          os.curdir + "Output/" + folder + "/Final" + "/" + file.replace(codeToldBL, codeTnewBL))

            if codeTold in file and "T=" + codeTold == file[:len(codeTold) + 2]:
                print(codeTold, file)
                os.rename(os.curdir + "Output/" + folder + "/Final" + "/" + file,
                          os.curdir + "Output/" + folder + "/Final" + "/" + file.replace(codeTold, codeTnew))