import sparse
import numpy as np
import json
import pandas as pd
from ast import literal_eval


tupAge = [(-0.95197, "18-24"), (-0.07854, "22-34"), (0.49788, "35-44"), (1.09449, "45-54"), (1.82213, "55-64"), (2.59171, "65+")]
tupGender = [(0.48246, "Female"), (-0.48246, "Male")]
tupEd = [(-2.43591, "Left school before 16 years"), (-1.73790, "Left school at 16 years"), (-1.43719, "Left school at 17 years"), (-1.22751, "Left school at 18 years"), (-0.61113, "Some college or university, no certificate or degree"), (-0.05921, "Professional certificate/ diploma"), (0.45468, "University degree"), (1.16365, "Masters degree"), (1.98437, "Doctorate degree")]
tupDrugs = [("CL0", "Never Used"), ("CL1", "Used over a Decade Ago"), ("CL2", "Used in Last Decade"), ("CL3", "Used in Last Year"), ("CL4", "Used in Last Month"), ("CL5", "Used in Last Week"), ("CL5", "Used in Last Day")]

def tupToDic(tup):
    dic = {}
    for (k,v) in tup:
        dic[k]=v.replace(" ", "_")
    return dic

def toFloat(*args):
    arr=[]
    for a in args:
        arr.append(float(a))
    return arr

dicAge, dicGender, dicEd, dicDrugs = tupToDic(tupAge), tupToDic(tupGender), tupToDic(tupEd), tupToDic(tupDrugs)
arrSeuilSup = [1.5, 1.6, 1.2, 1.7, 1.7, 1.3, 1.5]
arrSeuilMin = [-1.5, -1.6, -1.2, -1.7, -1.7, -1.3, -1.5]
behavTot, behavTot2 = np.zeros((7)), np.zeros((7))
arrSeuilSup = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
arrSeuilMin = [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5]
arrNames = np.array(["NonNeurotics", "Introvert", "Closed", "Unpleasant", "Unconcious", "Calm", "NotSensationSeeking", "Neurotics", "Extrovert", "Opened", "Pleasant", "Concious", "Impulsive", "SensationSeeking"])
arrNamesDrugs = np.array(["alcohol", "amphet", "amyl", "benzos", "caff", "cannabis", "choc", "coke", "crack", "exta", "heroin", "ketamine", "legalh", "lsd", "meth", "mush", "nicotine", "semer", "vsa"])
with open("Data/Drugs/outcome.txt", "a", encoding="utf-8") as o:
    o.truncate(0)
    with open("Data/Drugs/feature_0.txt", "a", encoding="utf-8") as f1:
        f1.truncate(0)
        with open("Data/Drugs/feature_1.txt", "a", encoding="utf-8") as f2:
            f2.truncate(0)
            with open("Data/Drugs/feature_2.txt", "a", encoding="utf-8") as f3:
                f3.truncate(0)
                with open("Data/Drugs/feature_3.txt", "a", encoding="utf-8") as f4:
                    f4.truncate(0)
                    with open("Data/Drugs/Raw/drug_consumption.data", encoding="utf-8") as f:
                        for line in f:
                            tab = line.replace("\n", "").split(",")
                            ID, age, gender, education, country, ethnicity, neuroticism, extraversion, openness, agreeableness, conscientiousness, impulsivity, sensation = tab[:13]
                            alcohol, amphet, amyl, benzos, caff, cannabis, choc, coke, crack, exta, heroin, ketamine, legalh, lsd, meth, mush, nicotine, semer, vsa = tab[13:]
                            age, gender, education, neuroticism, extraversion, openness, agreeableness, conscientiousness, impulsivity, sensation = toFloat(age, gender, education, neuroticism, extraversion, openness, agreeableness, conscientiousness, impulsivity, sensation)
                            # Features : age, gender, education, drugs
                            # Outcome : neuroticism, extraversion, openness, agreeableness, conscientiousness, impulsivity, sensation

                            arrDrugs = np.array([alcohol, amphet, amyl, benzos, caff, cannabis, choc, coke, crack, exta, heroin, ketamine, legalh, lsd, meth, mush, nicotine, semer, vsa])
                            arrInds = ((arrDrugs=="CL6").astype(int)+(arrDrugs=="CL5").astype(int))#+(arrDrugs=="CL4").astype(int))
                            arrFeatDrugs = arrNamesDrugs[np.where(arrInds==1)]
                            f1.write(str(ID) + "\t")
                            for i, dr in enumerate(arrFeatDrugs):
                                if i!=0: f1.write(" ")
                                f1.write(dr.replace(" ", "_"))
                            f1.write("\n")

                            f2.write(str(ID)+"\t"+dicAge[age]+"\n")
                            f3.write(str(ID) + "\t" + dicGender[gender] + "\n")
                            f4.write(str(ID) + "\t" + dicEd[education] + "\n")

                            arrBehaviour = np.array([neuroticism, extraversion, openness, agreeableness, conscientiousness, impulsivity, sensation])
                            arrBehaviourMin = (arrBehaviour<arrSeuilMin).astype(int)
                            arrBehaviourSup = (arrBehaviour>arrSeuilSup).astype(int)
                            arrInds = arrBehaviourMin+arrBehaviourSup
                            behavTot += arrBehaviourMin
                            behavTot2 += arrBehaviourSup
                            arrOut = arrNames[np.where(arrInds==1)]
                            o.write(str(ID) + "\t")
                            for i, out in enumerate(arrOut):
                                if i!=0: o.write(" ")
                                o.write(out.replace(" ", "_"))
                            o.write("\n")

print(behavTot2)
print(behavTot)