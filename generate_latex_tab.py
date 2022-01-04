import os
import re

metricsMax = "F1 Acc P@1 P@2 P@3 P@5 P@10 AUCROC AUCPR RankAvgPrec"
metricsMin = "CovErr CovErrNorm"
toRem = ["Acc", "CovErr"]
modelExcl = ["Drugs", "Twitter", "Dota"]

with open("tableResLatex.txt", "w+") as o:
    for folder in os.listdir(f"Results"):

        isin = False
        for mod in modelExcl:
            if mod.lower() in folder.lower():
                isin = True
        if isin: continue

        files = os.listdir(f"Results/{folder}")

        for file in files:
            if "Avg" not in file: continue

            o.write("\\begin{table*}\n\t\\centering\n")
            o.write("\t\\begin{tabular}{|l|l|l")
            fstPass = True
            with open(f"Results/{folder}/{file}", "r") as f_avg:
                with open(f"Results/{folder}/{file.replace('Avg', 'Sem')}", "r") as f_sem:
                    firstline = f_avg.readline()
                    firstline_osef = f_sem.readline()
                    labels = firstline.split("\t")[1:]
                    labelsReduced = labels + []
                    for l in toRem:
                        labelsReduced.remove(l)
                    numLabels = len(labelsReduced)
                    if fstPass:
                        fstPass = False
                        for l in labelsReduced:
                            o.write("|S") #[table-format=1.3]
                        o.write("}\n")

                    o.write("\n\t\t\\cline{1-"+str(2+numLabels)+"}\n")

                    numRow = 0
                    dicDS = {}
                    bestResDS = [None]*(len(labels)-1)
                    tabRes = None
                    for line in f_avg:
                        line_sem = f_sem.readline()
                        modelandres = line.split("\t")
                        model_semandsem = line_sem.split("\t")
                        model = modelandres[0]
                        res = modelandres[1:-1]
                        sem = model_semandsem[1:-1]
                        DS = model[model.rfind("_")+1:]
                        if DS not in dicDS:
                            dicDS[DS] = []
                        for i, tuprs in enumerate(zip(res, sem)):
                            r = tuprs[0]
                            s = tuprs[1]
                            if labels[i] in toRem:
                                continue
                            if "PF" in model or "Rand" in model: continue
                            if labels[i] in metricsMax:
                                if bestResDS[i] is None:
                                    bestResDS[i] = r
                                elif round(float(bestResDS[i]), 3)<round(float(r), 3):
                                    bestResDS[i] = r
                            elif labels[i] in metricsMin:
                                if bestResDS[i] is None:
                                    bestResDS[i] = r
                                elif round(float(bestResDS[i]), 3)>round(float(r), 3):
                                    bestResDS[i] = r
                        dicDS[DS].append((model, res, sem))
                        numRow += 1

                    o.write("\t\t& & ")
                    for l in labels[:-1]:
                        if l in toRem:
                            continue
                        o.write("& \\text{" + l + "} ")
                    o.write("\\\\ \n")
                    o.write("\n\t\t\\cline{2-"+str(2+numLabels)+"}\n")
                    o.write("\t\t\\multirow{"+str(numRow-2*len(DS))+"}{*}{\\rotatebox[origin=c]{90}{\\footnotesize \\text{\\textbf{"+folder.replace("_", "-").replace("[", "(").replace("]", ")")+"}}}}\n")
                    for i, DS in enumerate(dicDS):
                        o.write("\n\t\t& \\multirow{"+str(len(dicDS[DS])-2)+"}{*}{\\rotatebox[origin=c]{90}{\\footnotesize \\text{\\textbf{"+DS.replace("_", "-").replace("[", "(").replace("]", ")")+"}}}}\n")

                        tabRes = dicDS[DS]
                        fstPass = True
                        for model, res, sem in tabRes:
                            if "PF" in model or "Rand" in model: continue
                            model = model[:model.rfind("_")]
                            if not fstPass:
                                o.write(f"\t\t& & {model.replace('_','-').replace('[', '(').replace(']', ')')} ")
                            else:
                                fstPass = False
                                o.write(f"\t\t& {model.replace('_', '-').replace('[', '(').replace(']', ')')} ")

                            for res_i, tuprs in enumerate(zip(res, sem)):
                                r = tuprs[0]
                                s = tuprs[1]
                                if labels[res_i] in toRem:
                                    continue
                                o.write(f"& ")
                                if bestResDS[res_i]==r:
                                    o.write("\\maxf{")

                                o.write(" \\num{ ")
                                o.write(f"{r}")
                                if s != "nan":
                                    o.write(f" +- {s}")
                                o.write(" } ")

                                if bestResDS[res_i]==r:
                                    o.write("} ")
                            o.write("\\\\ \n")

                        if i!=len(dicDS)-1:
                            o.write("\n\t\t\\cline{2-"+str(2+numLabels)+"}\n")
                        else:
                            o.write("\n\t\t\\cline{1-"+str(2+numLabels)+"}\n")


                o.write("\t\\end{tabular}")
                o.write("\n\\end{table*}\n\n\n\n")