import os
import re

metricsMax = "F1 Acc P@1 P@2 P@3 P@5 P@10 AUCROC AUCPR RankAvgPrec"
metricsMin = "CovErr CovErrNorm"

with open("tableResLatex.txt", "w+") as o:
    o.write("\\begin{document}\n\n")
    for folder in os.listdir(f"Results"):
        files = os.listdir(f"Results/{folder}")

        for file in files:
            o.write("\\begin{table}\n\t\\centering\n")
            o.write("\t\\begin{tabular}{|l|l|l")
            fstPass = True
            with open(f"Results/{folder}/{file}", "r") as f:
                firstline = f.readline()
                labels = firstline.split("\t")[1:]
                numLabels = len(labels)
                if fstPass:
                    fstPass = False
                    for l in labels:
                        o.write("|S") #[table-format=1.3]
                    o.write("}\n")

                o.write("\n\t\t\\cline{1-"+str(2+numLabels)+"}\n")

                numRow = 0
                dicDS = {}
                tabRes = None
                bestResDS = {}
                for line in f:
                    model, res = line.split("\t")
                    DS = model[model.rfind("_")+1:]
                    if DS not in dicDS:
                        dicDS[DS] = []
                        bestResDS[DS] = [None]*(len(labels)-1)
                    res = res.split(", ")[:-1]

                    for i, r in enumerate(res):
                        if "PF" in model: continue
                        if labels[i] in metricsMax:
                            if bestResDS[DS][i] is None:
                                bestResDS[DS][i] = r
                            elif bestResDS[DS][i]<r:
                                bestResDS[DS][i] = r
                        elif labels[i] in metricsMin:
                            if bestResDS[DS][i] is None:
                                bestResDS[DS][i] = r
                            elif bestResDS[DS][i]>r:
                                bestResDS[DS][i] = r
                    dicDS[DS].append((model, res))
                    numRow += 1

                o.write("\t\t& & ")
                for l in labels[:-1]:
                    o.write("& \\text{" + l + "} ")
                o.write("\\\\ \n")
                o.write("\n\t\t\\cline{2-"+str(2+numLabels)+"}\n")
                o.write("\t\t\\multirow{"+str(numRow)+"}{*}{\\rotatebox[origin=c]{90}{\\footnotesize \\text{\\textbf{"+folder.replace("_", "-").replace("[", "(").replace("]", ")")+"}}}}\n")
                for i, DS in enumerate(dicDS):
                    o.write("\n\t\t& \\multirow{"+str(len(dicDS[DS]))+"}{*}{\\rotatebox[origin=c]{90}{\\footnotesize \\text{\\textbf{"+DS.replace("_", "-").replace("[", "(").replace("]", ")")+"}}}}\n")

                    tabRes = dicDS[DS]
                    fstPass = True
                    for model, res in tabRes:
                        model = model[:model.rfind("_")]
                        if not fstPass:
                            o.write(f"\t\t& & {model.replace('_','-').replace('[', '(').replace(']', ')')} ")
                        else:
                            fstPass = False
                            o.write(f"\t\t& {model.replace('_', '-').replace('[', '(').replace(']', ')')} ")

                        for res_i, r in enumerate(res):
                            o.write(f"& ")
                            if bestResDS[DS][res_i]==r:
                                o.write("\\maxf{")
                            o.write(f" {r} ")
                            if bestResDS[DS][res_i]==r:
                                o.write("} ")
                        o.write("\\\\ \n")

                    if i!=len(dicDS)-1:
                        o.write("\n\t\t\\cline{2-"+str(2+numLabels)+"}\n")
                    else:
                        o.write("\n\t\t\\cline{1-"+str(2+numLabels)+"}\n")


            o.write("\t\\end{tabular}")
            o.write("\n\\end{table}\n\n\n\n")
    o.write("\\end{document}\n\n")