import os
import re

with open("tableResLatex.txt", "w+") as o:
    o.write("\\begin{document}\n\n")
    for folder in os.listdir(f"Results"):
        files = os.listdir(f"Results/{folder}")

        for file in files:
            o.write("\\begin{table}\n\t\\centering\n")
            o.write("\t\\begin{tabular}{l|l|l")
            fstPass = True
            with open(f"Results/{folder}/{file}", "r") as f:
                firstline = f.readline()
                labels = firstline.split("\t")[1:]
                numLabels = len(labels)
                if fstPass:
                    fstPass = False
                    for l in labels:
                        o.write("|S[table-format=2.2]")
                    o.write("}\n")

                numRow = 0
                dicDS = {}
                tabRes = None
                for line in f:
                    model, res = line.split("\t")
                    DS = model[model.rfind("_")+1:]
                    if DS not in dicDS: dicDS[DS] = []
                    res = res.split(", ")[:-1]
                    dicDS[DS].append((model, res))
                    numRow += 1

                o.write("\n\t\t\\cline{1-"+str(2+numLabels)+"}\n")
                o.write("\t\t\\multirow{"+str(numRow)+"}{*}{\\rotatebox[origin=c]{90}{\\footnotesize \\textbf{"+folder.replace("_", "-").replace("[", "(").replace("]", ")")+"}}}\n")
                for DS in dicDS:
                    o.write("\n\t\t& \\multirow{"+str(len(dicDS[DS]))+"}{*}{\\rotatebox[origin=c]{90}{\\footnotesize \\textbf{"+DS.replace("_", "-").replace("[", "(").replace("]", ")")+"}}}\n")
                    tabRes = dicDS[DS]
                    fstPass = True
                    for model, res in tabRes:
                        model = model[:model.rfind("_")]
                        if not fstPass:
                            o.write(f"\t\t& & {model.replace('_','-').replace('[', '(').replace(']', ')')} ")
                        else:
                            fstPass = False
                            o.write(f"\t\t& {model.replace('_', '-').replace('[', '(').replace(']', ')')} ")

                        for r in res:
                            o.write(f"& {r} ")
                        o.write("\\\\ \n")
                    o.write("\n\t\t\\cline{1-"+str(2+numLabels)+"}\n")


            o.write("\t\\end{tabular}")
            o.write("\n\\end{table}\n\n\n\n")
    o.write("\\end{document}\n\n")