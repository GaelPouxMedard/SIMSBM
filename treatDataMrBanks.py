

with open("Data/MrBanks/outcome.txt", "a", encoding="utf-8") as o:
    o.truncate(0)
    with open("Data/MrBanks/feature_0.txt", "a", encoding="utf-8") as f1:
        f1.truncate(0)
        with open("Data/MrBanks/feature_1.txt", "a", encoding="utf-8") as f2:
            f2.truncate(0)
            with open("Data/MrBanks/feature_2.txt", "a", encoding="utf-8") as f3:
                f3.truncate(0)
                with open("Data/MrBanks/feature_3.txt", "a", encoding="utf-8") as f4:
                    f4.truncate(0)
                    with open("Data/MrBanks/feature_4.txt", "a", encoding="utf-8") as f5:
                        f5.truncate(0)

                        with open("Data/MrBanks/Raw/Data.csv", "r") as f:
                            prevUsr = None
                            prevExperiment = None
                            prevMarket = None
                            prevRes = None
                            prevExp = None
                            prevExpGuess = None
                            keyPrev = None
                            for line in f:
                                id, round, user, decision, market, result, exp, exp_guess, information_consulted, key, gender, experiment, _, _, age = line.replace("\n", "").split("\t")

                                if user==prevUsr and experiment==prevExperiment:
                                    f1.write(id+"\t"+user+"\n")
                                    strSit = id+"\t"
                                    if prevMarket=="1": strSit += "marketUp"
                                    elif prevMarket=="-1": strSit += "marketDw"
                                    if prevRes=='1': strSit += " usrRight"
                                    elif prevRes=='-1': strSit += " usrWrong"
                                    if exp=='0': strSit += " noExp"
                                    elif exp_guess=='1': strSit += " expUp"
                                    elif exp_guess=='-1': strSit += " expDw"
                                    strSit += "\n"
                                    f2.write(strSit)

                                    strMeta = id+"\t"
                                    if gender=='1': strMeta += "F"
                                    elif gender=='0': strMeta += "M"
                                    strMeta+="\n"
                                    f3.write(strMeta)

                                    strAge = id+"\t"
                                    strAge+= age+"\n"
                                    f4.write(strAge)

                                    f5.write(id+"\t"+keyPrev+"\n")

                                    o.write(id+"\t"+decision+"\n")


                                prevUsr=user
                                prevExperiment=experiment
                                prevMarket=market
                                prevRes=result
                                prevExp=exp
                                prevExpGuess=exp_guess
                                key = key[1:5]
                                if key[-2]==0:
                                    key[-1]="-"
                                keyPrev = key


