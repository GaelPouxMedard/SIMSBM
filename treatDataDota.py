import numpy as np

''' Game mode (eg tab[2])
0 Unknown
1 All Pick
2 Captain’s Mode
3 Random Draft
4 Single Draft
5 All Random
6 Intro
7 Diretide
8 Reverse Captain’s Mode
9 The Greeviling
10 Tutorial
11 Mid Only
12 Least Played
13 New Player Pool
14 Compendium Matchmaking
15 Custom
16 Captains Draft
17 Balanced Draft
18 Ability Draft
19 Event (?)
20 All Random Death Match
21 Solo Mid 1 vs 1
22 Ranked All Pick
'''

''' Lobby type (eg tab[3])
-1 Invalid
0 Public matchmaking
1 Practice
2 Tournament
3 Tutorial
4 Co-op with AI
5 Team match
6 Solo queue
7 Ranked matchmaking
8 Solo Mid 1 vs 1

'''

sumHeroes = np.zeros((113))
tabGameMode, tabGameLobby, tabWin, tabHeroesTeam, tabHeroesOpp = [], [], [], [], []
with open("Data/Dota/Raw/dota2Train.csv", encoding="utf-8") as f:
    for line in f:
        tabLine = np.array(line.replace("\n", "").split(","), dtype=int)
        win = tabLine[0]
        gameMode = tabLine[2]
        gamePick = tabLine[3]

        heroesTeam = np.where(tabLine[4:]==1)[0]
        heroesOpp = np.where(tabLine[4:]==-1)[0]
        #print(heroesWin)
        #print(heroesLose)
        if gameMode==2 and gamePick==2:
            sumHeroes += np.abs(tabLine[4:])

            tabWin.append(win)
            tabGameMode.append(gameMode)
            tabGameLobby.append(gamePick)
            tabHeroesTeam.append(heroesTeam)
            tabHeroesOpp.append(heroesOpp)

thres = 8000

with open("Data/Dota/outcome.txt", "w+", encoding="utf-8") as o:
    with open("Data/Dota/feature_0.txt", "w+", encoding="utf-8") as f1:
        with open("Data/Dota/feature_1.txt", "w+", encoding="utf-8") as f2:
            for i in range(len(tabWin)):
                o.write(str(i)+"\t"+str(tabWin[i])+"\n")
                f1.write(str(i) + "\t")
                for h in tabHeroesTeam[i]:
                    if sumHeroes[h]<thres: continue
                    f1.write(str(h)+" ")
                f1.write("\n")

                f2.write(str(i) + "\t")
                for h in tabHeroesOpp[i]:
                    if sumHeroes[h]<thres: continue
                    f2.write(str(h)+" ")
                f2.write("\n")








