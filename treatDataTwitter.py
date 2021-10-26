import TP1


preTreat=False

if preTreat:
    print("GRAPH")
    # Get usr keys
    setUsr = set()

    print("DATA")
    twUsr={}
    cntTw={}
    with open("Data/Twitter/Raw/link_status_search_with_ordering_real.csv") as f:
        f.readline()
        i=0
        lim=2859764  # nb lignes
        for l in f:
            if i%100000==0:
                print(i*100/lim, "%")
            qtes=l.replace("\n", "").split(",")
            link=qtes[0]
            time=qtes[3]
            usrID=qtes[10]
            replyTo=qtes[5]

            try:
                twUsr[usrID].append((link, time,"tw"))
            except:
                twUsr[usrID]=[(link, time,"tw")]
            try:
                cntTw[link].add(usrID)
            except:
                cntTw[link]=set(usrID)

            if i>lim:
                pass
            i+=1


    for url in cntTw:
        cntTw[url]=len(cntTw[url])

    keysTwUsr = list(twUsr.keys())
    for u in keysTwUsr:
        i=0
        while i < len(twUsr[u]):
            if cntTw[twUsr[u][i][0]]<50:
                del twUsr[u][i]
                i-=1

            i+=1
        if twUsr[u]==[]:
            del twUsr[u]

    for u in twUsr:
        pass
        # print(u, twUsr[u])
    print(len(cntTw), len(twUsr), len(keysTwUsr))


    print("WRITE GRAPH")
    with open("Data/Twitter/Raw/active_follower_real.sql") as f:
        with open("Data/Twitter/Raw/graph.txt", "a") as graph:
            graph.truncate(0)
            sameProd=True
            dic = {}
            i=0
            nbLiens=0
            for line in f:
                if "insert  into `active_follower_real`(`user_id`,`follower_id`) values" not in line:
                    #print(line.replace("\n", ""))
                    continue

                line = line.replace("\n", "").replace("insert  into `active_follower_real`(`user_id`,`follower_id`) values ", "")
                line = line.split("),(")
                line[0] = line[0].replace("(", "")
                line[-1] = line[-1].replace(");", "")

                fsUsr=line[0].split(",")[0]

                if fsUsr not in twUsr:
                    continue

                for couple in line:
                    usr, follower = couple.split(",")
                    if follower not in twUsr:
                        continue
                    graph.write(str(int(usr))+"\t"+str(int(follower))+"\n")
                    nbLiens+=1

                i+=1


    print("GET GRAPH")
    g=TP1.Graph().genFromFile("Data/Raw/Twitter/graph.txt")  # Syntaxe follower -> users
    print("GRAPH GOT")
    i=0
    lg=len(g.graphDict)

    with open("Data/Twitter/Raw/histUsr.txt", "a") as f: f.truncate(0)
    with open("Data/Twitter/Raw/twUsr.txt", "a") as f: f.truncate(0)
    nbUsr=len(g.graphDict)
    print(nbUsr)
    for u in g.graphDict:
        if u not in twUsr:
            continue

        if i%10==0 or False:
            print(i*100/nbUsr)
        histUsru=[]
        for v in g.graphDict[u]:
            if v in twUsr:
                for w in twUsr[v]:
                    histUsru.append((w[0], w[1], "hi"))

        with open("Data/Twitter/Raw/histUsr.txt", "a") as f: f.truncate(0)
        with open("Data/Twitter/Raw/twUsr.txt", "a") as f: f.truncate(0)

        with open("Data/Twitter/Raw/histUsr.txt", "a") as f:
            f.write(u+"\t")
            for p in histUsru:
                (link, time, twrt) = p
                f.write(time+" "+link+";")
            f.write("\n")
        with open("Data/Twitter/Raw/twUsr.txt", "a") as f:
            f.write(u+"\t")
            for p in twUsr[u]:
                (link, time, twrt) = p
                f.write(time+" "+link+";")
            f.write("\n")


        if i>=nbUsr:
            pass

        i+=1

histSsAds=False
if histSsAds:
    with open("Data/Twitter/Raw/histUsr.txt", "r") as f:
        with open("Data/Twitter/Raw/histUsrSsAds.txt", "a") as fSsAd:
            fSsAd.truncate(0)
            iter=0
            for line in f:
                l = line.split("\t")
                idUsr=l[0]
                tupleUrls = l[1].split(";")
                tupleUrls.pop(-1)
                #print(tupleUrls)
                setURL=set()
                for tuple in tupleUrls:
                    if tuple!="":
                        #print(tuple)
                        time, url = tuple.split(" ")
                        setURL.add(url)
                cntUrl={}
                for u in setURL:
                    cntUrl[u]=line.count(u)

                fSsAd.write(idUsr + "\t")
                for tuple in tupleUrls:
                    if tuple!="":
                        time, url = tuple.split(" ")
                        if cntUrl[url]<30:
                            fSsAd.write(time + " " + url + ";")
                fSsAd.write("\n")


                if iter>10:
                    pass
                    # break
                iter+=1


tweetsSsAds=False
if tweetsSsAds:
    setSpammer = set()
    setCleanUsr=set()
    i=0
    with open("Data/Twitter/Raw/twUsr.txt", "r") as f:
        with open("Data/Twitter/Raw/twUsrSsAds.txt", "a") as fSsAds:
            fSsAds.truncate(0)
            for line in f:
                l=line.replace("\n", "").replace(";;", ";")
                l = l.split("\t")
                tupleUrls=l[1].split(";")
                tupleUrls.pop(-1)
                idUsr=l[0]
                setURL=set()
                #print(line)
                for tuple in tupleUrls:
                    #print(tuple)
                    try:
                        time, url = tuple.split(" ")
                        setURL.add(url)
                    except Exception as e:
                        print(e, tuple)
                        pass
                cntUrl = []
                for u in setURL:
                    cntUrl.append(line.count(u))

                if max(cntUrl)<5:
                    fSsAds.write(idUsr + "\t")
                    for tuple in tupleUrls:
                        try:
                            time, url = tuple.split(" ")
                            fSsAds.write(time + " " + url + ";")
                            #print(line)
                        except Exception as e:
                            print(e, tuple)
                            pass
                    fSsAds.write("\n")



                i+=1
                if i>10000000:
                    pass



def saveData(folder, g, nounsPost):
    while True:
        try:
            listInfs=set()
            with open(folder + "nounsPost.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in nounsPost:
                    f.write(str(u) + "\t")
                    premPass = True
                    for v in nounsPost[u]:
                        if not premPass:
                            f.write(" ")
                        f.write(str(v))
                        listInfs.add(v)
                        premPass = False
                    f.write("\n")

            with open(folder+"graph.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in g.graphDict:
                    for v in g.graphDict[u]:
                        f.write(str(v)+"\t"+str(u)+"\n")


                break

        except Exception as e:
            print("Retrying to write file :", e)
            pass


def getCorpus():
    print("Get tweets")
    twUsr={}
    with open("Data/Twitter/Raw/twUsrSsAds.txt", "r") as fTw:
        for line in fTw:
            idUsr, infs=line.replace("\n", "").split("\t")
            infs = infs.split(";")
            infs.pop(-1)
            for inf in infs:
                time, url = inf.split(" ")
                try:
                    twUsr[idUsr].append((time, url, "tw"))
                except:
                    twUsr[idUsr]=[(time, url, "tw")]

    print("Get hist")
    histUsr={}
    i=0
    with open("Data/Twitter/Raw/histUsrSsAds.txt", "r") as fHist:
        for line in fHist:
            idUsr, infs = line.replace("\n", "").split("\t")

            i+=1
            if i>200000:
                pass
                break

            if idUsr not in twUsr:
                continue
            infs = infs.split(";")
            infs.pop(-1)
            for inf in infs:
                time, url = inf.split(" ")
                try:
                    histUsr[idUsr].append((time, url, "hi"))
                except:
                    histUsr[idUsr]=[(time, url, "hi")]



    return twUsr, histUsr


def getContentPosts(twUsr, histUsr):
    print("Get contentPosts")
    intervals={}
    g={}
    packTweets=False
    indWin=0
    i=0
    lg=len(twUsr)
    ilg=0
    for u in twUsr:
        ilg+=1
        if ilg%10000==0:
            print(ilg*100./lg, "%")
        if u in histUsr:

            seq = twUsr[u]+histUsr[u]
            seq = list(reversed(sorted(seq, key=lambda x: x[0])))

            indSeq=0
            selfEncounter = False
            while indSeq<len(seq):
                s=seq[indSeq]
                if s[2]=="tw":
                    if not packTweets:
                        i += 1
                        try:
                            g[-i].append(i)
                        except:
                            g[-i]=[i]
                    try:
                        intervals[i].append(s[1])
                    except:
                        intervals[i]=[s[1]]
                    packTweets=True
                    selfEncounter = False
                    indWin=0

                else:
                    packTweets=False
                    if i in intervals:
                        if s[1] in intervals[i]:
                            selfEncounter=True

                    if indWin<=2 and -i in g and selfEncounter:
                        try:
                            intervals[-i].append(s[1])
                        except:
                            intervals[-i]=[s[1]]
                        indWin+=1
                    else:
                        pass

                indSeq+=1

    return g, intervals


def epurate(intervals):
    lg=len(intervals)//2
    for i in range(1, lg+1):
        u = intervals[i][0]
        if u not in intervals[-i]:
            del intervals[i]
            del intervals[-i]

    return intervals

def retreat(folder):
    twUsr, histUsr = getCorpus()
    g, intervals = getContentPosts(twUsr, histUsr)
    #intervals=epurate(intervals)

    g=TP1.Graph(g)
    saveData("Data/" + folder + "/", g, intervals)

