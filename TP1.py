import random
import copy
#from scipy import sparse


import networkx as nx



from bisect import bisect
def choices(values, weights, k):
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    return [values[bisect(cum_weights, random.random() * total)] for i in range(k)]

def moment(arr, n=1):
    if len(arr)==0:
        return 0.

    if n == 1:
        return sum(arr) / len(arr)  # Faster

    mom = 0.
    for val in arr:
        mom += val ** n
    return mom / len(arr)

def var(arr):
    arrmean = moment(arr, 1)
    var = 0.
    for val in arr:
        var += (val - arrmean) ** 2
    return var / len(arr)

def ReadData(filePath):
    g={}
    f=open(filePath,'r')
    for line in f.readlines():
        line=line.replace("\n", "") # Read the text line by line
        tabLine=line.split('\t')
        a,b=int(tabLine[0]),int(tabLine[1])

        try:
           g[a].append(b) # If the node exists, add it a neighbour
        except:
           g[a]=[] # If the node doesn't exists, create it and then add its neighbour
           g[a].append(b)

        try:
           g[b].append(a) # Undirected graph so neighbours go both ways (a->b and b->a)
        except:
           g[b]=[]
           g[b].append(a)

    f.close()
    return g

def ReadAttributes(filePath):
    pos={}
    att={}
    f=open(filePath,'r')
    for line in f.readlines():

        line=line.replace("\n", "") # Read the text line by line
        tabLine=[]
        tabLine=line.split('\t')
        a=int(tabLine[0])

        posx,posy=None, None

        if len(tabLine)>=5:
            posx=tabLine[2]
            posy=tabLine[4]

        pos[a]=[posx,posy]
        att[a]={}

        for i in range(5,len(tabLine),2):
            att[a][tabLine[i]]=tabLine[i+1]


    f.close()

    return pos, att


class Graph(object):
    def __init__(self, graphDict=None, directed=False):
        if graphDict:
            self.graphDict = {}
            for u in graphDict:
                self.graphDict[u]=[]
            for u in graphDict:
                for v in graphDict[u]:
                    self.graphDict[u].append(v)

            self.N = len(self.graphDict)

            self.L = len(self.edges())//2

        else:
            self.graphDict = dict()

        self.P, self.SP = None, None

    def vertices(self):
        return list(self.graphDict.keys())

    def edges(self):
        edges = []
        for i in self.graphDict:
            for j in self.graphDict[i]:
                edges.append([i, j])

        return edges

    def adjacencyMatrix(self, weighted=False, weights=None):
        N = self.N

        row_ind, col_ind, data = [], [], []


        if not weighted:
            for u in self.vertices():
                for v in self.graphDict[u]:
                    row_ind.append(u)
                    col_ind.append(v)
                    data.append(1.)


        else:
            if weights is not None:
                for u in self.vertices():
                    for v in self.graphDict[u]:
                        row_ind.append(u)
                        col_ind.append(v)
                        data.append(weights[u,v])

            else:
                for u in self.vertices():
                    for v in self.graphDict[u]:
                        row_ind.append(u)
                        col_ind.append(v)
                        data.append(random.random())

        return sparse.coo_matrix((data, (row_ind, col_ind))).tocsr()

    def addVertex(self,vertex):
        self.graphDict[vertex]=[]
        self.nodes[vertex]=Node(label=vertex)

        self.N+=1

    def addEdge(self,edge): # Link from edge[0] to edge[1]
        self.graphDict[edge[0]].append(edge[1])
        self.nodes[edge[0]].neigh.append(self.nodes[edge[1]])

        self.graphDict[edge[1]].append(edge[0])
        self.nodes[edge[1]].neigh.append(self.nodes[edge[0]])

        self.L+=1

    def removeEdge(self, edge):
        self.graphDict[edge[0]].remove(edge[1])
        self.nodes[edge[0]].neigh.remove(self.nodes[edge[1]])

        self.graphDict[edge[1]].remove(edge[0])
        self.nodes[edge[1]].neigh.remove(self.nodes[edge[0]])

        self.L-=1

    def swapEdges(self, edge1, edge2):
        e1=[edge1[1], edge2[0]]
        e2=[edge1[0], edge2[1]]

        self.addEdge(e1)
        self.addEdge(e2)

        self.removeEdge(edge1)
        self.removeEdge(edge2)

    def verticesDegrees(self):
        degrees={}

        for u in self.vertices():
            degrees[u]=len(self.graphDict[u])

        return degrees

    def findIsolatedVertices(self):
        isoVert=[]
        deg=self.verticesDegrees()
        for u in self.vertices():
            if deg[u]==0:
                isoVert.append(u)

        return isoVert

    def removeIsolatedVertices(self):
        vert=self.findIsolatedVertices()
        for u in vert:
            del self.graphDict[u]
            del self.nodes[u]
            self.N-=1

    def density(self):
        m=0
        for i in self.vertices():
            m += self.verticesDegrees()[i]
        m/=2 # Edges are counted twice

        N=len(self.vertices())

        return 2*m/(N*(N-1)) # Number of edges over the maximum possible number of edges

    def degreeSequence(self):
        degSeq=[]
        vertDeg=self.verticesDegrees()

        for u in sorted(self.verticesDegrees()): # Get the nodes degree
            degSeq.append(vertDeg[u])

        return degSeq

    def degreeDist(self):
        Pk=[0.]*self.N
        for k in self.degreeSequence():
            Pk[k]+=1

        i=len(Pk)-1
        while i>=0:
            Pk[i] /= self.N
            if Pk[i]==0:
                del(Pk[i])
            i-=1

        return range(len(Pk)), Pk

    def excessDegreeDist(self):
        kmax = max(self.degreeSequence())
        nk = [0.] * (kmax+1)
        knn = [0.] * (kmax+1)
        for u in self.graphDict:
            ku = len(self.graphDict[u])
            ktemp = 0.
            for v in self.graphDict[u]:
                ktemp += float(len(self.graphDict[v])-1)
            if ku!= 0:
                knn[ku] += ktemp / ku
            nk[ku] += 1

        for i in range(len(knn)):
            if nk[i] != 0:
                knn[i] /= nk[i]

        return knn

    def neighboursDegreeDist(self):
        kmax = max(self.degreeSequence())
        nk = [0.] * (kmax+1)
        knn = [0.] * (kmax+1)
        for u in self.graphDict:
            ku = len(self.graphDict[u])
            ktemp = 0.
            for v in self.graphDict[u]:
                ktemp += float(len(self.graphDict[v]))
            if ku!= 0:
                knn[ku] += ktemp / ku
            nk[ku] += 1

        tabk=[]
        tabknn=[]
        for i in range(len(knn)):
            if nk[i] != 0:
                tabknn.append(knn[i] / nk[i])
                tabk.append(i)


        return tabk, tabknn

    def erdosGallai(self, degSeq=[]):

        if degSeq==[]:
            degSeq=self.degreeSequence()
        n=len(degSeq)
        termLeft=0
        termRight=0
        isEven=0
        fulfilled=True

        for i in degSeq:
            isEven+=i

        if isEven%2!=0: #Sequence has to be even
            fulfilled=False

        for k in range(n):
            if n%2!=0: #Do not enter the loop if n is not even
                break

            for i in range(k):
                termLeft+=degSeq[i]

            termRight+=k*(k-1)
            for i in range(k+1,n):
                termRight+=min(degSeq[i],k)

            if termLeft>termRight: #If condition not fulfilled for any k, break the loop
                fulfilled=False
                break

        return fulfilled

    def globalClusteringCoeff(self):
        triangles=0 #Here triangle means connected triplets
        triplets=0

        for u in self.graphDict:
            for eu1 in self.graphDict[u]:
                for eu2 in self.graphDict[u]:
                    if eu1!=eu2: # Triplet
                        if eu1 in self.graphDict[eu2]: # Closed triplet. Undirected, so it's equivalent to eu2 in graph(eu1)
                            triangles+=1

                        triplets+=1

        triangles/=2 # Because they are counted twice (ABC and CBA are the same triplets)
        triplets/=2 # Same

        return triangles/triplets

    def localClusteringCoeff(self, u):
        degu = self.verticesDegrees()[u]
        triangles=0
        triplets=0

        if degu==0:
            return 0

        for eu1 in self.graphDict[u]:
            for eu2 in self.graphDict[u]:
                if eu1 != eu2:  # Triplet
                    if eu1 in self.graphDict[eu2]: # Closed triplet. Undirected, so it's equivalent to eu2 in graph(eu1)
                        triangles += 1

                    triplets += 1

        if triangles==0:
            return 0

        return triangles/triplets

    def connectedComponents(self):
        g=self.graphDict
        listVert=self.vertices()

        connComp=[]

        while listVert!=[]: #Continue until all the nodes have been analyzed
            n=listVert[int(random.random() * len(listVert))] #Random beginning vertex
            vertConnComp = [n]
            listVert.remove(n)
            for u in vertConnComp:
                for v in g[u]:
                    if v in listVert:
                        vertConnComp.append(v) #Append the neighbour to the pile
                        listVert.remove(v) #Remove the nodes that has been examined

            connComp.append(vertConnComp) # All nodes in a connected component examined when vertConnComp=[] (no more new neighbours)

        return connComp

    def shortestPath(self, computePaths=False):
        if computePaths:
            self.SP, self.P = self.shortestPathHomemade(computePaths=True)
            return self.SP, self.P
        else:
            self.SP=dict(nx.algorithms.shortest_path_length(self.toNx()))
            return self.SP

    def shortestPathHomemade(self, computePaths=False):
        N=self.N
        SP={}
        P={}
        for u in self.vertices():  # Initialization
            SP[u]={}
            P[u]={}
            for v in self.vertices():
                SP[u][v]=N**2
                P[u][v]=[[]]
                if v in self.graphDict[u]:
                    SP[u][v]=1
                    P[u][v]=[[u, v]]
                if u==v:
                    SP[u][v]=0

        if computePaths:
            for k in self.vertices(): # Computation
                for u in self.vertices():
                    for v in self.vertices():
                        if k!=u and k!=v:
                            if SP[u][v] > SP[u][k] + SP[k][v]:
                                SP[u][v] = SP[u][k] + SP[k][v]
                                P[u][v] = self.joinPaths(P[u][k], P[k][v])
                            elif SP[u][v] == SP[u][k] + SP[k][v]:
                                newPaths=self.joinPaths(P[u][k], P[k][v])
                                for p in newPaths:  # Avoid adding identical path
                                    if p not in P[u][v]:
                                        P[u][v] = P[u][v] + [p]
            self.P = P
        else:
            for k in self.vertices(): # Computation
                for u in self.vertices():
                    for v in self.vertices():
                        if k!=u and k!=v:
                            if SP[u][v] > SP[u][k] + SP[k][v]:
                                SP[u][v] = SP[u][k] + SP[k][v]

        for u in self.vertices():  # Cleaning
            for v in self.vertices():
                if SP[u][v]==N**2:
                    SP[u][v]=-1

        self.SP = SP

        if computePaths:
            return SP, P
        else:
            return SP

    def joinPaths(self, t1, t2, jonctionEgale=True):
        tab=[]
        for one in range(len(t1)):
            for two in range(len(t2)):
                if jonctionEgale:
                    tab.append(t1[one] + t2[two][1:])
                else:
                    tab.append(t1[one] + t2[two])
        return tab

    def betweenessCentrality(self):
        return nx.closeness_centrality(self.toNx())

    def harmonicCentrality(self):
        if self.SP is None:
            self.shortestPath(computePaths=False)
        har = []
        for u in self.SP:
            s=0.
            for v in self.SP[u]:
                if v!=u and self.SP[u][v]!=-1:
                    s+=1./self.SP[u][v]
            har.append(s)
        return har

    def closenessCentrality(self):
        if self.SP is None:
            self.shortestPath(computePaths=False)
        cls = []
        for u in self.SP:
            s=0.
            div=0
            for v in self.SP[u]:
                if self.SP[u][v]!=-1:
                    s+=self.SP[u][v]
                    div+=1
            cls.append(float(div-1)/s)
        return cls

    def floodingTime(self):
        if self.SP is None:
            self.shortestPath(computePaths=False)

        fld=[]
        for u in self.SP:
            fld.append(max(self.SP[u].values()))
        maxFld=max(fld)
        for u in range(len(fld)):
            fld[u]/=maxFld
        return fld

    def eigenvectorCentrality(self):
        A=self.adjacencyMatrix()
        lamb, eigCent = sparse.linalg.eigs(A.tocsr(), k=1, which='LM')
        return eigCent

    def averageShortestPath(self):
        SP=self.shortestPath(computePaths=False)
        N=len(self.graphDict)
        avgSP=0

        isDisjoint=False

        for i in SP:
            for j in SP[i]:
                if SP[i][j]==-1:
                    isDisjoint=True
                elif i!=j:
                    avgSP+=SP[i][j]
        return avgSP/(N*(N-1))

    def diameter(self):
        SP, P=self.shortestPath(computePaths=False)
        diam=0
        for u in SP:
            for v in SP[u]:
                if SP[u][v]>diam:
                    diam=SP[u][v] # Diameter is the longest shortest path
        return diam

    def biggestComponentDiameter(self): #Biggest component is the one having the biggest diameter here
        components=self.connectedComponents()
        diamComp=[]

        for comp in components:
            g={} # Consider each connected component as a subgraph
            for u in comp:
                g[u]=self.graphDict[u]
            g=Graph(g)

            diamComp.append(g.diameter()) # Append the diameters of the sub graphs

        return max(diamComp) # Return the biggest diameter

    def spanningTree(self, source_node=None):
        if len(self.connectedComponents())>1:
            print("Graph not connected")
            return -1
        vert=self.vertices()
        N=self.N
        dic={}

        if source_node is None:
            source_node = random.choice(vert)

        cost, pred = {}, {}
        Q=[]
        for u in vert:
            dic[u]=[]
            cost[u]=N**2
            pred[u]=None
            if u==source_node:
                cost[u] = 0
            Q.append((u, cost[u]))

        Q=BinaryHeap(Q)

        while len(Q)!=0:
            u = Q.delete_min()[0]
            for v in self.graphDict[u]:
                if cost[v] > cost[u]:
                    pred[v]=u
                    cost[v]=cost[u]
                    Q.delete(v)
                    Q.insert((v, cost[v]))

        for u in pred:
            if u != source_node:
                dic[u].append(pred[u])
                dic[pred[u]].append(u)

        return Graph(dic)

    def toNx(self):
        g=self
        graphDraw = nx.Graph()
        for u in g.vertices():
            for v in g.graphDict[u]:
                graphDraw.add_edge(u, v)

        return graphDraw

    def plotGraph(self, circular=False, pos=None):
        import matplotlib.pyplot as plt
        graphDraw=self.toNx()

        plt.subplot(111)
        if circular:
            nx.draw_circular(graphDraw, with_labels=True)#, pos=nx.drawing.circular_layout(self.graphDict))
        elif pos is not None:
            nx.draw(graphDraw, with_labels=True, pos=pos)
        else:
            nx.draw(graphDraw, with_labels=True)
        plt.show()

        return graphDraw

    def write(self, filename="Graph.txt"):
        txt=""
        for u in self.vertices():
            for v in self.graphDict[u]:
                txt+=str(u)+"\t"+str(v)+"\n"

        f=open(filename, "w+")
        f.write(txt)
        f.close()

    def genFromFile(self, filePath, directed=False):
        g = {}
        f = open(filePath, 'r')

        for line in f.readlines():
            tabLine = line.replace("\n", "").split('\t')
            a, b = tabLine[0], tabLine[1]

            if not directed:
                try:
                    if b not in g[a]:  # Forces a simple graph
                        g[a].append(b)  # If the node exists, add it a neighbour
                except:
                    g[a] = [b]  # If the node doesn't exists, create it and then add its neighbour

            try:
                if a not in g[b]:
                    g[b].append(a)  # Undirected graph so neighbours go both ways (a->b and b->a)
            except:
                g[b] = [a]

        f.close()
        return Graph(g)

    def genER_np(self, n, p, allowSelfLoops=False):
        graphDict = {}

        for i in range(n):  # Initializes the dictionary
            graphDict[i] = []

        for i in range(n):
            for j in range(i, n):
                if random.random() < p and i != j:  # Avoid loops : i!=j
                    graphDict[i].append(j)
                    graphDict[j].append(i)
                elif random.random() < p and i == j and allowSelfLoops:
                    graphDict[i].append(j)

        return Graph(graphDict)

    def genER_nm(self, n, m):
        graphDict = {}

        for i in range(n):  # Initializes the dictionary
            graphDict[i] = []

        for i in range(m):
            u = round(random.random() * (n - 1))
            v = round(random.random() * (n - 1))

            if not (u == v or v in graphDict[u]):  # Avoid loops (while u==v) and multiple edges (if v, u are already neighbours)
                graphDict[u].append(v)
                graphDict[v].append(u)

        return Graph(graphDict)

    def genER_np_assortative(self, n, p, rho_lim):
        ## REF : https://arxiv.org/pdf/0705.4503.pdf

        g=self.genER_np(n, p)
        g.removeIsolatedVertices()
        N=g.N
        deg=g.verticesDegrees()
        vert=g.vertices()

        n1 = N * moment(g.degreeSequence(), 1)
        n2 = N * moment(g.degreeSequence(), 2)
        n3 = 0.
        for u in g.vertices():
            for v in g.graphDict[u]:
                n3+=deg[u]*deg[v]

        rho=n1*n3/(n2**2)

        if rho_lim>rho:
            s=1
        else:
            s=-1

        stopCount=0

        while s*(rho_lim-rho)>0:
            i, j, n, m = 0, 0, 0, 0
            while i==m or n==j:
                ind1=int(random.random()*N)
                ind2=int(random.random()*N)
                i, n=vert[ind1], vert[ind2]
                ind3, ind4=int(random.random()*deg[i]), int(random.random()**deg[n])
                j, m=g.graphDict[i][ind3], g.graphDict[n][ind4]

            H=deg[n]*deg[m]+deg[i]*deg[j]-deg[n]*deg[j]-deg[i]*deg[m]

            if s*H<0:
                g.swapEdges([n,m],[i,j])
                n3-=2*H  # x2 because in this graph format we count each edge twice
                rho=n1*n3/(n2**2)
                stopCount=0

            else:
                stopCount+=1
                if stopCount>g.L:  # N^2 so we tried each edge once on average
                    break

        return Graph(g.graphDict)

    def genWS_nkb(self, n, k, beta):
        dic = {}
        for i in range(0, n):
            dic[i] = []
            for j in range(int(k / 2)):
                s = (i - j - 1) % n
                dic[i].append(s)
                s = (i + j + 1) % n
                dic[i].append(s)

        dic_final = copy.deepcopy(dic)

        for i in dic:
            for j in dic[i]:
                if j > i and random.random() < beta:
                    dic_final[i].remove(j)
                    dic_final[j].remove(i)
                    l = i
                    while l == i or l in dic_final[i] or i in dic_final[l]:
                        l = random.randint(0, n - 1)
                    dic_final[i].append(l)
                    dic_final[l].append(i)

        return Graph(dic_final)

    def genBA_nm(self, n, m, n_init=None, p_init=0.7):
        if n_init is None:
            n_init=m+1
            p_init=1.

        g = Graph().genER_np(n_init, p_init)
        while g.findIsolatedVertices() != []:  # Have to begin with connected graph
            g = Graph().genER_np(n_init, p_init)


        nodesSeq=[]
        for u in g.graphDict:
            for v in g.graphDict[u]:
                nodesSeq.append(v)
        nbChoices = len(nodesSeq)

        for i in range(n_init, n):
            if i%(n/10)==0:
                print(i)

            rndInd=[]
            for j in range(m):
                cnt=1
                while cnt!=m:
                    a=int(random.random() * nbChoices)
                    if a not in rndInd:
                        rndInd.append(int(random.random()*nbChoices))
                        cnt+=1
                    else:
                        cnt-=1

            g.addVertex(i)
            for j in rndInd:
                target=nodesSeq[j]
                g.addEdge([i, target])
                nodesSeq.append(i)
                nodesSeq.append(target)



            nbChoices+=2*m

        return Graph(g.graphDict)

    def genCN(self, degSeq, forceSimpleGraph=True):
        dictGraph = {}
        degSeqMR=[]
        for i in range(len(degSeq)):  # Initializes the graph dictionary
            dictGraph[i] = []
            for j in range(degSeq[i]):  # Builds Malloy-Reed degree seq
                degSeqMR.append(i)

        if len(degSeqMR)%2!=0:  # To have an even number of stubs
            degSeqMR.append(random.randint(0, len(degSeq)-1))

        random.shuffle(degSeqMR)

        i=0
        while i <= len(degSeqMR)-2:
            if (forceSimpleGraph and not ((degSeqMR[i+1] in dictGraph[degSeqMR[i]]) or degSeqMR[i]==degSeqMR[i+1])) or not forceSimpleGraph:  # Rejects self loops and multiedges
                dictGraph[degSeqMR[i]].append(degSeqMR[i+1])
                dictGraph[degSeqMR[i+1]].append(degSeqMR[i])

            elif forceSimpleGraph and ((degSeqMR[i+1] in dictGraph[degSeqMR[i]]) or degSeqMR[i]==degSeqMR[i+1]) and i<len(degSeqMR)-3:  # If multiedge or self-loop : exchange two indices
                if i+2 == len(degSeqMR)-2:  # To avoid a loop if the last nodes block the configuration network generation
                    del degSeqMR[i]
                    del degSeqMR[i+1]
                else:
                    r=random.randint(i+2, len(degSeqMR)-1)
                    degSeqMR[i], degSeqMR[r] = degSeqMR[r], degSeqMR[i]
                i-=2

            elif degSeq[degSeqMR[i]]==2 and degSeqMR[i]==degSeqMR[i+1] and forceSimpleGraph and i<len(degSeqMR)-3:  # If m_node = 2 and it did a self loop : avoids having k_node=0
                r=random.randint(i+2, len(degSeqMR)-1)
                degSeqMR[i], degSeqMR[r] = degSeqMR[r], degSeqMR[i]
                i-=2



            i+=2

        g=Graph(dictGraph)

        return g

    def genPL(self, n, alpha, kmin=3, kmax=None):
        if kmax is None:
            if n**0.5>kmin+1:
                kmax=n**0.5  # Guarantees an uncorrelated network
            else:
                kmax=n

        seq=[]
        for u in range(n):
            k=kmax+1
            while k>kmax:
                r = random.random()
                k = int(kmin*(1.-r)**(-1./(alpha-1.)))

            seq.append(k)

        return self.genCN(seq)

    def genRegularTree(self, N, nbSons):
        dic={}
        i=1
        listSons=[0]
        for u in listSons:
            dic[u]=[]
            for j in range(nbSons):
                dic[u].append(i)
                dic[i]=[u]
                listSons.append(i)
                i+=1

            if i>=N:
                break



        return Graph(dic)

    def genPrice(self, N, a, c):
        listTargets=[0, 1]

        dic={}
        dic[0]=[1]
        dic[1]=[0]
        l=2

        for i in range(2, N):
            dic[i] = []
            r_1=random.random()

            if r_1<(float(c)/(c+a)):
                r_2=int(random.random()*l)
                targ=listTargets[r_2]
            else:
                targ=int(random.random()*i)

            dic[i].append(targ)
            dic[targ].append(i)
            listTargets+=[i, targ]
            l+=2

        g=Graph(dic)
        return g

    def genStar(self, N):
        dic={}
        dic[0]=[]
        for i in range(1, N):
            dic[0].append(i)
            dic[i]=[0]
        return Graph(dic)


class Node(object):
    def __init__(self, label="", recovery_time=-1, neigh=None, state="", attributes=None, pos=None):
        self.label=label

        if neigh is None:
            self.neigh = []
        else:
            self.neigh = neigh

        self.k=len(self.neigh)

        if attributes is not None:
            self.attributes = attributes
        else:
            self.attributes = {}

        if pos is not None:
            self.pos = pos
        else:
            self.pos = []

        # Epidemics specific #

        self.recovery_time=recovery_time
        self.state=state


class BinaryHeap(object):
    def __init__(self, alist):
        i = len(alist) // 2  # Sorts tuple (obj, val) according to increasing "value"

        self.size=1 + len(alist)
        self.len = self.size-1

        self.items = [(None, None)] + alist
        while i > 0:
            self.percolate_down(i)
            i = i - 1

    def __len__(self):
        return self.size-1

    def __repr__(self):
        return str(self.items)

    def percolate_up(self):
        i = self.len
        while i // 2 > 0:
            if self.items[i][1] < self.items[i // 2][1]:
                self.items[i // 2], self.items[i] = self.items[i], self.items[i // 2]
            i = i // 2

    def insert(self, k):
        self.items.append(k)
        self.size+=1
        self.len+=1
        self.percolate_up()

    def percolate_down(self, i):
        while i * 2 <= self.len:
            mc = self.min_child(i)
            if self.items[i][1] > self.items[mc][1]:
                self.items[i], self.items[mc] = self.items[mc], self.items[i]
            i = mc

    def min_child(self, i):
        if i * 2 + 1 > self.len:
            return i * 2

        if self.items[i * 2][1] < self.items[i * 2 + 1][1]:
            return i * 2

        return i * 2 + 1

    def delete_min(self):
        return_value = self.items[1]
        self.items[1] = self.items[self.len]
        self.items.pop()
        self.size-=1
        self.len-=1
        self.percolate_down(1)
        return return_value

    def delete(self, item):
        for i in range(self.size):
            if self.items[i][0]==item:
                del self.items[i]
                break
        self.update()

    def update(self):
        self.__init__(self.items[1:])
