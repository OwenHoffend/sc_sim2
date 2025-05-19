import numpy as np

class Graph:
    def __init__(self, n):
        self.n = n
        self.E = [[] for _ in range(n)] #empty adjacency list
        self.C = [[] for _ in range(n)]
        self.clusters = []

    def add_edge(self, src, dest, c=0):
        if src < self.n and dest < self.n:
            self.E[src].append(dest)
            self.E[dest].append(src)
            self.C[src].append(c)
            self.C[dest].append(c)
        else:
            raise ValueError("Not a valid edge between {} and {}".format(src, dest))
        
    #If the graph is a cluster graph, returns sets representing the cluster nodes
    #Otherwise, returns None
    def get_clusters(self):
        visited = [False for _ in range(self.n)]
        clusters = []
        for src_idx in range(self.n):
            if visited[src_idx]:
                continue
            visited[src_idx] = True
            match_set = set([src_idx] + self.E[src_idx])
            for dest_idx in self.E[src_idx]:
                visited[dest_idx] = True
                adj_match_set = set([dest_idx] + self.E[dest_idx])
                if adj_match_set != match_set:
                    return None
            clusters.append(match_set)
        return clusters

#Given a correlation matrix C containing only 0, 1, -1, determine if it's satisfiable
#If it's satisfiable, return the S, L, and R sets, otherwise return None
def sat(C):
    n, _ = C.shape
    g = Graph(n)
    for i in range(n):
        for j in range(i):
            if C[i, j] != 0:
                g.add_edge(i, j, C[i, j])

    S = g.get_clusters()
    if S is None:
        return None

    n_star = len(S)
    P = [None for _ in range(n)]
    L = [set() for _ in range(n_star)]
    R = [set() for _ in range(n_star)]
    for i, Si in enumerate(S):
        for src_idx in Si:
            if P[src_idx] is None:
                P[src_idx] = False
                L[i].add(src_idx)
            for dest_adj_idx, dest_idx in enumerate(g.E[src_idx]):
                corr = g.C[src_idx][dest_adj_idx]
                if corr == 1:
                    if P[dest_idx] == None:
                        if P[src_idx]:
                            P[dest_idx] = True
                            R[i].add(dest_idx) 
                        else:
                            P[dest_idx] = False
                            L[i].add(dest_idx)
                    elif P[dest_idx] != P[src_idx]:
                        return None
                elif corr == -1:
                    if P[dest_idx] == None:
                        if P[src_idx]:
                            P[dest_idx] = False
                            L[i].add(dest_idx)
                        else:
                            P[dest_idx] = True
                            R[i].add(dest_idx)                  
                    elif P[dest_idx] == P[src_idx]:
                        return None
                else:
                    raise ValueError("Unexpected correlation value")
    return (S, L, R)

def sat_via_axioms(C):
    n, _ = C.shape
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j == k:
                    continue

                #SCC = 1 axiom
                if C[i, j] == 1 and C[j, k] == 1 and not C[k, i] == 1:
                    return False

                #SCC = -1 axiom
                if C[i, j] == -1 and C[j, k] == -1 and not C[k, i] == 1:
                    return False

                #SCC = 0 axiom
                if C[i, j] == 0 and abs(C[j, k]) == 1 and not C[k, i] == 0:
                    return False

    return True