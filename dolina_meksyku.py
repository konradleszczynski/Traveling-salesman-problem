import random
import networkx as nx
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

n = random.randint(1, 15)
m = random.randint(0, int(n * (n - 1) / 2))
print("n =", n)
print("m =", m)


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.visited = False

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def removeNeighbours(self):
        self.connectedTo = {}


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)
        self.vertList[t].addNeighbor(self.vertList[f], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())


'''tworzenie grafu:'''

G = Graph()
for i in range(n):
    G.addVertex(i)
    # print(i)

i = 0

vertices = np.arange(0, n).tolist()

'''losowanie krawedzi:'''

while i < m:
    a = random.randint(0, n - 1)
    b = random.randint(0, n - 1)
    while a == b:
        b = random.randint(0, n - 1)
    if G.getVertex(b) not in G.getVertex(a).getConnections():
        G.addEdge(a, b, random.randint(1, 10))
        i += 1


cost = 0
cheapest_cost = float('inf')
best_path = ()


def hamiltonian_path_list(G, ver, path=[], weights=[0], used=0):
    global cost
    global cheapest_cost
    global best_path
    curr_ver = G.getVertex(ver)
    curr_ver.visited = True
    path.append(curr_ver.getId())
    used += 1

    for neighbour in curr_ver.getConnections():
        if not neighbour.visited:
            weights.append(curr_ver.getWeight(neighbour))
            hamiltonian_path_list(G, neighbour.getId(), path, weights, used)
        if used == n:
            for i in range(n):
                cost += weights[i]
            if cost < cheapest_cost:
                cheapest_cost = cost
                best_path = tuple(path)
            cost = 0
            break

    curr_ver.visited = False
    path.pop(len(path) - 1)
    weights.pop(len(weights) - 1)
    used -= 1
    return [best_path, cheapest_cost]


def hamiltonian_path_all_ver(G):
    for i in range(n):
        best = [(), float('inf')]
        cheapest_cost = float('inf')
        current = hamiltonian_path_list(G, i, [], [0], 0)
        if current[1] < cheapest_cost:
            best = current
    return best


najlepszy = hamiltonian_path_all_ver(G)
fastest_path = ()
if najlepszy[1] == float('inf'):
    print('Brak ścieżki Hamiltona')
else:
    print('Najlepsza trasa oraz jej czas:', najlepszy)
    fastest_path = najlepszy[0]

# rysowanie grafu i scieżki
edges = []
edge_labels = {}
for v in G:
    for w in v.getConnections():
        edges.append((v.getId(), w.getId()))
        edge_labels[(v.getId(), w.getId())] = v.getWeight(w)

G2 = nx.Graph()
G2.add_nodes_from(vertices)
G2.add_edges_from(edges)

pos = nx.spring_layout(G2)
nx.draw_networkx(G2, pos, edge_color="blue", edgelist=edges, width=0.7)

if len(fastest_path) > 0:
    best_path = list(fastest_path)
    best_edges = [(best_path[0], best_path[1])]
    for i in range(1, len(best_path) - 1):
        best_edges.append((best_path[i], best_path[i + 1]))

    nx.draw_networkx(
        G2,
        pos,
        with_labels=True,
        edgelist=best_edges,
        edge_color="red",
        node_size=200,
        width=4)

    nx.draw_networkx_edge_labels(
        G2, pos,
        edge_labels,
        font_color='green',
        font_size=10)
plt.show()
