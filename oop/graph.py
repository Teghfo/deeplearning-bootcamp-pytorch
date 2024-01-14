# [0 1 0 1
#  1 0 1 0

#  ]

class Graph:
    """Graph"""

    def __init__(self, directed=False) -> None:
        self.directed = directed
        self.__adj_mat = {}

    def _add_node(self, node):
        if node not in self.__adj_mat:
            self.__adj_mat[node] = []

    def add_nodes(self, *nodes):
        for node in nodes:
            self._add_node(node)

    def add_edge(self, node1, node2):
        if (node1 not in self.__adj_mat) or (node2 not in self.__adj_mat):
            self.add_nodes(node1, node2)
        self.__adj_mat[node1].append(node2)
        if not self.directed:
            self.__adj_mat[node2].append(node1)

    def node_degree(self, node):
        if node in self.__adj_mat:
            return len(self.__adj_mat[node])

    def connected(self) -> bool:
        pass

    def shortest_path(self, node1, node2) -> list:
        pass

    def print_graph(self):
        print(self.__adj_mat)


g1 = Graph()
g1.add_nodes("1", "2", "3", "4", "5")
g1.add_edge("1", "2")
g1.add_edge("1", "5")
g1.add_edge("2", "5")
g1.add_edge("2", "4")
g1.add_edge("2", "3")
g1.add_edge("3", "4")
g1.add_edge("4", "5")

print("node 1 degree:",  g1.node_degree("1"))

g1.print_graph()
