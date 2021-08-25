from enum import Enum
import numpy as np
import scipy.sparse as sp

from py2neo import Graph, NodeMatcher


class NodeType(Enum):
    Product_Component = "Product&Component"
    Tossing = "toss"


class MyNode:
    def __init__(self, name=None, bug_num=None, mistossed_bug_num=None):
        self.name = name
        self.bug_num = bug_num
        self.mistossed_bug_num = mistossed_bug_num

    def __repr__(self):
        return f'{self.name}-{self.bug_num}-{self.mistossed_bug_num}'

    def __str__(self):
        return f'{self.name}-{self.bug_num}-{self.mistossed_bug_num}'

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self.name))


class Edge:
    def __init__(self, begin_node=None, end_node=None, frequency=None, probability=None):
        self.begin_node = begin_node
        self.end_node = end_node
        self.frequency = frequency
        self.probability = probability

    def __repr__(self):
        return f'{self.begin_node} - {self.end_node} - {self.frequency} - {self.probability}'

    def __str__(self):
        return f'{self.begin_node} - {self.end_node} - {self.frequency} - {self.probability}'

    def __eq__(self, other):
        return self.begin_node == other.begin_node and self.end_node == other.end_node

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def get_probability(node_set, edge_set):
        for node in node_set:
            # print(node)
            outdegree = 0
            node_edge_set = set()
            for edge in edge_set:
                # print(edge.begin_node)
                if node == edge.begin_node:
                    outdegree = outdegree + edge.frequency
                    node_edge_set.add(edge)
            for edge in node_edge_set:
                if outdegree != 0:
                    edge.probability = edge.frequency / outdegree
                else:
                    edge.probability = 0
            # print(f'{outdegree}::{edge.probability}')


class EdgeSet:
    def __init__(self, edge_set=None):
        self.edge_set = edge_set
        self.begin_node_edge_set_dict = dict()

    def __iter__(self):
        for edge in self.edge_set:
            yield edge

    def filter_by_node_set(self, node_set):
        filtered_edge_set = set()
        for edge in self.edge_set:
            if edge.begin_node in node_set and edge.end_node in node_set:
                filtered_edge_set.add(edge)
        self.edge_set = filtered_edge_set
        return filtered_edge_set

    def get_begin_node_edge_set_dict(self, node_set):
        for node in node_set:
            # print(node)
            edge_set = set()
            for edge in self.edge_set:
                if edge.begin_node == node:
                    edge_set.add(edge)
            self.begin_node_edge_set_dict[node] = self.begin_node_edge_set_dict.get(node, edge_set)
            # print(self.begin_node_edge_set_dict[node])
            # input()

    def get_top1_frequency_end_node_by_begin_node(self, begin_node):
        if begin_node in self.begin_node_edge_set_dict.keys():
            edge_list = list(self.begin_node_edge_set_dict[begin_node])
            if len(edge_list) > 0:
                edge_list.sort(key=lambda x: x.frequency, reverse=True)
                # if edge_list[0].frequency >= 25 and edge_list[0].probability >= 0.15:
                if edge_list[0].frequency >= 25 and edge_list[0].probability >= 0.30:
                    return edge_list[0].end_node
        return None


class GraphUtil:

    @staticmethod
    def link_neo4j(address='http://localhost:7474', username='Yanqi Su', password='111'):
        """
        连接neo4j数据库
        :param address:
        :param username:
        :param password:
        :return:
        """
        neo4j_graph = Graph(address, username=username, password=password)
        return neo4j_graph

    @staticmethod
    def find_node_by_name(graph, nodetype, nodename):
        """
        按照nodetype及nodename查找node
        :param graph:
        :param nodetype:
        :param nodename:
        :return: node
        """
        matcher = NodeMatcher(graph)
        return matcher.match(nodetype).where(f"_.name='{nodename}'").first()

    # @staticmethod
    # def check_path(graph, start_node, toss_node, end_node):
    #     rmatcher = RelationshipMatcher(graph)
    #
    #     r3 = rmatcher.match({start_node, toss_node}).first()
    #     r4 = rmatcher.match({toss_node, end_node}).first()
    #     w = r3 + r4
    #     print(Path(w))

    @staticmethod
    def construct_product_component_adjacency_matrix(product_component_list, node_set, edge_set):
        """
        construct product component adjacency matrix for negative sampling
        :param product_component_list:
        :param node_set:
        :param edge_set:
        :return:
        """
        pc_adjacency_matrix = np.zeros((186, 186))
        pc_index_dict = product_component_list.get_product_component_pair_name_index_dict()

        # 创建edge
        for edge in edge_set:
            if edge.begin_node in node_set and edge.end_node in node_set:
                # if edge.frequency >= 2:
                # print(f"{edge.begin_node}:{pc_index_dict[edge.begin_node.name]} -> "
                #       f"{edge.end_node}:{pc_index_dict[edge.end_node.name]} {edge.frequency} {edge.probability}")
                pc_adjacency_matrix[pc_index_dict[edge.begin_node.name]][pc_index_dict[edge.end_node.name]] = \
                    edge.frequency
                # (edge.frequency, edge.probability)

        return pc_adjacency_matrix
        # print(sp.csr_matrix(pc_adjacency_matrix))
