from py2neo import Relationship, Node

from bug_tossing.types.product_component_pair import ProductComponentPair
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.graph_util import GraphUtil, NodeType
from bug_tossing.utils.path_util import PathUtil


def generate_tossing_graph_goal_oriented_path_node_without_bugnum():
    # product = "Firefox"
    # component = "about:logins"
    #
    # product = "Toolkit"
    # component = "Password Manager"
    #
    # bugs = FileUtil.load_pickle(
    #     PathUtil.get_specified_product_component_bugs_filepath(
    #         ProductComponentPair(product, component)))

    # bugs.overall_bugs()
    # print(bugs.product_component_pair_framework_list)

    # bugs = FileUtil.load_pickle(PathUtil.get_filtered_bugs_filepath())
    bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())

    graph = GraphUtil.link_neo4j()

    graph.delete_all()

    node_set, edge_set = bugs.get_nodes_edges_for_graph_goal_oriented_path()

    # product_component_pair_filepath = PathUtil.get_pc_filepath()
    # pc_dataset = FileUtil.load_pickle(product_component_pair_filepath)
    # for pc in pc_dataset:
    #     node_set.add(f'{pc.product}::'
    #                  f'{pc.component}')
    # print(len(node_set))
    #
    # 创建node
    for node in node_set:
        graph.create(Node(NodeType.Product_Component.value, name=node))

    # 创建edge
    for edge in edge_set:
        if edge.begin_node in node_set and edge.end_node in node_set:
            begin_node = GraphUtil.find_node_by_name(graph, NodeType.Product_Component.value, edge.begin_node)
            end_node = GraphUtil.find_node_by_name(graph, NodeType.Product_Component.value, edge.end_node)
            # if edge.frequency >= 5:
            begin_to_end = Relationship(begin_node, 'tossing', end_node,
                                        frequency=edge.frequency, probability=edge.probability)
            graph.create(begin_to_end)


if __name__ == "__main__":

    bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())
    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    # bugs = FileUtil.load_pickle('/Users/suyanqi/PycharmProjects/BugTossing/data/all_status/filtered_bugs.json')

    graph = GraphUtil.link_neo4j()

    graph.delete_all()

    node_set, edge_set = bugs.get_nodes_edges_for_graph_goal_oriented_path(pc_list)

    # print(edge_set)

    # product_component_pair_filepath = PathUtil.get_pc_filepath()
    # pc_dataset = FileUtil.load_pickle(product_component_pair_filepath)
    # for pc in pc_dataset:
    #     node_set.add(f'{pc.product}::'
    #                  f'{pc.component}')
    # print(len(node_set))
    #
    # 创建node
    for node in node_set:
        graph.create(Node(NodeType.Product_Component.value, name=node.name, bug_num=node.bug_num,
                          mistossed_bug_num=node.mistossed_bug_num))

    # 创建edge
    for edge in edge_set:
        if edge.begin_node in node_set and edge.end_node in node_set:
            begin_node = GraphUtil.find_node_by_name(graph, NodeType.Product_Component.value, edge.begin_node.name)
            end_node = GraphUtil.find_node_by_name(graph, NodeType.Product_Component.value, edge.end_node.name)
            if edge.frequency >= 2:  # 5
                begin_to_end = Relationship(begin_node, 'tossing', end_node,
                                            frequency=edge.frequency, probability=edge.probability)
                graph.create(begin_to_end)

    # 将数据从Neo4j中导出，使用   CALL apoc.export.graphml.all("pc_goal_oriented_frequency_4.graphml", {})
