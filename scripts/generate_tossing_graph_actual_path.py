from py2neo import Relationship, Node

from bug_tossing.types.product_component_pair import ProductComponentPair
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.graph_util import GraphUtil, NodeType
from bug_tossing.utils.path_util import PathUtil

if __name__ == "__main__":
    # bugs = FileUtil.load_pickle(
    #     PathUtil.get_specified_product_component_bugs_filepath(
    #         ProductComponentPair("ToolKit", "Password Manager")))

    # bugs.overall_bugs()

    bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())

    graph = GraphUtil.link_neo4j()

    graph.delete_all()

    node_set, edge_set = bugs.get_nodes_edges_for_graph_actual_path()

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
            # print('OK')
            begin_node = GraphUtil.find_node_by_name(graph, NodeType.Product_Component.value, edge.begin_node)
            end_node = GraphUtil.find_node_by_name(graph, NodeType.Product_Component.value, edge.end_node)
            begin_to_end = Relationship(begin_node, 'tossing', end_node, frequency=edge.frequency, probability=edge.probability)
            graph.create(begin_to_end)
