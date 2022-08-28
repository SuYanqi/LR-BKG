import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from bug_tossing.types.bug import Bug
from bug_tossing.types.result import Result
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.graph_util import EdgeSet
from bug_tossing.utils.metrics_util import MetricsUtil
from bug_tossing.utils.path_util import PathUtil
from config import OUTPUT_DIR, METRICS_DIR
import numpy as np


def get_pc_direct_link_set_dict(pc_list):
    train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())
    node_set, edge_set = train_bugs.get_nodes_edges_for_graph_goal_oriented_path(pc_list)  # ???
    edge_set = EdgeSet(edge_set).filter_by_node_set(node_set)
    pc_direct_link_set_dict = dict()

    for pc in pc_list:
        pc_name = f"{pc.product}::{pc.component}"
        pc_direct_link_set_dict[pc_name] = pc_direct_link_set_dict.get(pc_name, set())
        for edge in edge_set:
            if edge.begin_node.name == pc_name:
                pc_direct_link_set_dict[pc_name].add(edge.end_node.name)
    return pc_direct_link_set_dict


def get_community_num_dict(pc_community_dict):
    """
    统计 community以及其对应的pc数量
    :param pc_community_dict:
    :return:
    """
    community_num_dict = dict()
    for key in pc_community_dict.keys():
        community_num_dict[pc_community_dict[key]] = community_num_dict.get(pc_community_dict[key], 0) + 1
    return community_num_dict


def get_pc_bug_result_list_dict(bug_result_list, pc_list):
    """
    将 bug results 按照pc分类
    :param bug_result_list:
    :return:
    """
    pc_bug_result_list_dict = dict()
    for index, bug_result in enumerate(bug_result_list):
        true_class = bug_result["class"]
        pc_bug_result_list_dict[true_class] = pc_bug_result_list_dict.get(true_class, list())
        pc_bug_result_list_dict[true_class].append(bug_result)
    for pc in pc_list:
        pc_bug_result_list_dict[f"{pc.product}::{pc.component}"] = pc_bug_result_list_dict.get(f"{pc.product}::{pc.component}", list())
    return pc_bug_result_list_dict


def convert_result_into_vec(product_component_pair_list, pc_bug_result_list_dict):
    # 为了求average
    all_numpy_list = list()
    # 为了求以pc为unit的accuracy
    all_pc_numpy_list = list()

    for pc in product_component_pair_list:
        pc_numpy_list = list()
        pc_name = f"{pc.product}::{pc.component}"
        if pc_name in pc_bug_result_list_dict.keys():
            for bug_result in pc_bug_result_list_dict[pc_name]:
                top10_class = np.zeros(len(bug_result["top10_class"]))
                for class_index, one_class in enumerate(bug_result["top10_class"]):
                    if one_class["class"] == pc_name:
                        top10_class[class_index] = 1
                pc_numpy_list.append(top10_class)
        else:
            top10_class = np.zeros(10)
            pc_numpy_list.append(top10_class)
        all_pc_numpy_list.append(pc_numpy_list)
        all_numpy_list.extend(pc_numpy_list)

    return all_numpy_list, all_pc_numpy_list


def convert_result_into_community_vec(product_component_pair_list, pc_bug_result_list_dict):
    pc_community_dict = pc_list.get_product_component_pair_name_community_dict()

    # 为了求average
    all_numpy_list = list()
    # 为了求以pc为unit的accuracy
    all_pc_numpy_list = list()

    for pc in product_component_pair_list:
        # print(pc)
        pc_numpy_list = list()
        pc_name = f"{pc.product}::{pc.component}"
        if pc_name in pc_bug_result_list_dict.keys():
            for bug_result in pc_bug_result_list_dict[pc_name]:
                # print(bug_result)
                top10_class = np.zeros(len(bug_result["top10_class"]))
                for class_index, one_class in enumerate(bug_result["top10_class"]):
                    if one_class["class"] in pc_community_dict.keys():
                        if pc_community_dict[one_class["class"]] == pc_community_dict[pc_name]:
                            top10_class[class_index] = 1
                # print(top10_class)
                # input()
                pc_numpy_list.append(top10_class)
        else:
            top10_class = np.zeros(10)
            pc_numpy_list.append(top10_class)
        # input()
        all_pc_numpy_list.append(pc_numpy_list)
        all_numpy_list.extend(pc_numpy_list)

    return all_numpy_list, all_pc_numpy_list


def convert_result_into_direct_link_vec(product_component_pair_list, pc_bug_result_list_dict):
    pc_direct_link_set_dict = get_pc_direct_link_set_dict(product_component_pair_list)

    # 为了求average
    all_numpy_list = list()
    # 为了求以pc为unit的accuracy
    all_pc_numpy_list = list()

    for pc in product_component_pair_list:
        # print(pc)
        pc_numpy_list = list()
        pc_name = f"{pc.product}::{pc.component}"
        if pc_name in pc_bug_result_list_dict.keys():
            for bug_result in pc_bug_result_list_dict[pc_name]:
                # print(bug_result)
                top10_class = np.zeros(len(bug_result["top10_class"]))
                for class_index, one_class in enumerate(bug_result["top10_class"]):
                    if one_class["class"] == pc_name or one_class["class"] in pc_direct_link_set_dict[pc_name]:
                        top10_class[class_index] = 1
                # print(top10_class)
                # print(len(top10_class))
                # input()
                pc_numpy_list.append(top10_class)
        else:
            top10_class = np.zeros(10)
            pc_numpy_list.append(top10_class)
        all_pc_numpy_list.append(pc_numpy_list)
        all_numpy_list.extend(pc_numpy_list)

    return all_numpy_list, all_pc_numpy_list


def get_accuracy(all_numpy_list, all_pc_numpy_list):
    """

    :param product_component_pair_list:
    :param pc_bug_result_list_dict:
    :return:
    """
    average_accuracy = MetricsUtil.accuracy(all_numpy_list)
    # print(average_accuracy)
    accuracy_list = list()
    for index, pc_numpy in enumerate(all_pc_numpy_list):
        accuracy = MetricsUtil.accuracy(pc_numpy)
        accuracy_list.append(accuracy)
        # print(accuracy)
    return average_accuracy, accuracy_list


def get_ndcg(all_numpy_list, all_pc_numpy_list):
    average_ndcg = MetricsUtil.ndcg(all_numpy_list)
    # print(average_ndcg)
    ndcg_list = list()
    for index, pc_numpy in enumerate(all_pc_numpy_list):
        ndcg = MetricsUtil.ndcg(pc_numpy)
        ndcg_list.append(ndcg)
        # print(ndcg)
    return average_ndcg, ndcg_list


if __name__ == "__main__":
    # ablation = "degree"
    # metrics = FileUtil.load_json(PathUtil.get_our_metrics_ablation_filepath(ablation))
    all_bugs = ""
    # all_bugs = "tossed"
    # all_bugs = "untossed"

    if all_bugs == "":
        metrics = FileUtil.load_json(PathUtil.get_our_metrics_filepath())
    elif all_bugs == "tossed":
        metrics = FileUtil.load_json(PathUtil.get_our_tossed_metrics_filepath())
    else:
        metrics = FileUtil.load_json(PathUtil.get_our_untossed_metrics_filepath())

    # metrics = FileUtil.load_json(PathUtil.get_bugbug_metrics_filepath())
    # metrics = FileUtil.load_json(PathUtil.get_bugbug_tossed_metrics_filepath())
    # metrics = FileUtil.load_json(PathUtil.get_bugbug_untossed_metrics_filepath())

    # metrics = FileUtil.load_json(PathUtil.get_bugbug_with_tossing_graph_metrics_filepath())
    # metrics = FileUtil.load_json(PathUtil.get_bugbug_with_tossing_graph_tossed_metrics_filepath())
    # metrics = FileUtil.load_json(PathUtil.get_bugbug_with_tossing_graph_untossed_metrics_filepath())


    bug_result_list = metrics["compareList"]

    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())

    pc_bug_result_list_dict = get_pc_bug_result_list_dict(bug_result_list, pc_list)

    all_numpy_list, all_pc_numpy_list = convert_result_into_vec(pc_list, pc_bug_result_list_dict)
    average_accuracy, accuracy_list = get_accuracy(all_numpy_list, all_pc_numpy_list)
    accuracy_list.append(average_accuracy)

    all_numpy_list, all_pc_numpy_list = convert_result_into_community_vec(pc_list, pc_bug_result_list_dict)
    average_accuracy_community, accuracy_community_list = get_accuracy(all_numpy_list, all_pc_numpy_list)
    average_ndcg_community, ndcg_community_list = get_ndcg(all_numpy_list, all_pc_numpy_list)
    accuracy_community_list.append(average_accuracy_community)
    ndcg_community_list.append(average_ndcg_community)

    all_numpy_list, all_pc_numpy_list = convert_result_into_direct_link_vec(pc_list, pc_bug_result_list_dict)
    average_accuracy_direct_link, accuracy_direct_link_list = get_accuracy(all_numpy_list, all_pc_numpy_list)
    average_ndcg_direct_link, ndcg_direct_link_list = get_ndcg(all_numpy_list, all_pc_numpy_list)
    accuracy_direct_link_list.append(average_accuracy_direct_link)
    ndcg_direct_link_list.append(average_ndcg_direct_link)

    result_list = list()
    for index in tqdm(range(0, len(accuracy_list)), ascii=True):
        if index < pc_list.get_length():
            pc = f"{pc_list.product_component_pair_list[index].product}::{pc_list.product_component_pair_list[index].component}"
            result = Result(pc)
            result.bug_num = len(pc_bug_result_list_dict[pc])
        else:
            pc = "overall"
            result = Result(pc)
            result.bug_num = len(bug_result_list)
        # if index == 156:
        #     print(pc)
        #     print(accuracy_list[index])
        result.accuracy_1 = accuracy_list[index][1]
        result.accuracy_3 = accuracy_list[index][3]
        result.accuracy_5 = accuracy_list[index][5]
        result.accuracy_10 = accuracy_list[index][10]
        result.community_accuracy_1 = accuracy_community_list[index][1]
        result.community_accuracy_3 = accuracy_community_list[index][3]
        result.community_accuracy_5 = accuracy_community_list[index][5]
        result.community_accuracy_10 = accuracy_community_list[index][10]
        result.community_ndcg_1 = ndcg_community_list[index][1]
        result.community_ndcg_3 = ndcg_community_list[index][3]
        result.community_ndcg_5 = ndcg_community_list[index][5]
        result.community_ndcg_10 = ndcg_community_list[index][10]
        result.direct_link_accuracy_1 = accuracy_direct_link_list[index][1]
        result.direct_link_accuracy_3 = accuracy_direct_link_list[index][3]
        result.direct_link_accuracy_5 = accuracy_direct_link_list[index][5]
        result.direct_link_accuracy_10 = accuracy_direct_link_list[index][10]
        result.direct_link_ndcg_1 = ndcg_direct_link_list[index][1]
        result.direct_link_ndcg_3 = ndcg_direct_link_list[index][3]
        result.direct_link_ndcg_5 = ndcg_direct_link_list[index][5]
        result.direct_link_ndcg_10 = ndcg_direct_link_list[index][10]

        result_list.append(result)

    for result in result_list:
        print(result)

    # FileUtil.dump_pickle(PathUtil.get_our_result_ablation_filepath(ablation), result_list)
    if all_bugs == "":
        FileUtil.dump_pickle(PathUtil.get_our_result_filepath(), result_list)
    elif all_bugs == "tossed":
        FileUtil.dump_pickle(PathUtil.get_our_tossed_result_filepath(), result_list)
    else:
        FileUtil.dump_pickle(PathUtil.get_our_untossed_result_filepath(), result_list)

    # FileUtil.dump_pickle(PathUtil.get_bugbug_result_filepath(), result_list)
    # FileUtil.dump_pickle(PathUtil.get_bugbug_tossed_result_filepath(), result_list)
    # FileUtil.dump_pickle(PathUtil.get_bugbug_untossed_result_filepath(), result_list)

    # FileUtil.dump_pickle(PathUtil.get_bugbug_with_tossing_graph_result_filepath(), result_list)
    # FileUtil.dump_pickle(PathUtil.get_bugbug_with_tossing_graph_tossed_result_filepath(), result_list)
    # FileUtil.dump_pickle(PathUtil.get_bugbug_with_tossing_graph_untossed_result_filepath(), result_list)
