from pathlib import Path
import pandas as pd

from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.list_util import ListUtil
from bug_tossing.utils.metrics_util import MetricsUtil
from bug_tossing.utils.path_util import PathUtil
from config import OUTPUT_DIR, PRODUCT_COMPONENT_PAIR_NUM
import numpy as np


def get_relevance_label_list_preds_list(data):
    relevance_label = data['relevance_label']
    relevance_label = list(relevance_label)
    new_relevance_label = []
    for one_relevance_label in relevance_label:
        if one_relevance_label != 1:
            one_relevance_label = 0
        new_relevance_label.append(one_relevance_label)
    relevance_label = new_relevance_label

    relevance_label = ListUtil.list_of_groups(relevance_label, PRODUCT_COMPONENT_PAIR_NUM)

    preds = data['preds']
    preds = list(preds)
    preds = ListUtil.list_of_groups(preds, PRODUCT_COMPONENT_PAIR_NUM)

    return relevance_label, preds


def get_pc_mrr_dict(test_bugs, label_list, test_bug_probs_list):
    pc_mrr_dict = dict()

    pc_label_list_dict = dict()
    pc_test_bug_probs_list_dict = dict()

    for index, bug in enumerate(test_bugs):
        bug_pc_name = f"{bug.product_component_pair.product}::{bug.product_component_pair.component}"

        pc_label_list_dict[bug_pc_name] = pc_label_list_dict.get(bug_pc_name, [])
        pc_label_list_dict[bug_pc_name].append(label_list[index])
        pc_test_bug_probs_list_dict[bug_pc_name] = pc_test_bug_probs_list_dict.get(bug_pc_name, [])
        pc_test_bug_probs_list_dict[bug_pc_name].append(test_bug_probs_list[index])
    for pc in pc_label_list_dict.keys():
        pc_mrr_dict[pc] = pc_mrr_dict.get(pc,
                                          MetricsUtil.mrr(np.array(pc_label_list_dict[pc]),
                                                          np.array(pc_test_bug_probs_list_dict[pc])))
    pc_mrr_dict["all"] = pc_mrr_dict.get("all",
                                         MetricsUtil.mrr(np.array(relevance_label_list),
                                                         np.array(test_bug_probs_list)))

    return pc_mrr_dict


if __name__ == "__main__":
    # test_list = np.array([])
    # scores = MetricsUtil.accuracy(test_list)
    # print(scores)
    # result_dir = "best_result_now"
    # result_dir = "for_ICSE2022/bug_description"
    # result_dir = "for_ICSE2022/bug_summary_description_15"

    ############################################
    test_bugs_type = ""  # test all test bugs
    # test_bugs_type = "tossed_"  # test tossed test bugs
    # test_bugs_type = "untossed_"  # test untossed test bugs
    ############################################
    # test_bugs = FileUtil.load_pickle(PathUtil.get_test_bugs_filepath())
    test_bugs = FileUtil.load_pickle(PathUtil.get_tossed_test_bugs_filepath())
    # test_bugs = FileUtil.load_pickle(PathUtil.get_untossed_test_bugs_filepath())
    ############################################

    data = pd.read_csv(Path(OUTPUT_DIR, f"{test_bugs_type}result.csv"))
    relevance_label_list, preds_list = get_relevance_label_list_preds_list(data)
    get_pc_mrr_dict = get_pc_mrr_dict(test_bugs, relevance_label_list, preds_list)
    for pc in get_pc_mrr_dict.keys():
        print(f"{pc}: {get_pc_mrr_dict[pc]}")

