import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from bug_tossing.types.bug import Bug
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil
from config import OUTPUT_DIR, METRICS_DIR, PRODUCT_COMPONENT_PAIR_NUM


def compare_result(bug, pred_class, top10_class):
    compare = {}
    compare['id'] = bug.id
    compare['summary'] = bug.summary
    compare['summary_rollback'] = bug.summary
    compare['class'] = f"{bug.product_component_pair.product}::{bug.product_component_pair.component}"
    compare['pred_class'] = pred_class
    compare['top10_class'] = top10_class
    compare['isTrue'] = compare['class'] == compare['pred_class']
    return compare


def accuracy(list):
    count = 0
    for one in list:
        if one['isTrue']:
            count = count + 1
    return count / len(list)


if __name__ == "__main__":
    tool_name = "LRBKG"
    # ablation = "ablation"
    tossed_or_untossed = ""
    # tossed_or_untossed = "tossed"
    # tossed_or_untossed = "untossed"

    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())

    if tossed_or_untossed == "tossed":
        test_bugs_filepath = PathUtil.get_tossed_test_bugs_filepath()
    elif tossed_or_untossed == "untossed":
        test_bugs_filepath = PathUtil.get_untossed_test_bugs_filepath()
    else:
        test_bugs_filepath = PathUtil.get_test_bugs_filepath()
    test_bugs = FileUtil.load_pickle(test_bugs_filepath)

    tmp_lst = list()
    if tossed_or_untossed == "":
        with open(Path(OUTPUT_DIR, f"result.csv"), 'r') as f:
        # with open(Path(OUTPUT_DIR, f"result_{ablation}.csv"), 'r') as f:

            reader = csv.reader(f)
            for row in reader:
                tmp_lst.append(row)
    else:
        with open(Path(OUTPUT_DIR, f"{tossed_or_untossed}_result.csv"), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_lst.append(row)

    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])

    result_list = []
    for i, bug in tqdm(enumerate(test_bugs), ascii=True):
        data = df[i * PRODUCT_COMPONENT_PAIR_NUM: i * PRODUCT_COMPONENT_PAIR_NUM + PRODUCT_COMPONENT_PAIR_NUM]
        data['preds'] = pd.to_numeric(data['preds'], errors='coerce')

        sorted_data = data.sort_values(by='preds', ascending=False)
        predicted_pc = sorted_data.iloc[0]['product_component_pair']

        top10_class = []
        top_n = PRODUCT_COMPONENT_PAIR_NUM
        for j in range(0, top_n):
            top = dict()
            top['rank'] = j
            top['class'] = sorted_data.iloc[j]['product_component_pair']
            top['probability'] = sorted_data.iloc[j]['preds']
            top10_class.append(top)
        compare = compare_result(bug, predicted_pc, top10_class)
        result_list.append(compare)

    result = accuracy(result_list)
    output = dict()
    output['accuracy'] = result
    output['compareList'] = result_list

    Path(METRICS_DIR, tool_name).mkdir(exist_ok=True, parents=True)

    if tossed_or_untossed == "":
        # FileUtil.dump_json(Path(METRICS_DIR, "our_approach", f"metrics_{ablation}.json"), output)
        FileUtil.dump_json(Path(METRICS_DIR, tool_name, f"metrics.json"), output)

    else:
        FileUtil.dump_json(Path(METRICS_DIR, tool_name, f"{tossed_or_untossed}_metrics.json"), output)
