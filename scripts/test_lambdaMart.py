from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm

from bug_tossing.product_component_assignment.learning_to_rank.lambdaMART import LambdaMART
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil
from config import FEATURE_VECTOR_DIR, OUTPUT_DIR


if __name__ == "__main__":
    ablation = "bug_description"

    test_dir = Path(FEATURE_VECTOR_DIR, f"test_top_30_tfidf_onehot_percentage_{ablation}_graph_feature_vector")

    # test_dir = Path(FEATURE_VECTOR_DIR, f"test_top_30_tfidf_onehot_percentage_graph_{ablation}")

    product_component_pairs = FileUtil.load_pickle(PathUtil.get_pc_filepath())

    preds = LambdaMART.test_model(PathUtil.load_lambdaMart_model_filepath(),
                                  test_dir)
    id_list = list()
    pc_list = list()
    relevance_label_list = list()
    i = 0
    n = len(list(test_dir.glob("*.txt")))
    for k in range(0, n):
        test_file = Path(test_dir, f'feature_vector_{k}.txt')
        # print(test_file)
        with open(test_file) as f:  # 利用追加模式,参数从w替换为a即可
            for line in tqdm(f, ascii=True):
                tokens = line.split(' ')
                relevance_label_list.append(tokens[0])
                id_list.append(re.findall(r'\d+', tokens[1])[0])
                pc_list.append(product_component_pairs.product_component_pair_list[i % 186].product + "::"
                               + product_component_pairs.product_component_pair_list[i % 186].component)
                i = i + 1

    result = pd.DataFrame({'id': id_list, 'product_component_pair': pc_list,
                           'relevance_label': relevance_label_list, 'preds': preds})

    result.to_csv(Path(OUTPUT_DIR, f"result_{ablation}.csv"), sep=',', header=True, index=True)
