import random
from pathlib import Path

from tqdm import tqdm

from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.list_util import ListUtil
from bug_tossing.utils.path_util import PathUtil
from config import FEATURE_VECTOR_DIR, PRODUCT_COMPONENT_PAIR_NUM
import numpy as np
import scipy.sparse as sp


def get_relevance_label_nonzero(one_bug_lines):
    """
    pc_index, relevance_label_nonzero_lines
    :param one_bug_lines:
    :return:
    """
    relevance_label_nonzero_lines = list()
    pc_index = -1
    for line_index, line in enumerate(one_bug_lines):
        # line = line.replace("\n", "")
        tokens = line.split(' ')
        # print(tokens[0])
        relevance_label = float(tokens[0])
        if relevance_label != 0.0:
            # print('OK')
            relevance_label_nonzero_lines.append(line.replace("\n", ""))
            if relevance_label == 1.0:
                pc_index = line_index
    return pc_index, relevance_label_nonzero_lines


def get_pc_adjacency_matrix_nonzero_all(one_bug_lines, pc_adjacency_vec):
    nonzero_lines = list()
    nonzero_index = np.nonzero(pc_adjacency_vec)
    for index in nonzero_index[0]:
        nonzero_lines.append(one_bug_lines[index].replace("\n", ""))
    return nonzero_lines


def get_pc_adjacency_matrix_nonzero(one_bug_lines, pc_adjacency_vec):
    nonzero_num = 5
    nonzero_lines = list()
    nonzero_index = np.nonzero(pc_adjacency_vec)[0]
    nums = random.sample(range(0, len(nonzero_index)), min(nonzero_num, len(nonzero_index)))  # 随机抽样
    for num in nums:
        nonzero_lines.append(one_bug_lines[nonzero_index[num]].replace("\n", ""))
    return nonzero_lines


def get_pc_adjacency_matrix_zero(one_bug_lines, pc_adjacency_vec):
    zero_num = 3
    zero_lines = list()
    zero_index = np.where(pc_adjacency_vec == 0)[0]  # frequency = 0的pc的index(<class 'numpy.ndarray'>)
    nums = random.sample(range(0, len(zero_index)), min(zero_num, len(zero_index)))  # 随机抽样
    for num in nums:
        zero_lines.append(one_bug_lines[zero_index[num]].replace("\n", ""))
    return zero_lines


if __name__ == "__main__":
    top_num = 30
    pc_adjacency_matrix = FileUtil.load_pickle(PathUtil.get_pc_adjacency_matrix_filepath())
    data_dir = Path(FEATURE_VECTOR_DIR, f"train_top_{top_num}_tfidf_onehot_percentage")
    for index, data_file in tqdm(enumerate(data_dir.glob("*.txt")), ascii=True):
        # print(data_file)
        lines = list()
        with open(data_file, "r") as f:
            negative_sampling = list()
            lines = f.readlines()
            # print(len(lines))
            line_blocks = ListUtil.list_of_groups(lines, PRODUCT_COMPONENT_PAIR_NUM)
            # print(len(line_blocks))
            for line_block_index, line_block in enumerate(line_blocks):
                one_sampling = list()
                # a. relevance_label != 0 and get relevance_label == 1 's pc_index
                pc_index, rl_nonzero_lines = get_relevance_label_nonzero(line_block)
                one_sampling.extend(rl_nonzero_lines)
                # b. pc_adjacency_matrix pc_index对应的nonzero pc
                pc_nonzero_lines = get_pc_adjacency_matrix_nonzero(line_block, pc_adjacency_matrix[pc_index])
                one_sampling.extend(pc_nonzero_lines)
                # c. pc_adjacency_matrix pc_index对应的zero pc 随机抽取
                pc_zero_lines = get_pc_adjacency_matrix_zero(line_block, pc_adjacency_matrix[pc_index])
                one_sampling.extend(pc_zero_lines)

                one_sampling = set(one_sampling)  # 为了去重

                negative_sampling.extend(one_sampling)

                # print("\n".join(one_sampling))
                # print(pc_index)
                #
                # print(sp.csr_matrix(pc_adjacency_matrix[pc_index]))
                # print(len(one_sampling))
                # print(len(sp.csr_matrix(pc_adjacency_matrix[pc_index]).nonzero()[1]))
                # input()
                # print(len(line_block))

            # input()

        data_out_dir = Path(FEATURE_VECTOR_DIR, f"train_top_{top_num}_tfidf_onehot_percentage_negative_sampling_5_3")
        data_out_dir.mkdir(exist_ok=True, parents=True)
        with open(Path(str(data_out_dir), f"feature_vector_{index}.txt"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(negative_sampling))
