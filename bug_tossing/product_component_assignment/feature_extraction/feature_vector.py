from pathlib import Path
import scipy.sparse as sp
import numpy as np
from scipy import sparse
from tqdm import tqdm

from bug_tossing.product_component_assignment.feature_extraction.relevance_label import RelevanceLabel
from bug_tossing.types.bug import Bug
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.format_util import FormatUtil
from bug_tossing.utils.list_util import ListUtil
from bug_tossing.utils.matrix_util import MatrixUtil
from config import FEATURE_VECTOR_NUMS_PER_FILE, BLOCK_SIZE, IS_MEAN, PRODUCT_COMPONENT_PAIR_NUM


class FeatureExtractor:

    @staticmethod
    def construct_feature_vector_matrix(product_component_name_vec, product_component_description_vec,
                                        product_component_name_onehot_vec, product_component_description_onehot_vec,
                                        top_n_cos_sim_bug_summary_matrix,
                                        top_n_cos_sim_mistossed_bug_summary_matrix,
                                        top_n_cos_sim_bug_summary_onehot_matrix,
                                        top_n_cos_sim_mistossed_bug_summary_onehot_matrix,
                                        ):
        """
        concatenate different feature vector
        :param product_component_description_onehot_vec:
        :param product_component_name_onehot_vec:
        :param top_n_cos_sim_mistossed_bug_summary_onehot_matrix:
        :param top_n_cos_sim_bug_summary_onehot_matrix:
        :param product_component_name_vec:
        :param product_component_description_vec:
        :param top_n_cos_sim_bug_summary_matrix:
        :param top_n_cos_sim_mistossed_bug_summary_matrix:
        :return: feature vector matrix
        """
        product_component_name_vec = product_component_name_vec.reshape(
            len(product_component_name_vec), 1)
        product_component_description_vec = product_component_description_vec.reshape(
            len(product_component_description_vec), 1)
        product_component_name_onehot_vec = product_component_name_onehot_vec.reshape(
            len(product_component_name_onehot_vec), 1)
        product_component_description_onehot_vec = product_component_description_onehot_vec.reshape(
            len(product_component_description_onehot_vec), 1)

        feature_vector_matrix = np.concatenate((product_component_name_vec, product_component_description_vec,
                                                product_component_name_onehot_vec,
                                                product_component_description_onehot_vec,
                                                top_n_cos_sim_bug_summary_matrix,
                                                top_n_cos_sim_mistossed_bug_summary_matrix,
                                                top_n_cos_sim_bug_summary_onehot_matrix,
                                                top_n_cos_sim_mistossed_bug_summary_onehot_matrix
                                                ), axis=1)
        return feature_vector_matrix

    @staticmethod
    def get_relevance_label_list(relevance_label, bug, product_component_pair_list):
        """
        get the bug's relevance label list about product_component_pair_list
        :param relevance_label:
        :param bug:
        :param product_component_pair_list:
        :return: bug's relevance_label_list
        """
        relevance_label_list = list()
        for pc in product_component_pair_list:
            relevance_label_list.append(relevance_label.get_relevance_label(bug, pc))
        return relevance_label_list

    @staticmethod
    def get_bug_summary_block_feature_vector_matrix(bug_summary_vec_block_matrix,
                                                    feature_vector_matrix):
        """
        bug summary vec and pc name or description or historical summary ... vec cosine similarity
        :param bug_summary_vec_block_matrix: bug summary vec
        :param feature_vector_matrix: pc name or description or historical summary ... vec
        :return: cosine similarity matrixu
        """
        if feature_vector_matrix is None:
            return None
        summary_feature_matrix = MatrixUtil.cos_matrix_matrix(bug_summary_vec_block_matrix,
                                                              feature_vector_matrix)
        summary_feature_matrix[np.isnan(summary_feature_matrix)] = 0
        summary_feature_matrix[np.isinf(summary_feature_matrix)] = 0
        return summary_feature_matrix

    # @staticmethod
    # def construct_bug_summary_pc_list_top_n_feature_vector_matrix(bug_index, bug_summary_result_matrix,
    #                                                               pc_bug_num_list, n,
    #                                                               relevance_label_list,
    #                                                               is_train, is_cut):
    #     """
    #     construct one bug about pc_list feature vector matrix
    #     :param is_cut: 是否需要切除top1
    #     :param is_train:
    #     :param relevance_label_list:
    #     :param bug_summary_result_matrix: bug's feature vector about pc_list(len(bug_block) *
    #                                                                          sum(pc_bug_num in pc_bug_num_list) vec)
    #     :param bug_index: bug index in bug block
    #     :param pc_bug_num_list: each pc has bug_num bugs
    #     :param n: top_n
    #     :return: one bug about pc_list's feature vector matrix(len(pc_list) * n)
    #     """
    #     one_bug_pc_list_vec = bug_summary_result_matrix[bug_index]
    #     top_n_cos_sim_bug_summary_matrix = list()
    #     j = 0
    #     for index, bug_num in enumerate(pc_bug_num_list):
    #         temp_vec = one_bug_pc_list_vec[j: j + bug_num]
    #         j = j + bug_num
    #         temp_vec = list(temp_vec)
    #         temp_vec.sort(reverse=True)
    #         if is_train and is_cut and relevance_label_list[index] == 1:
    #             temp_vec = np.array(temp_vec[1: min(bug_num, n + 1)] + [0] * (n + 1 - bug_num))
    #         else:
    #             temp_vec = np.array(temp_vec[0: min(bug_num, n)] + [0] * (n - bug_num))
    #         top_n_cos_sim_bug_summary_matrix.append(temp_vec)
    #
    #     top_n_cos_sim_bug_summary_matrix = np.array(top_n_cos_sim_bug_summary_matrix)
    #     return top_n_cos_sim_bug_summary_matrix

    @staticmethod
    def construct_bug_summary_pc_list_top_n_feature_vector_matrix(bug_index, bug_summary_result_matrix,
                                                                  pc_bug_num_list, n,
                                                                  relevance_label_list,
                                                                  is_train, is_cut):
        """
        construct one bug about pc_list feature vector matrix
        :param is_cut: 是否需要切除top1
        :param is_train:
        :param relevance_label_list:
        :param bug_summary_result_matrix: bug's feature vector about pc_list(len(bug_block) *
                                                                             sum(pc_bug_num in pc_bug_num_list) vec)
        :param bug_index: bug index in bug block
        :param pc_bug_num_list: each pc has bug_num bugs
        :param n: top_n
        :return: np.concatenate(( bug about pc_list's feature vector matrix(len(pc_list) * n),
                 the number of nonzero cosine similarity / bug_num matrix(len(pc_list) * 1)), axis=1)
        """
        one_bug_pc_list_vec = bug_summary_result_matrix[bug_index]
        top_n_cos_sim_bug_summary_matrix = list()
        nonzero_cos_sim_percentage_vec = list()

        j = 0
        for index, bug_num in enumerate(pc_bug_num_list):
            temp_vec = one_bug_pc_list_vec[j: j + bug_num]
            nonzero_cos_sim_percentage_vec.append(len(np.nonzero(temp_vec)[0]) / bug_num)
            j = j + bug_num
            temp_vec = list(temp_vec)
            temp_vec.sort(reverse=True)
            if is_train and is_cut and relevance_label_list[index] == 1:
                temp_vec = np.array(temp_vec[1: min(bug_num, n + 1)] + [0] * (n + 1 - bug_num))
            else:
                temp_vec = np.array(temp_vec[0: min(bug_num, n)] + [0] * (n - bug_num))
            top_n_cos_sim_bug_summary_matrix.append(temp_vec)

        top_n_cos_sim_bug_summary_matrix = np.array(top_n_cos_sim_bug_summary_matrix)
        nonzero_cos_sim_percentage_vec = np.array(nonzero_cos_sim_percentage_vec)
        nonzero_cos_sim_percentage_matrix = nonzero_cos_sim_percentage_vec.reshape(len(nonzero_cos_sim_percentage_vec)
                                                                                   , 1)
        top_n_cos_sim_bug_summary_matrix = np.concatenate((nonzero_cos_sim_percentage_matrix,
                                                           top_n_cos_sim_bug_summary_matrix
                                                           ), axis=1)

        return top_n_cos_sim_bug_summary_matrix

    @staticmethod
    def get_matrix(bugs_summary_vec_matrix, bug_block_index, historical_summary_matrix,
                   pc_name_vec_matrix, pc_description_vec_matrix, historical_mistossed_summary_matrix):
        one_block_matrix = bugs_summary_vec_matrix[bug_block_index * BLOCK_SIZE: (bug_block_index + 1) * BLOCK_SIZE]

        summary_historical_summary_matrix = FeatureExtractor.get_bug_summary_block_feature_vector_matrix(
            one_block_matrix, historical_summary_matrix)

        summary_pc_name_matrix = FeatureExtractor.get_bug_summary_block_feature_vector_matrix(
            one_block_matrix, pc_name_vec_matrix)
        summary_pc_description_matrix = FeatureExtractor.get_bug_summary_block_feature_vector_matrix(
            one_block_matrix, pc_description_vec_matrix)

        summary_mistossed_historical_summary_matrix = FeatureExtractor.get_bug_summary_block_feature_vector_matrix(
            one_block_matrix, historical_mistossed_summary_matrix)

        return summary_historical_summary_matrix, summary_pc_name_matrix, summary_pc_description_matrix, \
               summary_mistossed_historical_summary_matrix

    @staticmethod
    def get_feature_vector(n, pc_list, pc_bug_num_list
                           , historical_summary_matrix
                           , bugs
                           , is_train
                           , output_feature_vector_dir
                           , pc_name_vec_matrix=None
                           , pc_description_vec_matrix=None
                           , m=None
                           , pc_mistossed_bug_num_list=None
                           , historical_mistossed_summary_matrix=None
                           , pc_name_vec_onehot_matrix=None
                           , pc_description_vec_onehot_matrix=None
                           , historical_summary_onehot_matrix=None
                           , historical_mistossed_summary_onehot_matrix=None
                           ):
        """
        a. bug summary and pc name
        b. bug summary and pc description
        c. bug summary 和 pc下的所有bug summary相乘，取top n构成向量
        d. bug summary 和 pc下的所有mistossed bug summary相乘，取top m构成向量
        :param historical_mistossed_summary_onehot_matrix:
        :param pc_description_vec_onehot_matrix:
        :param historical_summary_onehot_matrix:
        :param pc_name_vec_onehot_matrix:
        :param m:
        :param historical_mistossed_summary_matrix:
        :param pc_mistossed_bug_num_list:
        :param pc_description_vec_matrix:
        :param pc_name_vec_matrix:
        :param bugs:
        :param is_train:
        :param n:
        :param historical_summary_matrix:
        :param pc_bug_num_list:
        :param output_feature_vector_dir:
        :param pc_list:
        :return:
        """
        lines = list()

        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        bugs_summary_vec_matrix = bugs.get_bug_summary_vec_matrix(Bug.VEC_TYPE_TFIDF)
        bugs_summary_vec_onehot_matrix = bugs.get_bug_summary_vec_matrix(Bug.VEC_TYPE_ONEHOT)

        bug_blocks = ListUtil.list_of_groups(list(bugs), BLOCK_SIZE)

        i = 0  # 控制写入文件

        for bug_block_index, bug_block in enumerate(bug_blocks):

            summary_historical_summary_matrix, summary_pc_name_matrix, \
            summary_pc_description_matrix, summary_mistossed_historical_summary_matrix = FeatureExtractor.get_matrix \
                (bugs_summary_vec_matrix, bug_block_index, historical_summary_matrix,
                 pc_name_vec_matrix, pc_description_vec_matrix,
                 historical_mistossed_summary_matrix)

            # if pc_name_vec_onehot_matrix is not None:
            summary_historical_summary_onehot_matrix, summary_pc_name_onehot_matrix, \
            summary_pc_description_onehot_matrix, \
            summary_mistossed_historical_summary_onehot_matrix \
                = FeatureExtractor.get_matrix(bugs_summary_vec_onehot_matrix, bug_block_index,
                                              historical_summary_onehot_matrix,
                                              pc_name_vec_onehot_matrix, pc_description_vec_onehot_matrix,
                                              historical_mistossed_summary_onehot_matrix)

            for bug_index, bug in tqdm(enumerate(bug_block), ascii=True):

                relevance_label_list = FeatureExtractor.get_relevance_label_list(relevance_label, bug, pc_list)

                top_n_cos_sim_bug_summary_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_historical_summary_matrix, pc_bug_num_list, n,
                    relevance_label_list, is_train, is_cut=1)

                top_n_cos_sim_mistossed_bug_summary_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_mistossed_historical_summary_matrix, pc_mistossed_bug_num_list, m,
                    relevance_label_list, is_train, is_cut=0)

                top_n_cos_sim_bug_summary_onehot_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_historical_summary_onehot_matrix, pc_bug_num_list, n,
                    relevance_label_list, is_train, is_cut=1)

                top_n_cos_sim_mistossed_bug_summary_onehot_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_mistossed_historical_summary_onehot_matrix, pc_mistossed_bug_num_list, m,
                    relevance_label_list, is_train, is_cut=0)

                if summary_pc_name_matrix is not None and summary_pc_description_matrix is not None \
                        and top_n_cos_sim_mistossed_bug_summary_matrix is not None:
                    feature_vector_matrix = FeatureExtractor.construct_feature_vector_matrix(
                        summary_pc_name_matrix[bug_index],
                        summary_pc_description_matrix[bug_index],
                        summary_pc_name_onehot_matrix[bug_index],
                        summary_pc_description_onehot_matrix[bug_index],
                        top_n_cos_sim_bug_summary_matrix,
                        top_n_cos_sim_mistossed_bug_summary_matrix,
                        top_n_cos_sim_bug_summary_onehot_matrix,
                        top_n_cos_sim_mistossed_bug_summary_onehot_matrix)
                else:
                    feature_vector_matrix = top_n_cos_sim_bug_summary_matrix

                lines.extend(FormatUtil.format_feature(bug.id,
                                                       relevance_label_list,
                                                       feature_vector_matrix))
                # print(lines)
                # input()
                if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                    with open(Path(str(output_feature_vector_dir),
                                   f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                        # 利用追加模式,参数从w替换为a即可
                        f.write("\n".join(lines))
                        lines.clear()
                    i = i + FEATURE_VECTOR_NUMS_PER_FILE

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir), f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                f.write("\n".join(lines))

    @staticmethod
    def get_feature_vector_bug_description(n, pc_list, pc_bug_num_list
                                           , historical_summary_matrix
                                           , bugs
                                           , is_train
                                           , output_feature_vector_dir
                                           , pc_name_vec_matrix=None
                                           , pc_description_vec_matrix=None
                                           , m=None
                                           , pc_mistossed_bug_num_list=None
                                           , historical_mistossed_summary_matrix=None
                                           , pc_name_vec_onehot_matrix=None
                                           , pc_description_vec_onehot_matrix=None
                                           , historical_summary_onehot_matrix=None
                                           , historical_mistossed_summary_onehot_matrix=None
                                           ):
        """
        a. bug summary and pc name
        b. bug summary and pc description
        c. bug summary 和 pc下的所有bug summary相乘，取top n构成向量
        d. bug summary 和 pc下的所有mistossed bug summary相乘，取top m构成向量
        :param historical_mistossed_summary_onehot_matrix:
        :param pc_description_vec_onehot_matrix:
        :param historical_summary_onehot_matrix:
        :param pc_name_vec_onehot_matrix:
        :param m:
        :param historical_mistossed_summary_matrix:
        :param pc_mistossed_bug_num_list:
        :param pc_description_vec_matrix:
        :param pc_name_vec_matrix:
        :param bugs:
        :param is_train:
        :param n:
        :param historical_summary_matrix:
        :param pc_bug_num_list:
        :param output_feature_vector_dir:
        :param pc_list:
        :return:
        """
        lines = list()

        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        bugs_summary_vec_matrix = bugs.get_bug_summary_vec_matrix(Bug.VEC_TYPE_TFIDF)
        bugs_summary_vec_onehot_matrix = bugs.get_bug_summary_vec_matrix(Bug.VEC_TYPE_ONEHOT)

        bug_blocks = ListUtil.list_of_groups(list(bugs), BLOCK_SIZE)

        i = 0  # 控制写入文件

        for bug_block_index, bug_block in enumerate(bug_blocks):

            summary_historical_summary_matrix, summary_pc_name_matrix, \
            summary_pc_description_matrix, summary_mistossed_historical_summary_matrix = FeatureExtractor.get_matrix \
                (bugs_summary_vec_matrix, bug_block_index, historical_summary_matrix,
                 pc_name_vec_matrix, pc_description_vec_matrix,
                 historical_mistossed_summary_matrix)

            # if pc_name_vec_onehot_matrix is not None:
            summary_historical_summary_onehot_matrix, summary_pc_name_onehot_matrix, \
            summary_pc_description_onehot_matrix, \
            summary_mistossed_historical_summary_onehot_matrix \
                = FeatureExtractor.get_matrix(bugs_summary_vec_onehot_matrix, bug_block_index,
                                              historical_summary_onehot_matrix,
                                              pc_name_vec_onehot_matrix, pc_description_vec_onehot_matrix,
                                              historical_mistossed_summary_onehot_matrix)

            for bug_index, bug in tqdm(enumerate(bug_block), ascii=True):

                relevance_label_list = FeatureExtractor.get_relevance_label_list(relevance_label, bug, pc_list)

                top_n_cos_sim_bug_summary_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_historical_summary_matrix, pc_bug_num_list, n,
                    relevance_label_list, is_train, is_cut=1)

                top_n_cos_sim_mistossed_bug_summary_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_mistossed_historical_summary_matrix, pc_mistossed_bug_num_list, m,
                    relevance_label_list, is_train, is_cut=0)

                top_n_cos_sim_bug_summary_onehot_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_historical_summary_onehot_matrix, pc_bug_num_list, n,
                    relevance_label_list, is_train, is_cut=1)

                top_n_cos_sim_mistossed_bug_summary_onehot_matrix = FeatureExtractor. \
                    construct_bug_summary_pc_list_top_n_feature_vector_matrix(
                    bug_index, summary_mistossed_historical_summary_onehot_matrix, pc_mistossed_bug_num_list, m,
                    relevance_label_list, is_train, is_cut=0)

                if summary_pc_name_matrix is not None and summary_pc_description_matrix is not None \
                        and top_n_cos_sim_mistossed_bug_summary_matrix is not None:
                    feature_vector_matrix = FeatureExtractor.construct_feature_vector_matrix(
                        summary_pc_name_matrix[bug_index],
                        summary_pc_description_matrix[bug_index],
                        summary_pc_name_onehot_matrix[bug_index],
                        summary_pc_description_onehot_matrix[bug_index],
                        top_n_cos_sim_bug_summary_matrix,
                        top_n_cos_sim_mistossed_bug_summary_matrix,
                        top_n_cos_sim_bug_summary_onehot_matrix,
                        top_n_cos_sim_mistossed_bug_summary_onehot_matrix)
                else:
                    feature_vector_matrix = top_n_cos_sim_bug_summary_matrix

                lines.extend(FormatUtil.format_feature(bug.id,
                                                       relevance_label_list,
                                                       feature_vector_matrix))
                # print(lines)
                # input()
                if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                    with open(Path(str(output_feature_vector_dir),
                                   f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                        # 利用追加模式,参数从w替换为a即可
                        f.write("\n".join(lines))
                        lines.clear()
                    i = i + FEATURE_VECTOR_NUMS_PER_FILE

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir), f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                f.write("\n".join(lines))

    @staticmethod
    def get_top_n_cos_sim_bug_summary_vector(n, pc_list, pc_bug_num_list, historical_summary_matrix, bugs
                                             , output_feature_vector_dir
                                             , is_train):
        """
        bug summary 和 pc下的所有bug summary相乘，取top n 构成向量
        :param bugs:
        :param is_train:
        :param n:
        :param historical_summary_matrix:
        :param pc_bug_num_list:
        :param output_feature_vector_dir:
        :param pc_list:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        bugs_summary_vec_matrix = bugs.get_bug_summary_vec_matrix()

        bug_blocks = ListUtil.list_of_groups(list(bugs), BLOCK_SIZE)

        i = 0  # 控制写入文件

        for index, bug_block in enumerate(bug_blocks):
            # for index in range(0, len(bug_blocks)):
            one_block_matrix = bugs_summary_vec_matrix[index * BLOCK_SIZE: (index + 1) * BLOCK_SIZE]
            result_matrix = MatrixUtil.cos_matrix_matrix(one_block_matrix, historical_summary_matrix)
            # print(result_matrix.shape)

            k = 0  # 用以取vec
            for bug in tqdm(bug_block, ascii=True):
                top_n_cos_sim_bug_summary_matrix = list()
                vec = result_matrix[k]

                k = k + 1
                relevance_label_list = list()
                for pc in pc_list:
                    relevance_label_list.append(relevance_label.get_relevance_label(bug, pc))

                # vec = MatrixUtil.cos_vector_matrix(bug.summary_mean_vec, historical_summary_matrix)
                j = 0
                for bug_num in pc_bug_num_list:
                    temp_vec = vec[j: j + bug_num]
                    j = j + bug_num
                    temp_vec = list(temp_vec)
                    temp_vec.sort(reverse=True)
                    temp_vec = np.array(temp_vec[0: min(bug_num, n + 1)] + [0] * (n + 1 - bug_num))
                    top_n_cos_sim_bug_summary_matrix.append(temp_vec)
                    # print(temp_vec.shape)

                top_n_cos_sim_bug_summary_matrix = np.array(top_n_cos_sim_bug_summary_matrix)
                # set nan = 0
                top_n_cos_sim_bug_summary_matrix[np.isnan(top_n_cos_sim_bug_summary_matrix)] = 0
                # set inf = 0
                top_n_cos_sim_bug_summary_matrix[np.isinf(top_n_cos_sim_bug_summary_matrix)] = 0

                lines.extend(FormatUtil.format_summary_feature(bug.id, relevance_label_list,
                                                               top_n_cos_sim_bug_summary_matrix, is_train))
                # print(lines)
                # input()
                if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                    with open(Path(str(output_feature_vector_dir),
                                   f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                              "w") as f:  # 利用追加模式,参数从w替换为a即可
                        f.write("\n".join(lines))
                        # f.write("\n")
                        lines.clear()
                        # print(lines)
                    # input()
                    i = i + FEATURE_VECTOR_NUMS_PER_FILE

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir), f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                f.write("\n".join(lines))

    @staticmethod
    def get_concept_set_one_hot_and_feature_vector(pc_list, pc_concept_set_vec_matrix,
                                                   # pc_unique_concept_set_vec_matrix,
                                                   # pc_common_concept_set_vec_matrix,
                                                   # pc_controversial_concept_set_vec_matrix,
                                                   # pc_uncommon_concept_set_vec_matrix,
                                                   bugs,
                                                   output_feature_vector_dir
                                                   ):
        """
        one hot
        :param pc_list:
        :param pc_concept_set_vec_matrix:
        :param bugs:
        :param output_feature_vector_dir:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()
        # relevance_label_list = list()

        # x_train_list = list()

        # num_matrix_list = list()
        # num_matrix_list.append(pc_unique_concept_set_vec_matrix)
        # num_matrix_list.append(pc_common_concept_set_vec_matrix)
        # num_matrix_list.append(pc_controversial_concept_set_vec_matrix)
        # concept_matrix_for_num = sp.vstack(num_matrix_list)
        i = 0
        for bug in tqdm(bugs, ascii=True):
            concept_set_matrix = bug.summary_concept_set_vec.multiply(pc_concept_set_vec_matrix)
            # concept_set_matrix = bug.summary_uncommon_concept_set_vec.multiply(pc_concept_set_vec_matrix)

            for pc_index, pc in enumerate(pc_list):
                relevance_label_value = relevance_label.get_relevance_label(bug, pc)
                line = str(relevance_label_value) + " "
                cindex_value = [(j, concept_set_matrix[pc_index, j]) for i, j in
                                zip(*concept_set_matrix[pc_index].nonzero())]
                cindex_value.sort()
                for pair in cindex_value:
                    line = line + f'{pair[0] + 1}:{pair[1]}' + " "
                # line = FormatUtil.format_concept_set_onehot_feature(relevance_label_value, cindex_value)

                lines.append(line)

            if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                with open(Path(str(output_feature_vector_dir),
                               f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                    # 因为用稀疏表示的时候，有时候后面的有些位没有展现，所以需要用这种方式预防
                    lines[0] = lines[0] + f'{pc_concept_set_vec_matrix.shape[1]}:{0.0}'
                    # 利用追加模式,参数从w替换为a即可
                    f.write("\n".join(lines))

                    lines.clear()
                i = i + FEATURE_VECTOR_NUMS_PER_FILE
                # break

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir),
                         f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                lines[0] = lines[0] + f'{pc_concept_set_vec_matrix.shape[1]}:{0.0}'
                f.write("\n".join(lines))

    @staticmethod
    def get_bug_summary_concept_set_one_hot_feature_vector(pc_list, pc_concept_set_vec_matrix,
                                                           # pc_unique_concept_set_vec_matrix,
                                                           # pc_common_concept_set_vec_matrix,
                                                           # pc_controversial_concept_set_vec_matrix,
                                                           # pc_uncommon_concept_set_vec_matrix,
                                                           bugs,
                                                           output_feature_vector_dir
                                                           ):
        """
        one hot
        :param pc_list:
        :param pc_concept_set_vec_matrix:
        :param bugs:
        :param output_feature_vector_dir:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        i = 0
        for bug in tqdm(bugs, ascii=True):

            for pc_index, pc in enumerate(pc_list):
                relevance_label_value = relevance_label.get_relevance_label(bug, pc)
                # concept_set_vec = bug.summary_concept_set_vec.maximum(pc_concept_set_vec_matrix[pc_index])
                concept_set_vec = bug.summary_concept_set_vec

                line = str(relevance_label_value) + " "
                cindex_value = [(j, concept_set_vec[i, j]) for i, j in
                                zip(*concept_set_vec.nonzero())]
                cindex_value.sort()
                for pair in cindex_value:
                    line = line + f'{pair[0] + 1}:{pair[1]}' + " "
                # line = FormatUtil.format_concept_set_onehot_feature(relevance_label_value, cindex_value)

                lines.append(line)

            if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                with open(Path(str(output_feature_vector_dir),
                               f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                    # 因为用稀疏表示的时候，有时候后面的有些位没有展现，所以需要用这种方式预防
                    lines[0] = lines[0] + f'{pc_concept_set_vec_matrix.shape[1]}:{0.0}'
                    # 利用追加模式,参数从w替换为a即可
                    f.write("\n".join(lines))

                    lines.clear()
                i = i + FEATURE_VECTOR_NUMS_PER_FILE
                # break

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir),
                         f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                lines[0] = lines[0] + f'{pc_concept_set_vec_matrix.shape[1]}:{0.0}'
                f.write("\n".join(lines))

    @staticmethod
    def get_concept_set_one_hot_union_feature_vector(pc_list, pc_concept_set_vec_matrix,
                                                     # pc_unique_concept_set_vec_matrix,
                                                     # pc_common_concept_set_vec_matrix,
                                                     # pc_controversial_concept_set_vec_matrix,
                                                     # pc_uncommon_concept_set_vec_matrix,
                                                     bugs,
                                                     output_feature_vector_dir
                                                     ):
        """
        one hot
        :param pc_list:
        :param pc_concept_set_vec_matrix:
        :param bugs:
        :param output_feature_vector_dir:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        i = 0
        for bug in tqdm(bugs, ascii=True):

            for pc_index, pc in enumerate(pc_list):
                relevance_label_value = relevance_label.get_relevance_label(bug, pc)
                concept_set_vec = bug.summary_concept_set_vec.maximum(pc_concept_set_vec_matrix[pc_index])

                # line = str(relevance_label_value) + " "
                cindex_value = [(j, concept_set_vec[i, j]) for i, j in
                                zip(*concept_set_vec.nonzero())]
                cindex_value.sort()
                # for pair in cindex_value:
                #     line = line + f'{pair[0] + 1}:{pair[1]}' + " "
                line = FormatUtil.format_concept_set_onehot_feature(relevance_label_value, cindex_value)

                lines.append(line)

            if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                with open(Path(str(output_feature_vector_dir),
                               f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                    # 因为用稀疏表示的时候，有时候后面的有些位没有展现，所以需要用这种方式预防
                    lines[0] = lines[0] + f'{pc_concept_set_vec_matrix.shape[1]}:{0.0}'
                    # 利用追加模式,参数从w替换为a即可
                    f.write("\n".join(lines))

                    lines.clear()
                i = i + FEATURE_VECTOR_NUMS_PER_FILE
                # break

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir),
                         f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                lines[0] = lines[0] + f'{pc_concept_set_vec_matrix.shape[1]}:{0.0}'
                f.write("\n".join(lines))

    @staticmethod
    def get_concept_set_feature_vector(pc_list, pc_concept_set_vec_matrix,
                                       pc_unique_concept_set_vec_matrix,
                                       pc_common_concept_set_vec_matrix,
                                       pc_controversial_concept_set_vec_matrix,
                                       # pc_uncommon_concept_set_vec_matrix,
                                       bugs,
                                       output_feature_vector_dir
                                       ):
        """
        one hot cos similarity
        :param pc_list:
        :param pc_concept_set_vec_matrix:
        :param pc_unique_concept_set_vec_matrix:
        :param pc_common_concept_set_vec_matrix:
        :param pc_controversial_concept_set_vec_matrix:
        :param bugs:
        :param output_feature_vector_dir:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        matrix_list = list()
        matrix_list.append(pc_concept_set_vec_matrix)
        matrix_list.append(pc_unique_concept_set_vec_matrix)
        matrix_list.append(pc_common_concept_set_vec_matrix)
        matrix_list.append(pc_controversial_concept_set_vec_matrix)
        concept_matrix = sp.vstack(matrix_list)

        bugs_concept_vec_matrix = bugs.get_bug_concept_set_vec_matrix()

        bug_blocks = ListUtil.list_of_groups(list(bugs), BLOCK_SIZE)

        i = 0  # 控制写入文件

        for index, bug_block in enumerate(bug_blocks):
            # for index in range(0, len(bug_blocks)):
            one_block_matrix = bugs_concept_vec_matrix[index * BLOCK_SIZE: (index + 1) * BLOCK_SIZE]
            result_matrix = MatrixUtil.cos_matrix_matrix(one_block_matrix, concept_matrix)
            # print(result_matrix.shape)

            # k = 0  # 用以取vec
            for bug_index, bug in tqdm(enumerate(bug_block), ascii=True):
                vec = result_matrix[bug_index]
                relevance_label_list = list()

                for pc_index, pc in enumerate(pc_list):
                    relevance_label_list.append(relevance_label.get_relevance_label(bug, pc))
                temp_vec = list()
                for j in range(0, len(matrix_list)):
                    temp_vec.append(vec[j * PRODUCT_COMPONENT_PAIR_NUM: (j + 1) * PRODUCT_COMPONENT_PAIR_NUM])

                feature_vector_matrix = np.array(temp_vec).T
                # print(feature_vector_matrix.shape)
                # print(type(feature_vector_matrix))
                lines.extend(FormatUtil.format_feature(bug.id,
                                                       relevance_label_list,
                                                       feature_vector_matrix))
                # print(lines)
                # input()
                if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                    with open(Path(str(output_feature_vector_dir),
                                   f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                        # 利用追加模式,参数从w替换为a即可
                        f.write("\n".join(lines))
                        lines.clear()
                    i = i + FEATURE_VECTOR_NUMS_PER_FILE
                    # break

            if len(lines) > 0:
                with open(
                        Path(str(output_feature_vector_dir),
                             f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                        "w") as f:  # 利用追加模式,参数从w替换为a即可
                    f.write("\n".join(lines))

    @staticmethod
    def get_concept_set_num_feature_vector(pc_list, pc_concept_set_vec_matrix,
                                           pc_unique_concept_set_vec_matrix,
                                           pc_common_concept_set_vec_matrix,
                                           pc_controversial_concept_set_vec_matrix,
                                           # pc_uncommon_concept_set_vec_matrix,
                                           bugs,
                                           output_feature_vector_dir
                                           ):
        """
        num / bug_token_len
        :param pc_controversial_concept_set_vec_matrix:
        :param pc_common_concept_set_vec_matrix:
        :param pc_unique_concept_set_vec_matrix:
        :param pc_list:
        :param pc_concept_set_vec_matrix:
        :param bugs:
        :param output_feature_vector_dir:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        matrix_list = list()
        matrix_list.append(pc_concept_set_vec_matrix)
        matrix_list.append(pc_unique_concept_set_vec_matrix)
        matrix_list.append(pc_common_concept_set_vec_matrix)
        matrix_list.append(pc_controversial_concept_set_vec_matrix)
        concept_matrix = sp.vstack(matrix_list)

        bugs_concept_vec_matrix = bugs.get_bug_concept_set_vec_matrix()
        # print(type(bugs_concept_vec_matrix))

        bug_blocks = ListUtil.list_of_groups(list(bugs), BLOCK_SIZE)

        i = 0  # 控制写入文件

        for index, bug_block in enumerate(bug_blocks):
            # for index in range(0, len(bug_blocks)):
            one_block_matrix = bugs_concept_vec_matrix[index * BLOCK_SIZE: (index + 1) * BLOCK_SIZE]
            result_matrix = one_block_matrix.dot(concept_matrix.T)
            # print(result_matrix.shape)

            # k = 0  # 用以取vec
            for bug_index, bug in tqdm(enumerate(bug_block), ascii=True):
                if len(bug.summary_token) == 0:
                    vec = np.zeros(PRODUCT_COMPONENT_PAIR_NUM * len(matrix_list))
                else:
                    vec = np.array(result_matrix[bug_index].todense())[0] / len(bug.summary_token)
                # print(vec.shape)
                relevance_label_list = list()

                for pc_index, pc in enumerate(pc_list):
                    relevance_label_list.append(relevance_label.get_relevance_label(bug, pc))
                temp_vec = list()
                for j in range(0, len(matrix_list)):
                    temp_vec.append(vec[j * PRODUCT_COMPONENT_PAIR_NUM: (j + 1) * PRODUCT_COMPONENT_PAIR_NUM])
                    # print(vec[j * PRODUCT_COMPONENT_PAIR_NUM: (j + 1) * PRODUCT_COMPONENT_PAIR_NUM].shape)

                feature_vector_matrix = np.array(temp_vec).T
                # print(feature_vector_matrix.shape)
                # print(type(feature_vector_matrix))
                lines.extend(FormatUtil.format_feature(bug.id,
                                                       relevance_label_list,
                                                       feature_vector_matrix))
                # print(lines)
                # input()
                if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                    with open(Path(str(output_feature_vector_dir),
                                   f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                        # 利用追加模式,参数从w替换为a即可
                        f.write("\n".join(lines))
                        lines.clear()
                    i = i + FEATURE_VECTOR_NUMS_PER_FILE
                    # break

            if len(lines) > 0:
                with open(
                        Path(str(output_feature_vector_dir),
                             f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                        "w") as f:  # 利用追加模式,参数从w替换为a即可
                    f.write("\n".join(lines))

    @staticmethod
    def get_graph_feature_vector(pc_list, bugs, output_feature_vector_dir):
        """
        :param pc_list:
        :param bugs:
        :param output_feature_vector_dir:
        :return:
        """
        lines = list()

        # directory = Path(output_feature_vector_dir)
        output_feature_vector_dir.mkdir(exist_ok=True, parents=True)

        relevance_label = RelevanceLabel()

        i = 0
        for bug in tqdm(bugs, ascii=True):

            for pc_index, pc in enumerate(pc_list):
                relevance_label_value = relevance_label.get_relevance_label(bug, pc)
                line = f"{relevance_label_value} qid:{bug.id} "
                # line = str(relevance_label_value) + " qid:" + bug.id
                line = line + f'{1}:{len(bug.summary_token)} {2}:{pc.community} {3}:{pc.resolver_probability} ' \
                              f'{4}:{pc.participant_probability} {5}:{pc.in_degree} {6}:{pc.out_degree} {7}:{pc.degree} ' \
                              f'{8}:{pc.in_degree_weight} {9}:{pc.out_degree_weight} {10}:{pc.degree_weight}'
                # line = FormatUtil.format_concept_set_onehot_feature(relevance_label_value, cindex_value)

                lines.append(line)

            if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:
                with open(Path(str(output_feature_vector_dir),
                               f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"), "w") as f:
                    # 利用追加模式,参数从w替换为a即可
                    f.write("\n".join(lines))

                    lines.clear()
                i = i + FEATURE_VECTOR_NUMS_PER_FILE
                # break

        if len(lines) > 0:
            with open(
                    Path(str(output_feature_vector_dir),
                         f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                    "w") as f:  # 利用追加模式,参数从w替换为a即可
                f.write("\n".join(lines))
