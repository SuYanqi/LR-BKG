from pathlib import Path

from tqdm import tqdm
import numpy as np

from config import TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, FEATURE_VECTOR_DIR

TEMPLATE_SUMMARY = "{} qid:{} " + " ".join(str(f_idx) + ":{}" for f_idx in range(1, TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM + 1, 1))
TEMPLATE_PC_SUMMARY = "{} qid:{} 1:{} 2:{} " + " ".join(str(f_idx + 2) + ":{}"
                                                        for f_idx in range(1, TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM + 1, 1))


class FormatUtil:

    @staticmethod
    def format_feature(bug_id, relevance_labels, matrix):

        template = "{} qid:{} " + " ".join(str(f_idx) + ":{}" for f_idx in range(1, matrix.shape[1] + 1, 1))
        lines = list()
        for i in range(0, len(relevance_labels)):
            line = template.format(relevance_labels[i], bug_id, *matrix[i])
            # print(line)
            # input()
            lines.append(line)

        return lines

    @staticmethod
    def format_concept_set_onehot_feature(relevance_label, cindex_value):
        """

        :param relevance_label:
        :param cindex_value: sparse matrix's column index and tfidf value pair list
        :return:
        """
        template = "{} " + " ".join(f"{pair[0]}:{pair[1]}" for pair in cindex_value)
        line = template.format(relevance_label, cindex_value)
        return line

    # @staticmethod
    # def format_summary_feature(bug_id, relevance_labels, bug_summary_feature_matrix, is_train):
    #     """
    #     ⚠️需要调整TEMPLATE中的 FEATURE_VECTOR_NUM
    #     :param is_train:
    #     :param bug_id:
    #     :param relevance_labels:
    #     :param bug_summary_feature_matrix:
    #     :return:
    #     """
    #     TEMPLATE_SUMMARY = "{} qid:{} " + " ".join(str(f_idx) + ":{}" for f_idx in range(1, TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM + 1, 1))
    #     lines = list()
    #     for i in range(0, len(relevance_labels)):
    #         begin = 0
    #         if is_train == 1 and relevance_labels[i] == 1.0:
    #             begin = 1
    #         line = TEMPLATE_SUMMARY.format(str(relevance_labels[i]), bug_id,
    #                                        *bug_summary_feature_matrix[i][begin:])
    #         # print(line)
    #         # input()
    #         lines.append(line)
    #
    #     return lines
    #
    # @staticmethod
    # def format_pc_summary_feature(bug_id, relevance_labels,
    #                               pc_name_vec,
    #                               pc_description_vec,
    #                               bug_summary_topic_keyword_feature_matrix, is_train):
    #     """
    #     ⚠️需要调整TEMPLATE中的 FEATURE_VECTOR_NUM
    #     :param pc_description_vec:
    #     :param pc_name_vec:
    #     :param is_train:
    #     :param bug_id:
    #     :param relevance_labels:
    #     :param bug_summary_topic_keyword_feature_matrix:
    #     :return:
    #     """
    #     dim = 1 + 1
    #     TEMPLATE_PC_SUMMARY = "{} qid:{} 1:{} 2:{} " + " ".join(str(f_idx + 2) + ":{}"
    #                                                             for f_idx in range(1, TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM + 1, 1))
    #     lines = list()
    #     for i in range(0, len(relevance_labels)):
    #         begin = 0
    #         if is_train == 1 and relevance_labels[i] == 1.0:
    #             begin = 1
    #         line = TEMPLATE_PC_SUMMARY.format(str(relevance_labels[i]), bug_id, pc_name_vec[i], pc_description_vec[i],
    #                                           *bug_summary_topic_keyword_feature_matrix[i][begin:])
    #         # print(line)
    #         # input()
    #         lines.append(line)
    #
    #     return lines

    @staticmethod
    def adjust_top_feature_vector_number(train_or_test, feature_num, top_num, template):
        """
        :param train_or_test:
        :param feature_num:
        :param top_num:
        :param template:
        :return:
        """
        data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_{feature_num}_tfidf")
        for index, data_file in tqdm(enumerate(data_dir.glob("*.txt")), ascii=True):
            # print(data_file)
            lines = list()
            with open(data_file, "r") as f:
                for line in f.readlines():
                    tokens = line.split(' ')
                    # print(len(tokens) - (feature_num - top_num))
                    # input()
                    line = template.format(*tokens[0: len(tokens) - (feature_num - top_num)])
                    lines.append(line)
                    # print(line)
                    # input()

            data_out_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_{top_num}_tfidf")
            data_out_dir.mkdir(exist_ok=True, parents=True)
            with open(Path(str(data_out_dir), f"feature_vector_{index}.txt"), "w") as f:
                # 利用追加模式,参数从w替换为a即可
                f.write("\n".join(lines))

    # @staticmethod
    # def merge_feature_vector():
    #     data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_{feature_num}_tfidf")
    #     for index, data_file in tqdm(enumerate(data_dir.glob("*.txt")), ascii=True):
    #         # print(data_file)
    #         lines = list()
    #         with open(data_file, "r") as f:
    #             for line in f.readlines():
    #                 tokens = line.split(' ')
    #                 line = template.format(tokens[0], tokens[1], *tokens[2: len(tokens) - 20])
    #                 lines.append(line)
    #
    #         data_out_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_{top_num}_tfidf")
    #         data_out_dir.mkdir(exist_ok=True, parents=True)
    #         with open(Path(str(data_out_dir), f"feature_vector_{index}.txt"), "w") as f:
    #             # 利用追加模式,参数从w替换为a即可
    #             f.write("\n".join(lines))
