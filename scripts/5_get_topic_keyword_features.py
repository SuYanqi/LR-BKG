from pathlib import Path

from bug_tossing.product_component_assignment.feature_extraction.relevance_label import RelevanceLabel
from bug_tossing.product_component_assignment.feature_extraction.topic_keyword_features \
    import TopicKeywordFeature
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.format_util import FormatUtil
from bug_tossing.utils.matrix_util import MatrixUtil
from bug_tossing.utils.path_util import PathUtil
import numpy as np
from tqdm import tqdm
from config import FEATURE_VECTOR_DIR, FEATURE_VECTOR_NUMS_PER_FILE


# def get_topic_keyword_features(topic_keyword_feature, input_filepath, output_filepath):
#     """
#     从文件中将数据读入 转换成feature vector 存入文件
#     :param topic_keyword_feature:
#     :param input_filepath: 输入文件名
#     :param output_filepath: 输出文件名
#     :return: null
#     """
#     bugs = FileUtil.load_pickle(input_filepath)
#
#     f = open(output_filepath, "w")  # 利用追加模式,参数从w替换为a即可
#
#     relevance_label = RelevanceLabel()
#
#     lines = list()
#     i = 0
#
#     for bug in tqdm(bugs, ascii=True):
#         # print('******************************************************************************************')
#         # print(bug.product_component_pair)
#         # print(bug.tossing_path)
#         # print(f"{i}: {bug.id}")
#         bug_summary_token_matrix = bug.get_summary_token_matrix()
#         bug_summary_topic_matrix = MatrixUtil.cos_matrix_matrix(bug_summary_token_matrix,
#                                                                 topic_keyword_feature.matrix)
#         # print(bug.summary_token)
#         # print(bug_summary_topic_matrix.shape)
#
#         # set nan = 0
#         bug_summary_topic_matrix[np.isnan(bug_summary_topic_matrix)] = 0
#         # set inf = 0
#         bug_summary_topic_matrix[np.isinf(bug_summary_topic_matrix)] = 0
#
#         # set threshold = * , number < threshold, set it to 0
#         bug_summary_topic_matrix = np.where(bug_summary_topic_matrix < 0.5, 0, bug_summary_topic_matrix)
#         # print(bug_summary_topic_matrix)
#         # 将matrix 从N * 911 -> 1 * 911 求mean
#         vec = np.mean(bug_summary_topic_matrix, axis=0)
#
#         # 根据product_component_pair选择对应的值，构建186*911 matrix
#         bug_summary_topic_keyword_feature_matrix = topic_keyword_feature.get_feature_vector(vec)
#         # get relevance label list
#         relevance_label_list = list()
#         for pc in topic_keyword_feature.product_component_pair_list:
#             relevance_label_list.append(relevance_label.get_relevance_label(bug, pc))
#             # print(f'{pc}:{relevance_label.get_relevance_label(bug, pc)}')
#         # print(len(relevance_label_list))
#         lines.extend(FormatUtil.format_summary_topic_feature(bug.id, relevance_label_list,
#                                                              bug_summary_topic_keyword_feature_matrix))
#         i = i + 1
#
#         if i == 2000:
#             f.write("\n".join(lines))
#             f.write("\n")
#             lines.clear()
#             i = 0
#             # print(lines)
#             # input()
#     f.write("\n".join(lines))
#     lines.clear()
#     # k = k + 1
#     # print(len(lines))
#
#     f.close()


def get_topic_keyword_features(topic_keyword_feature, input_filepath, output_feature_vector_dir):
    """
    从文件中将数据读入 转换成feature vector 存入文件
    :param topic_keyword_feature:
    :param input_filepath: 输入文件名
    :param output_filepath: 输出文件名
    :return: null
    """
    bugs = FileUtil.load_pickle(input_filepath)
    directory = Path(output_feature_vector_dir)
    directory.mkdir(exist_ok=True, parents=True)

    relevance_label = RelevanceLabel()

    lines = list()
    i = 0

    for bug in tqdm(bugs, ascii=True):
        # print('******************************************************************************************')
        # print(bug.product_component_pair)
        # print(bug.tossing_path)
        # print(f"{i}: {bug.id}")
        bug_summary_token_matrix = bug.get_summary_token_matrix()
        bug_summary_topic_matrix = MatrixUtil.cos_matrix_matrix(bug_summary_token_matrix,
                                                                topic_keyword_feature.matrix)
        # print(bug.summary_token)
        # print(bug_summary_topic_matrix.shape)

        # set nan = 0
        bug_summary_topic_matrix[np.isnan(bug_summary_topic_matrix)] = 0
        # set inf = 0
        bug_summary_topic_matrix[np.isinf(bug_summary_topic_matrix)] = 0

        # set threshold = * , number < threshold, set it to 0
        # bug_summary_topic_matrix = np.where(bug_summary_topic_matrix < 0.5, 0, bug_summary_topic_matrix)
        # print(bug_summary_topic_matrix)
        # 将matrix 从N * 911 -> 1 * 911 求max from column
        vec = np.amax(bug_summary_topic_matrix, axis=0)
        # print(vec)
        # print(type(vec))
        # 将matrix 从N * 911 -> 1 * 911 求mean
        # vec = np.mean(bug_summary_topic_matrix, axis=0)
        # print(vec)
        # print(type(vec))
        # input()

        # 根据product_component_pair选择对应的值，构建186*911 matrix
        bug_summary_topic_keyword_feature_matrix = topic_keyword_feature.get_feature_vector(vec)
        # get relevance label list
        relevance_label_list = list()
        for pc in topic_keyword_feature.product_component_pair_list:
            relevance_label_list.append(relevance_label.get_relevance_label(bug, pc))
            # print(f'{pc}:{relevance_label.get_relevance_label(bug, pc)}')
        # print(len(relevance_label_list))
        lines.extend(FormatUtil.format_pc_summary_feature(bug.id, relevance_label_list,
                                                          bug_summary_topic_keyword_feature_matrix))

        if len(lines) == FEATURE_VECTOR_NUMS_PER_FILE:

            with open(Path(str(directory), f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                      "w") as f:  # 利用追加模式,参数从w替换为a即可
                f.write("\n".join(lines))
                # f.write("\n")
                lines.clear()
                # print(lines)
            # input()
            i = i + FEATURE_VECTOR_NUMS_PER_FILE
    if len(lines) > 0:
        with open(Path(str(directory), f"feature_vector_{int(i / FEATURE_VECTOR_NUMS_PER_FILE)}.txt"),
                  "w") as f:  # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(lines))
            # lines.clear()
        # k = k + 1


if __name__ == "__main__":
    pc_topics_list_filepath = PathUtil.get_pc_with_topics_filepath()
    pc_topics_list = FileUtil.load_pickle(pc_topics_list_filepath)
    topic_keyword_feature = TopicKeywordFeature(pc_topics_list)
    # print(topic_keyword_feature.keyword_index_dict)

    train_input_bugs_filepath = PathUtil.get_train_bugs_filepath()
    get_topic_keyword_features(topic_keyword_feature,
                               train_input_bugs_filepath,
                               Path(FEATURE_VECTOR_DIR, "train"))

    # test_input_bugs_filepath = PathUtil.get_test_bugs_filepath()
    # get_topic_keyword_features(topic_keyword_feature,
    #                            test_input_bugs_filepath,
    #                            Path(FEATURE_VECTOR_DIR, "test"))
