from pathlib import Path
from bug_tossing.product_component_assignment.feature_extraction.feature_vector import FeatureExtractor
from bug_tossing.types.bug import Bug
from bug_tossing.types.product_component_pair import ProductComponentPair
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil
from config import FEATURE_VECTOR_DIR, TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, \
    TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM
import numpy as np
import scipy.sparse as sp


def merge_bug_summary_vec_into_one_matrix(product_component_pair_list, pc_summary_vec_dict, VEC_TYPE):
    matrix = list()
    bug_num_list = list()
    for pc in product_component_pair_list:
        summary_vec_list = pc_summary_vec_dict[pc]
        bug_num_list.append(summary_vec_list.shape[0])  # sparse matrix cannot use len(), use .shape[0]
        matrix.extend(summary_vec_list)
    if VEC_TYPE != 3:
        matrix = np.array(matrix)
    else:
        matrix = sp.vstack(matrix)
    return bug_num_list, matrix


# def get_feature_vector_by_word_embedding():
#     pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
#     train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())
#     pc_summary_vec_dict = train_bugs.get_pc_summary_vec_dict(pc_list)
#     bug_num_list, historical_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list, pc_summary_vec_dict)
#
#     pc_mistossed_summary_vec_dict = train_bugs.get_pc_mistossed_bug_summary_vec_dict(pc_list)
#     mistossed_bug_num_list, historical_mistossed_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list,
#                                                                                                         pc_mistossed_summary_vec_dict)
#
#     pc_name_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_TFIDF_NAME)
#     pc_description_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_TFIDF_DESCRIPTION)
#
#     test_bugs = FileUtil.load_pickle(PathUtil.get_test_bugs_filepath())
#
#     is_train = 1
#     FeatureExtractor.get_feature_vector(TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, pc_list, bug_num_list,
#                                         historical_summary_matrix,
#                                         train_bugs,
#                                         is_train,
#                                         Path(FEATURE_VECTOR_DIR,
#                                              f"train_top_{TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM}_tfidf"),
#                                         pc_name_vec_matrix,
#                                         pc_description_vec_matrix,
#                                         TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM,
#                                         mistossed_bug_num_list, historical_mistossed_summary_matrix
#                                         )
#     is_train = 0
#     FeatureExtractor.get_feature_vector(TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, pc_list, bug_num_list,
#                                         historical_summary_matrix,
#                                         test_bugs,
#                                         is_train,
#                                         Path(FEATURE_VECTOR_DIR,
#                                              f"test_top_{TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM}_tfidf"),
#                                         pc_name_vec_matrix,
#                                         pc_description_vec_matrix,
#                                         TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM,
#                                         mistossed_bug_num_list, historical_mistossed_summary_matrix
#                                         )

def get_feature_vector_by_one_hot():
    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())
    pc_summary_vec_dict = train_bugs.get_pc_summary_vec_dict(pc_list, Bug.VEC_TYPE_ONEHOT)
    bug_num_list, historical_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list, pc_summary_vec_dict,
                                                                                    Bug.VEC_TYPE_ONEHOT)

    pc_mistossed_summary_vec_dict = train_bugs.get_pc_mistossed_bug_summary_vec_dict(pc_list, Bug.VEC_TYPE_ONEHOT)
    mistossed_bug_num_list, historical_mistossed_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list,
                                                                                                        pc_mistossed_summary_vec_dict,
                                                                                                        Bug.VEC_TYPE_ONEHOT)

    pc_name_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_ONEHOT_NAME)
    pc_description_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_ONEHOT_DESCRIPTION)

    test_bugs = FileUtil.load_pickle(PathUtil.get_test_bugs_filepath())

    is_train = 1
    FeatureExtractor.get_feature_vector(TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, pc_list, bug_num_list,
                                        historical_summary_matrix,
                                        train_bugs,
                                        is_train,
                                        Path(FEATURE_VECTOR_DIR,
                                             f"train_top_{TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM}_onehot"),
                                        pc_name_vec_matrix,
                                        pc_description_vec_matrix,
                                        TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM,
                                        mistossed_bug_num_list, historical_mistossed_summary_matrix
                                        )
    is_train = 0
    FeatureExtractor.get_feature_vector(TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, pc_list, bug_num_list,
                                        historical_summary_matrix,
                                        test_bugs,
                                        is_train,
                                        Path(FEATURE_VECTOR_DIR,
                                             f"test_top_{TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM}_onehot"),
                                        pc_name_vec_matrix,
                                        pc_description_vec_matrix,
                                        TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM,
                                        mistossed_bug_num_list, historical_mistossed_summary_matrix
                                        )


if __name__ == "__main__":
    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())
    # one hot
    pc_summary_onehot_vec_dict = train_bugs.get_pc_summary_vec_dict(pc_list, Bug.VEC_TYPE_ONEHOT)
    bug_num_list, historical_onehot_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list, pc_summary_onehot_vec_dict,
                                                                                           Bug.VEC_TYPE_ONEHOT)

    pc_mistossed_onehot_summary_vec_dict = train_bugs.get_pc_mistossed_bug_summary_vec_dict(pc_list, Bug.VEC_TYPE_ONEHOT)
    mistossed_bug_num_list, historical_mistossed_onehot_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list,
                                                                                                               pc_mistossed_onehot_summary_vec_dict,
                                                                                                               Bug.VEC_TYPE_ONEHOT)

    pc_name_onehot_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_ONEHOT_NAME)
    pc_description_onehot_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_ONEHOT_DESCRIPTION)

    ###############################################################################################################
    #tfidf
    pc_summary_tfidf_vec_dict = train_bugs.get_pc_summary_vec_dict(pc_list, Bug.VEC_TYPE_TFIDF)
    bug_num_list, historical_tfidf_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list, pc_summary_tfidf_vec_dict,
                                                                                           Bug.VEC_TYPE_TFIDF)

    pc_mistossed_tfidf_summary_vec_dict = train_bugs.get_pc_mistossed_bug_summary_vec_dict(pc_list, Bug.VEC_TYPE_TFIDF)
    mistossed_bug_num_list, historical_mistossed_tfidf_summary_matrix = merge_bug_summary_vec_into_one_matrix(pc_list,
                                                                                                               pc_mistossed_tfidf_summary_vec_dict,
                                                                                                               Bug.VEC_TYPE_TFIDF)

    pc_name_tfidf_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_TFIDF_NAME)
    pc_description_tfidf_vec_matrix = pc_list.get_vec_matrix(ProductComponentPair.VEC_TYPE_TFIDF_DESCRIPTION)
    ###############################################################################################################

    test_bugs = FileUtil.load_pickle(PathUtil.get_test_bugs_filepath())

    is_train = 1
    FeatureExtractor.get_feature_vector(TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, pc_list, bug_num_list,
                                        historical_tfidf_summary_matrix,
                                        train_bugs,
                                        is_train,
                                        Path(FEATURE_VECTOR_DIR,
                                             f"train_top_{TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM}_tfidf_onehot_percentage"),
                                        pc_name_tfidf_vec_matrix,
                                        pc_description_tfidf_vec_matrix,
                                        TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM,
                                        mistossed_bug_num_list,
                                        historical_mistossed_tfidf_summary_matrix,

                                        pc_name_onehot_vec_matrix,
                                        pc_description_onehot_vec_matrix,
                                        historical_onehot_summary_matrix,
                                        historical_mistossed_onehot_summary_matrix
                                        )
    is_train = 0
    FeatureExtractor.get_feature_vector(TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM, pc_list, bug_num_list,
                                        historical_tfidf_summary_matrix,
                                        test_bugs,
                                        is_train,
                                        Path(FEATURE_VECTOR_DIR,
                                             f"test_top_{TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM}_tfidf_onehot_percentage"),
                                        pc_name_tfidf_vec_matrix,
                                        pc_description_tfidf_vec_matrix,
                                        TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM,
                                        mistossed_bug_num_list,
                                        historical_mistossed_tfidf_summary_matrix,

                                        pc_name_onehot_vec_matrix,
                                        pc_description_onehot_vec_matrix,
                                        historical_onehot_summary_matrix,
                                        historical_mistossed_onehot_summary_matrix
                                        )
