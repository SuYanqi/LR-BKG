from bug_tossing.types.bug import Bug
from bug_tossing.types.product_component_pair import ProductComponentPair, ProductComponentPairs
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil


def get_pc_bug_num_list(product_component_pair_list, pc_bug_summary_dict):
    """
    get each pc has how many bugs(resolved or mistossed)
    :param product_component_pair_list:
    :param pc_bug_summary_dict:
    :return:
    """
    pc_bug_num_list = list()
    for pc in product_component_pair_list:
        summary_vec_list = pc_bug_summary_dict[pc]
        pc_bug_num_list.append(summary_vec_list.shape[0])  # sparse matrix cannot use len(), use .shape[0]

    return pc_bug_num_list


if __name__ == "__main__":
    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    # pc_new_list = []
    # for pc in pc_list:
    #     pc = ProductComponentPair(pc.product, pc.component, pc.description, pc.community,
    #                               pc.concept_set, pc.unique_concept_set, pc.common_concept_set,
    #                               pc.controversial_concept_set, pc.topics, pc. product_component_pair_token,
    #                               pc.description_token, pc.product_component_pair_mean_vec,
    #                               pc.product_component_pair_tfidf_vec,
    #                               pc.product_component_pair_onehot_vec)
    #     pc_new_list.append(pc)
    # pc_new_list = ProductComponentPairs(pc_list)

    train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())

    pc_summary_onehot_vec_dict = train_bugs.get_pc_summary_vec_dict(pc_list, Bug.VEC_TYPE_ONEHOT)
    pc_bug_num_list = get_pc_bug_num_list(pc_list, pc_summary_onehot_vec_dict)

    pc_mistossed_summary_onehot_vec_dict = train_bugs.get_pc_mistossed_bug_summary_vec_dict(pc_list,
                                                                                            Bug.VEC_TYPE_ONEHOT)
    pc_mistossed_bug_num_list = get_pc_bug_num_list(pc_list, pc_mistossed_summary_onehot_vec_dict)
    pc_list.get_resolver_probability(pc_bug_num_list)  # , train_bugs.get_length()
    pc_list.get_participant_probability(pc_mistossed_bug_num_list)  # , train_bugs.get_length())
    pc_list.get_degree(train_bugs)
    FileUtil.dump_pickle(PathUtil.get_pc_filepath(), pc_list)
    # print(train_bugs.get_length())

    # for pc in pc_new_list:
    #     print(pc)
    #     # print(pc.community)
    #     # print(pc.resolver_probability)
    #     # print(pc.participant_probability)
    #     print(pc.in_degree)
    #     print(pc.in_degree_weight)
    #
    #     print(pc.out_degree)
    #     print(pc.out_degree_weight)
    #
    #     print(pc.degree)
    #     print(pc.degree_weight)
