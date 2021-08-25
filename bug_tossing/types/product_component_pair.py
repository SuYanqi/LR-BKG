from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm

from bug_tossing.utils.nlp_util import NLPUtil
import numpy as np
import scipy.sparse as sp


class ProductComponentPair:
    VEC_TYPE_MEAN_NAME = 1
    VEC_TYPE_TFIDF_NAME = 2
    VEC_TYPE_MEAN_DESCRIPTION = 3
    VEC_TYPE_TFIDF_DESCRIPTION = 4
    VEC_TYPE_ONEHOT_NAME = 5
    VEC_TYPE_ONEHOT_DESCRIPTION = 6

    CONCEPT_SET_TYPE_ALL = 7
    CONCEPT_SET_TYPE_UNIQUE = 8
    CONCEPT_SET_TYPE_COMMON = 9
    CONCEPT_SET_TYPE_CONTROVERSIAL = 10

    def __init__(self, product=None, component=None, description=None, community=None,
                 concept_set=dict(),
                 unique_concept_set=dict(),
                 common_concept_set=dict(), controversial_concept_set=dict(),
                 topics=None, product_component_pair_token=None, description_token=None,
                 product_component_pair_mean_vec=None, product_component_pair_tfidf_vec=None,
                 product_component_pair_onehot_vec=None,
                 description_mean_vec=None, description_tfidf_vec=None, description_onehot_vec=None,
                 concept_set_vec=None, concept_set_tfidf_vec=None, concept_set_idf_vec=None):
        self.product = product
        self.component = component
        self.description = description
        self.community = community
        # self.direct_link_set = None
        self.resolver_probability = None
        self.participant_probability = None
        self.in_degree = 0
        self.out_degree = 0
        self.degree = 0
        self.in_degree_weight = 0
        self.out_degree_weight = 0
        self.degree_weight = 0
        self.concept_set = concept_set  # dict(key=word, value(index, idf, word_num, tfidf))
        self.unique_concept_set = unique_concept_set
        self.common_concept_set = common_concept_set
        self.controversial_concept_set = controversial_concept_set

        self.topics = topics

        self.product_component_pair_token = product_component_pair_token
        self.product_component_pair_mean_vec = product_component_pair_mean_vec
        self.product_component_pair_tfidf_vec = product_component_pair_tfidf_vec
        self.product_component_pair_onehot_vec = product_component_pair_onehot_vec

        self.description_token = description_token
        self.description_mean_vec = description_mean_vec
        self.description_tfidf_vec = description_tfidf_vec
        self.description_onehot_vec = description_onehot_vec

        self.concept_set_vec = concept_set_vec
        self.unique_concept_set_vec = None
        self.common_concept_set_vec = None
        self.unique_concept_set_vec = None
        self.uncommon_concept_set_vec = None

        self.concept_set_tfidf_vec = concept_set_tfidf_vec
        self.unique_concept_set_tfidf_vec = None
        self.common_concept_set_tfidf_vec = None
        self.unique_concept_set_tfidf_vec = None
        self.uncommon_concept_set_tfidf_vec = None

        self.concept_set_idf_vec = concept_set_idf_vec
        self.unique_concept_set_idf_vec = None
        self.common_concept_set_idf_vec = None
        self.unique_concept_set_idf_vec = None
        self.uncommon_concept_set_idf_vec = None

    def __eq__(self, other):
        return self.product == other.product and self.component == other.component

    def __repr__(self):
        return f'{self.product}::{self.component}'  # - {self.community}'

    def __str__(self):
        return f'{self.product}::{self.component}'  # - {self.community}'

    def __hash__(self):
        # print(hash(str(self)))
        return hash(str(self))

    def get_product_component_pair_token(self):
        self.product_component_pair_token = NLPUtil.preprocess(f'{self.product} {self.component}')

    def get_description_token(self):
        self.description_token = NLPUtil.preprocess(self.description)

    def get_product_component_pair_mean_vec(self, nlp_model):
        if self.product_component_pair_token is None:
            self.get_product_component_pair_token()
        vec_list = list()
        for token in self.product_component_pair_token:
            vec = NLPUtil.convert_word_to_vector(nlp_model, token)
            # print(f'{token}: {vec}')
            vec_list.append(vec)
        vec_array = np.array(vec_list)
        # print(vec_array)
        self.product_component_pair_mean_vec = np.mean(vec_array, axis=0)
        # print(f'mean: {self.product_component_pair_mean_vec}')

    def get_description_mean_vec(self, nlp_model):
        if self.description_token is None:
            self.get_description_token()
        vec_list = list()
        for token in self.description_token:
            vec = NLPUtil.convert_word_to_vector(nlp_model, token)
            # print(f'{token}: {vec}')
            # print(vec.shape)
            vec_list.append(vec)
        vec_array = np.array(vec_list)
        # print(vec_array)
        self.description_mean_vec = np.mean(vec_array, axis=0)
        # print(f'mean: {self.description_mean_vec}')

    def get_topics_keywords_vec(self, word2vec):
        """
        给topics下所有keyword的keyword_vec赋值
        :param word2vec:
        :return:
        """
        for topic in self.topics:
            # print(topic)
            topic.keyword_vec = NLPUtil.convert_word_to_vector(word2vec, topic.keyword)

    def get_vec_by_vec_type(self, VEC_TYPE):
        if VEC_TYPE == ProductComponentPair.VEC_TYPE_MEAN_NAME:
            return self.product_component_pair_mean_vec
        elif VEC_TYPE == ProductComponentPair.VEC_TYPE_TFIDF_NAME:
            return self.product_component_pair_tfidf_vec
        elif VEC_TYPE == ProductComponentPair.VEC_TYPE_MEAN_DESCRIPTION:
            return self.description_mean_vec
        elif VEC_TYPE == ProductComponentPair.VEC_TYPE_ONEHOT_NAME:
            return self.product_component_pair_onehot_vec
        elif VEC_TYPE == ProductComponentPair.VEC_TYPE_ONEHOT_DESCRIPTION:
            return self.description_onehot_vec
        else:
            return self.description_tfidf_vec

    def set_unique_concept_set(self, max_idf):
        for word in self.concept_set.keys():
            if self.concept_set[word][1] == max_idf:
                self.unique_concept_set[word] = self.concept_set[word]

    def set_unique_common_controversial_concept_set(self, concept_set):
        self.unique_concept_set = dict()
        self.common_concept_set = dict()
        self.controversial_concept_set = dict()
        for word in self.concept_set.keys():
            if word in concept_set.unique_concept_set.keys():
                self.unique_concept_set[word] = self.concept_set[word]
            elif word in concept_set.controversial_concept_set.keys():
                self.controversial_concept_set[word] = self.concept_set[word]
            elif word in concept_set.common_concept_set.keys():
                self.common_concept_set[word] = self.concept_set[word]


class ProductComponentPairFramework:
    def __init__(self, product_component_pair=None, topic=None, bug_nums=None, tossing_bug_nums=0,
                 tossing_path_framework_list=None):
        self.product_component_pair = product_component_pair
        self.topic = topic
        self.bug_nums = bug_nums
        self.tossing_bug_nums = tossing_bug_nums
        self.tossing_path_framework_list = tossing_path_framework_list

    def get_tossing_bug_nums(self):
        flag = False
        for tossing_path_framework in self.tossing_path_framework_list:
            if tossing_path_framework.tossing_path.length == 1:
                self.tossing_bug_nums = self.bug_nums - tossing_path_framework.nums
                flag = True
                break
        if not flag:
            self.tossing_bug_nums = self.bug_nums
        return self.tossing_bug_nums

    def __repr__(self):
        return f'\n{self.product_component_pair} - {self.bug_nums} - {self.tossing_bug_nums}' \
               f'\n\t{self.tossing_path_framework_list}'

    def __str__(self):
        return f'\n{self.product_component_pair} - {self.bug_nums} - {self.tossing_bug_nums}' \
               f'\n\t{self.tossing_path_framework_list}'


class ProductComponentPairs:
    def __init__(self, product_component_pair_list=None):
        self.product_component_pair_list = product_component_pair_list

    def __repr__(self):
        return f'{self.product_component_pair_list}'

    def __str__(self):
        return f'{self.product_component_pair_list}'

    def __iter__(self):
        for pc in self.product_component_pair_list:
            yield pc

    def get_length(self):
        return len(self.product_component_pair_list)

    def get_resolver_probability(self, pc_bug_num_list, bugs_num=None):
        for index, pc in enumerate(self.product_component_pair_list):
            if bugs_num is None:
                pc.resolver_probability = pc_bug_num_list[index]
            else:
                pc.resolver_probability = pc_bug_num_list[index]/bugs_num

    def get_participant_probability(self, pc_mistossed_bug_num_list, bugs_num=None):
        for index, pc in enumerate(self.product_component_pair_list):
            if bugs_num is None:
                pc.participant_probability = pc_mistossed_bug_num_list[index]
            else:
                pc.participant_probability = pc_mistossed_bug_num_list[index]/bugs_num

    def get_degree(self, bugs):
        node_set, edge_set = bugs.get_nodes_edges_for_graph_goal_oriented_path()
        for pc in self.product_component_pair_list:
            pc_name = f'{pc.product}::{pc.component}'
            for edge in edge_set:
                if edge.end_node in node_set and edge.begin_node in node_set:
                    if pc_name == edge.begin_node.name:
                        pc.out_degree = pc.out_degree + 1
                        pc.out_degree_weight = pc.out_degree_weight + edge.frequency
                    elif pc_name == edge.end_node.name:
                        pc.in_degree = pc.in_degree + 1
                        pc.in_degree_weight = pc.in_degree_weight + edge.frequency
            pc.degree = pc.in_degree + pc.out_degree
            pc.degree_weight = pc.in_degree_weight + pc.out_degree_weight

    def get_product_component_pair_name_index_dict(self):
        pc_index_dict = dict()
        for index, pc in enumerate(self.product_component_pair_list):
            pc = f"{pc.product}::{pc.component}"
            pc_index_dict[pc] = pc_index_dict.get(pc, index)
        return pc_index_dict

    def get_product_component_pair_name_community_dict(self):
        pc_community_dict = dict()
        for pc in self.product_component_pair_list:
            pc_name = f"{pc.product}::{pc.component}"
            pc_community_dict[pc_name] = pc_community_dict.get(pc_name, pc.community)
        return pc_community_dict

    def get_product_component_pair_direct_link_set_dict(self, ):
        pc_direct_link_set_dict = dict()
        for pc in self.product_component_pair_list:
            pc_name = f"{pc.product}::{pc.component}"
        #     pc_community_dict[pc_name] = pc_community_dict.get(pc_name, pc.community)
        # return pc_community_dict

    def get_product_component_pair_name_list(self):
        pc_name_list = list()
        for pc in self.product_component_pair_list:
            pc_name_list.append(f'{pc.product}::{pc.component}')
        return pc_name_list

    def get_product_component_pair_token_list(self):
        pc_name_token_list = list()
        for pc in self.product_component_pair_list:
            if pc.product_component_pair_token is None:
                pc.get_product_component_pair_token()
            pc_name_token_list.append(pc.product_component_pair_token)
        return pc_name_token_list

    def get_description_token_list(self):
        description_token_list = list()
        for pc in self.product_component_pair_list:
            if pc.description_token is None:
                pc.get_description_token()
            description_token_list.append(pc.description_token)
        return description_token_list

    def get_product_component_pair_tfidf_vec(self, tfidf):
        pc_name_token_list = self.get_product_component_pair_token_list()
        tfidf_vec_list = tfidf.transform(pc_name_token_list)
        for index, pc in enumerate(self.product_component_pair_list):
            pc.product_component_pair_tfidf_vec = tfidf_vec_list[index]
            # if index == 0:
            #     print(pc.product_component_pair_tfidf_vec.shape)
            #
            #     print(pc.product_component_pair_tfidf_vec)

    def get_product_component_pair_onehot_vec(self, onehot):
        pc_name_token_list = self.get_product_component_pair_token_list()
        onehot_vec_list = onehot.transform(pc_name_token_list)
        onehot_vec_list = csr_matrix(onehot_vec_list)
        for index, pc in enumerate(self.product_component_pair_list):
            pc.product_component_pair_onehot_vec = onehot_vec_list[index]

    def get_description_tfidf_vec(self, tfidf):
        description_token_list = self.get_description_token_list()
        tfidf_vec_list = tfidf.transform(description_token_list)
        for index, pc in enumerate(self.product_component_pair_list):
            pc.description_tfidf_vec = tfidf_vec_list[index]
            # if index == 9:
            #     print(pc.description_tfidf_vec.shape)
            #
            #     print(pc.description_tfidf_vec)

    def get_description_onehot_vec(self, onehot):
        description_token_list = self.get_description_token_list()
        onehot_vec_list = onehot.transform(description_token_list)
        onehot_vec_list = csr_matrix(onehot_vec_list)
        for index, pc in enumerate(self.product_component_pair_list):
            pc.description_onehot_vec = onehot_vec_list[index]

    def get_vec(self, nlp_model, tfidf, onehot):
        for pc in tqdm(self.product_component_pair_list, ascii=True):
            # print(pc)
            # print(pc.description)
            pc.get_product_component_pair_mean_vec(nlp_model)
            pc.get_description_mean_vec(nlp_model)
            # input()
        self.get_product_component_pair_tfidf_vec(tfidf)
        self.get_description_tfidf_vec(tfidf)
        self.get_product_component_pair_onehot_vec(onehot)
        self.get_description_onehot_vec(onehot)

    def get_vec_matrix(self, VEC_TYPE):
        vec_list = list()
        for pc in self.product_component_pair_list:
            vec_list.append(pc.get_vec_by_vec_type(VEC_TYPE))
        if VEC_TYPE <= 4:
            return np.array(vec_list)
        else:
            return sp.vstack(vec_list)

    def set_concept_set(self, corpus, concept):
        """
        concept_set
        :param corpus:
        :param concept:
        :return:
        """

        for index, pc in enumerate(self.product_component_pair_list):
            pc.concept_set = dict()
            word_count_dict = NLPUtil.count_word_in_sentence(corpus[index])
            for word in corpus[index]:
                if word in concept.concept_set.keys():
                    pc.concept_set[word] = (concept.concept_set[word][0], concept.concept_set[word][1],
                                            word_count_dict.get(word, 1),
                                            word_count_dict.get(word, 1) / len(corpus[index]) *
                                            concept.concept_set[word][1])

    def get_concept_set_onehot_vec(self, concept_set):
        self.transform(concept_set, ProductComponentPair.CONCEPT_SET_TYPE_ALL)
        self.transform(concept_set, ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE)
        self.transform(concept_set, ProductComponentPair.CONCEPT_SET_TYPE_COMMON)
        self.transform(concept_set, ProductComponentPair.CONCEPT_SET_TYPE_CONTROVERSIAL)
        self.transform_uncommon(concept_set)

    def transform(self, concept_set, concept_set_type):
        concept_set_list = list()
        concept_set_tfidf_list = list()
        concept_set_idf_list = list()
        for pc in self.product_component_pair_list:
            word_array = np.zeros(len(concept_set.concept_set))
            word_array_tfidf = np.zeros(len(concept_set.concept_set))
            word_array_idf = np.zeros(len(concept_set.concept_set))
            # concept_set_dict = dict()
            if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
                concept_set_dict = pc.concept_set
            elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
                concept_set_dict = pc.unique_concept_set
            elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
                concept_set_dict = pc.common_concept_set
            else:
                concept_set_dict = pc.controversial_concept_set

            for word in concept_set_dict.keys():
                word_array[concept_set_dict[word][0]] = 1
                word_array_tfidf[concept_set_dict[word][0]] = concept_set_dict[word][3]
                word_array_idf[concept_set_dict[word][0]] = concept_set_dict[word][1]
            concept_set_list.append(word_array)
            concept_set_tfidf_list.append(word_array_tfidf)
            concept_set_idf_list.append(word_array_idf)
        # concept_set_list = csr_matrix(concept_set_list)
        concept_set_list = csr_matrix(concept_set_list)
        concept_set_tfidf_list = csr_matrix(concept_set_tfidf_list)
        concept_set_idf_list = csr_matrix(concept_set_idf_list)
        self.set_concept_set_vec_list(concept_set_list, concept_set_type)
        self.set_concept_set_tfidf_vec_list(concept_set_tfidf_list, concept_set_type)
        self.set_concept_set_idf_vec_list(concept_set_idf_list, concept_set_type)

    def transform_uncommon(self, concept_set):
        uncommon_concept_set_list = list()
        uncommon_concept_set_tfidf_list = list()
        uncommon_concept_set_idf_list = list()
        for pc in self.product_component_pair_list:
            word_array = np.zeros(len(concept_set.word_index_dict))
            word_array_tfidf = np.zeros(len(concept_set.word_index_dict))
            word_array_idf = np.zeros(len(concept_set.word_index_dict))
            for word in pc.concept_set.keys():
                if word in concept_set.word_index_dict.keys():
                    word_array[concept_set.word_index_dict[word]] = 1
                    word_array_tfidf[concept_set.word_index_dict[word]] = pc.concept_set[word][3]
                    word_array_idf[concept_set.word_index_dict[word]] = pc.concept_set[word][1]
            uncommon_concept_set_list.append(word_array)
            uncommon_concept_set_tfidf_list.append(word_array_tfidf)
            uncommon_concept_set_idf_list.append(word_array_idf)
        uncommon_concept_set_list = csr_matrix(uncommon_concept_set_list)
        uncommon_concept_set_tfidf_list = csr_matrix(uncommon_concept_set_tfidf_list)
        uncommon_concept_set_idf_list = csr_matrix(uncommon_concept_set_idf_list)

        for index, pc in enumerate(self.product_component_pair_list):
            pc.uncommon_concept_set_vec = uncommon_concept_set_list[index]
            pc.uncommon_concept_set_tfidf_vec = uncommon_concept_set_tfidf_list[index]
            pc.uncommon_concept_set_idf_vec = uncommon_concept_set_idf_list[index]

    def set_concept_set_vec_list(self, concept_set_vec_list, concept_set_type):
        """
        将transform -> concept_set_vec_list 赋给 product_component_pairs
        :param concept_set_vec_list:
        :param concept_set_type:
        :return:
        """
        if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.concept_set_vec = concept_set_vec_list[index]
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.unique_concept_set_vec = concept_set_vec_list[index]
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.common_concept_set_vec = concept_set_vec_list[index]
        else:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.controversial_concept_set_vec = concept_set_vec_list[index]

    def set_concept_set_tfidf_vec_list(self, concept_set_vec_list, concept_set_type):
        """
        将transform -> concept_set_vec_list 赋给 product_component_pairs
        :param concept_set_vec_list:
        :param concept_set_type:
        :return:
        """
        if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.concept_set_tfidf_vec = concept_set_vec_list[index]
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.unique_concept_set_tfidf_vec = concept_set_vec_list[index]
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.common_concept_set_tfidf_vec = concept_set_vec_list[index]
        else:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.controversial_concept_set_tfidf_vec = concept_set_vec_list[index]

    def set_concept_set_idf_vec_list(self, concept_set_vec_list, concept_set_type):
        """
        将transform -> concept_set_vec_list 赋给 product_component_pairs
        :param concept_set_vec_list:
        :param concept_set_type:
        :return:
        """
        if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.concept_set_idf_vec = concept_set_vec_list[index]
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.unique_concept_set_idf_vec = concept_set_vec_list[index]
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.common_concept_set_idf_vec = concept_set_vec_list[index]
        else:
            for index, pc in enumerate(self.product_component_pair_list):
                pc.controversial_concept_set_idf_vec = concept_set_vec_list[index]

    def get_concept_set_vec_matrix(self, concept_set_type):
        concept_set_vec_list = list()
        if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
            for pc in self.product_component_pair_list:
                concept_set_vec_list.append(pc.concept_set_vec)
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
            for pc in self.product_component_pair_list:
                concept_set_vec_list.append(pc.unique_concept_set_vec)
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
            for pc in self.product_component_pair_list:
                concept_set_vec_list.append(pc.common_concept_set_vec)
        else:
            for pc in self.product_component_pair_list:
                concept_set_vec_list.append(pc.controversial_concept_set_vec)
        return sp.vstack(concept_set_vec_list)

    def get_concept_set_tfidf_vec_matrix(self, concept_set_type):
        concept_set_tfidf_vec_list = list()
        if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
            for pc in self.product_component_pair_list:
                concept_set_tfidf_vec_list.append(pc.concept_set_tfidf_vec)
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
            for pc in self.product_component_pair_list:
                concept_set_tfidf_vec_list.append(pc.unique_concept_set_tfidf_vec)
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
            for pc in self.product_component_pair_list:
                concept_set_tfidf_vec_list.append(pc.common_concept_set_tfidf_vec)
        else:
            for pc in self.product_component_pair_list:
                concept_set_tfidf_vec_list.append(pc.controversial_concept_set_tfidf_vec)
        return sp.vstack(concept_set_tfidf_vec_list)

    def get_concept_set_idf_vec_matrix(self, concept_set_type):
        concept_set_idf_vec_list = list()
        if concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_ALL:
            for pc in self.product_component_pair_list:
                concept_set_idf_vec_list.append(pc.concept_set_idf_vec)
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_UNIQUE:
            for pc in self.product_component_pair_list:
                concept_set_idf_vec_list.append(pc.unique_concept_set_idf_vec)
        elif concept_set_type == ProductComponentPair.CONCEPT_SET_TYPE_COMMON:
            for pc in self.product_component_pair_list:
                concept_set_idf_vec_list.append(pc.common_concept_set_idf_vec)
        else:
            for pc in self.product_component_pair_list:
                concept_set_idf_vec_list.append(pc.controversial_concept_set_idf_vec)
        return sp.vstack(concept_set_idf_vec_list)

    def get_uncommon_concept_set_vec_matrix(self):
        uncommon_concept_set_vec_list = list()

        for pc in self.product_component_pair_list:
            uncommon_concept_set_vec_list.append(pc.uncommon_concept_set_vec)
        return sp.vstack(uncommon_concept_set_vec_list)

    # def get_unique_controversial_concept_set_vec_matrix(self):
    #     concept_set_vec_list = list()
    #
    #     for pc in self.product_component_pair_list:
    #         concept_set_vec_list.append(pc.unique_concept_set_vec)
    #         concept_set_vec_list.x
    #
    #     for pc in self.product_component_pair_list:
    #             concept_set_vec_list.append(pc.controversial_concept_set_vec)
    #     return sp.vstack(concept_set_vec_list)


class Topic:
    def __init__(self, keyword=None, weight=None, keyword_vec=None):
        self.keyword = keyword
        self.weight = weight

        self.keyword_vec = keyword_vec

    def __eq__(self, other):
        return self.keyword == other.keyword and self.weight == other.weight

    def __repr__(self):
        return f'{self.keyword}::{self.weight}'

    def __str__(self):
        return f'{self.keyword}::{self.weight}'

    def __hash__(self):
        # print(hash(str(self)))
        return hash(str(self))

    def get_keyword_vec(self, word2vec):
        """

        :param word2vec: word2vec model
        :return: keyword_vec
        """
        self.keyword_vec = NLPUtil.convert_word_to_vector(word2vec, self.keyword)
