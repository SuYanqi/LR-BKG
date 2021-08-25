import numpy as np


class TopicKeywordFeature:
    def __init__(self, product_component_pair_list=None):
        self.product_component_pair_list = product_component_pair_list
        self.keyword_index_dict = dict()
        self.matrix = []
        self.__get_topic_keyword_index_dict()
        self.construct_topic_feature_matrix()

    def __get_topic_keyword_index_dict(self):
        """
        构建topic_keyword_index_dict: value = dict(key = keyword); value is index
        :return:
        """
        i = 0
        for p_c_pair in self.product_component_pair_list:
            for topic in p_c_pair.topics:
                if topic.keyword not in self.keyword_index_dict.keys():
                    self.keyword_index_dict[topic.keyword] = (i, topic.keyword_vec)
                    i = i + 1

    def construct_topic_feature_matrix(self):
        """
        构建 numberOfKeywords (911) * dimOfWord2vec (300)维矩阵
        :return: matrix
        """
        matrix = []
        for key in self.keyword_index_dict:
            # print(f'{key}:: {self.keyword_index_dict[key][0]}')
            matrix.append(self.keyword_index_dict[key][1])
        self.matrix = np.array(matrix)

    def get_feature_vector(self, vec):
        """

        :param vec: 压缩成1*911的vector
        :return: matrix bug - pc1 :
                        bug - pc2 :
                        ...
                        bug - pc186:
        """
        matrix = []
        for pc in self.product_component_pair_list:
            # if bug.product_component_pair != pc:
            #     continue
            # print(pc)
            bug_pc_vec = np.zeros(len(vec))
            for topic in pc.topics:
                index = self.keyword_index_dict[topic.keyword][0]
                bug_pc_vec[index] = vec[index]  # * topic.weight
                # print(f'{topic.keyword}::{vec[index]} / {topic.weight} / {bug_pc_vec[index]}')
            matrix.append(bug_pc_vec)
            # print(bug_pc_vec)
        return matrix
