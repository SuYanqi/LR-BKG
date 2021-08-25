from datetime import datetime

from bug_tossing.types.product_component_pair import ProductComponentPair
from bug_tossing.types.tossing_path import TossingPath
from bug_tossing.utils.nlp_util import NLPUtil

import numpy as np

from config import WORD2VEC_DIM


class Bug:
    VEC_TYPE_MEAN = 1
    VEC_TYPE_TFIDF = 2
    VEC_TYPE_ONEHOT = 3

    def __init__(self, id=None, summary=None, description=None, product_component_pair=None, tossing_path=None,
                 creation_time=None, closed_time=None, last_change_time=None, status=None, summary_token=None,
                 summary_token_vec=None, summary_token_matrix=None, summary_mean_vec=None, summary_tfidf_vec=None,
                 summary_onehot_vec=None, summary_concept_set_vec=None, summary_uncommon_concept_set_vec=None,
                 description_token=None, description_token_vec=None, description_token_matrix=None,
                 description_mean_vec=None, description_tfidf_vec=None, description_onehot_vec=None):
        self.id = id
        self.summary = summary
        self.description = description
        self.product_component_pair = product_component_pair
        self.tossing_path = tossing_path
        self.creation_time = creation_time
        self.closed_time = closed_time
        self.last_change_time = last_change_time
        self.status = status

        self.summary_token = summary_token
        self.summary_token_vec = summary_token_vec
        self.summary_token_matrix = summary_token_matrix
        self.summary_mean_vec = summary_mean_vec
        self.summary_tfidf_vec = summary_tfidf_vec
        self.summary_onehot_vec = summary_onehot_vec
        self.summary_concept_set_vec = summary_concept_set_vec
        self.summary_uncommon_concept_set_vec = summary_uncommon_concept_set_vec

        self.description_token = description_token
        self.description_token_vec = description_token_vec
        self.description_token_matrix = description_token_matrix
        self.description_mean_vec = description_mean_vec
        self.description_tfidf_vec = description_tfidf_vec
        self.description_onehot_vec = description_onehot_vec

    def __repr__(self):
        return f'https://bugzilla.mozilla.org/show_bug.cgi?id={self.id} - {self.summary} - ' \
               f'{self.product_component_pair} - {self.tossing_path} - {self.creation_time} - ' \
               f'{self.closed_time} - {self.last_change_time}'

    def __str__(self):
        return f'https://bugzilla.mozilla.org/show_bug.cgi?id={self.id} - {self.summary} - ' \
               f'{self.product_component_pair} - {self.tossing_path} - {self.creation_time} - ' \
               f'{self.closed_time} - {self.last_change_time}'

    def get_summary_token(self):
        """

        :return: token_list
        """
        self.summary_token = NLPUtil.preprocess(self.summary)

    def get_summary_token_vec(self, word2vec):
        """

        :param word2vec: word2vec model
        :return: token_vec_list
        """
        self.summary_token_vec = list()
        if self.summary_token:
            for token in self.summary_token:
                token_vec = NLPUtil.convert_word_to_vector(word2vec, token)
                self.summary_token_vec.append(token_vec)
            # print(token)
            # print(self.summary_token_vec[i])
            # i = i + 1
            # input()
        else:
            self.summary_token_vec.append(np.zeros(WORD2VEC_DIM))

    def get_description_token_vec(self, word2vec):
        """

        :param word2vec: word2vec model
        :return: token_vec_list
        """
        self.description_token_vec = list()
        if self.description_token:
            for token in self.description_token:
                token_vec = NLPUtil.convert_word_to_vector(word2vec, token)
                self.description_token_vec.append(token_vec)
            # print(token)
            # print(self.summary_token_vec[i])
            # i = i + 1
            # input()
        else:
            self.description_token_vec.append(np.zeros(WORD2VEC_DIM))

    def get_summary_token_matrix(self):
        """
        构建tokens矩阵 N: number of tokens
        :return: matrix：N * dimOfWord2vec
        """
        matrix = []
        for token_vec in self.summary_token_vec:
            matrix.append(token_vec)
        # self.summary_token_matrix = np.array(matrix)
        return np.array(matrix)

    def get_description_token_matrix(self):
        """
        构建tokens矩阵 N: number of tokens
        :return: matrix：N * dimOfWord2vec
        """
        matrix = []
        for token_vec in self.description_token_vec:
            matrix.append(token_vec)
        # self.summary_token_matrix = np.array(matrix)
        return np.array(matrix)

    def get_summary_mean_vec(self):
        """
        计算N个token的vec的平均值，得到summary vec,用以表示summary向量 N * 300 -> 1 * 300
        可以用word mover distance来替换
        :return: summary_vec
        """
        # print(self.summary_token_matrix.shape)
        self.summary_mean_vec = np.mean(self.get_summary_token_matrix(), axis=0)
        return self.summary_mean_vec

    def get_description_mean_vec(self):
        """
        计算N个token的vec的平均值，得到summary vec,用以表示summary向量 N * 300 -> 1 * 300
        可以用word mover distance来替换
        :return: summary_vec
        """
        # print(self.summary_token_matrix.shape)
        self.description_mean_vec = np.mean(self.get_description_token_matrix(), axis=0)
        return self.description_mean_vec

    # def get_summary_tfidf_vec(self, tfidf):
    #     """
    #     计算N个token的vec的tfidf加权平均值，得到summary tfidf vec,用以表示summary向量 N * 300 -> 1 * 300
    #     可以用word mover distance来替换
    #     :return:
    #     """
    #     corpus = tfidf.transform(self.summary_token)
    #     pass

    @staticmethod
    def get_summary_vec_list(bug_list):
        sv_list = []
        for bug in bug_list:
            sv_list.append(bug.summary_mean_vec)
        return sv_list

    @staticmethod
    def dict_to_object(bug_dict):
        bug = Bug()
        bug.id = bug_dict['id']
        bug.summary = bug_dict['summary']
        bug.description = bug_dict['comments'][0]['text']
        bug.product_component_pair = ProductComponentPair(bug_dict['product'], bug_dict['component'])
        bug.tossing_path = TossingPath(Bug.get_tossing_path(bug_dict['history'], bug.product_component_pair))
        bug.creation_time = datetime.strptime(bug_dict['creation_time'], "%Y-%m-%dT%H:%M:%SZ")
        if bug_dict['cf_last_resolved'] is not None:
            bug.closed_time = datetime.strptime(bug_dict['cf_last_resolved'], "%Y-%m-%dT%H:%M:%SZ")
        bug.last_change_time = datetime.strptime(bug_dict['last_change_time'], "%Y-%m-%dT%H:%M:%SZ")
        bug.status = bug_dict['status']

        bug.summary_token = NLPUtil.preprocess(bug.summary)
        bug.description_token = NLPUtil.preprocess(bug.description)
        return bug

    @staticmethod
    def get_tossing_path(history, last_product_component_pair):
        tossing_path = []
        is_tossing = 0
        for one in history:
            product_component_pair = ProductComponentPair()
            for change in one['changes']:
                if change['field_name'] == 'product':
                    product_component_pair.product = change['removed']
                    is_tossing = 1
                if change['field_name'] == 'component':
                    product_component_pair.component = change['removed']
                    is_tossing = 1
            if is_tossing == 1 and \
                    (product_component_pair.product is not None or product_component_pair.component is not None):
                tossing_path.append(product_component_pair)
        tossing_path.append(last_product_component_pair)
        tossing_path = Bug.complete_tossing_path(tossing_path)

        return tossing_path

    @staticmethod
    def complete_tossing_path(tossing_path):
        n = len(tossing_path)
        i = 0
        for pair in reversed(tossing_path):
            if pair.product is None:
                tossing_path[n - i - 1].product = tossing_path[n - i].product
            if pair.component is None:
                tossing_path[n - i - 1].component = tossing_path[n - i].component
            i = i + 1
        return tossing_path
