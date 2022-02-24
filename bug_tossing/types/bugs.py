from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm

from bug_tossing.types.product_component_pair import ProductComponentPairFramework
from bug_tossing.types.tossing_path import TossingPathFramework
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.graph_util import Edge, MyNode

import numpy as np
import scipy.sparse as sp

from bug_tossing.utils.path_util import PathUtil
from config import WORD2VEC_DIM, ONEHOT_DIM


class Bugs:
    def __init__(self, bugs=None, product_component_pair_framework_list=None):
        self.bugs = bugs
        self.product_component_pair_framework_list = product_component_pair_framework_list
        # self.length = len(bugs)

    def __iter__(self):
        for bug in self.bugs:
            yield bug

    # def __repr__(self):
    #     return str(f'{bug}' for bug in self.bugs)
    #
    # def __str__(self):
    #     return str(f'{bug}' for bug in self.bugs)

    def get_length(self):
        return len(self.bugs)

    def get_vec(self, word2vec, tfidf, onehot):
        """
        get summary_token_vec summary_token_matrix summary_mean_vec summary_tfidf_vec
        :param tfidf:
        :param word2vec:
        :return: bugs with those
        """
        # i = 0
        for bug in tqdm(self.bugs, ascii=True):
            bug.get_summary_token_vec(word2vec)
            bug.get_summary_mean_vec()
            # bug.get_description_token_vec(word2vec)
            # bug.get_description_mean_vec()
            # i = i + 1
            # print(i)
            # print(f'https://bugzilla.mozilla.org/show_bug.cgi?id={bug.id}')
        # return self.bugs
        self.get_summary_token_tfidf_vec(tfidf)
        self.get_summary_token_onehot_vec(onehot)
        # self.get_description_token_tfidf_vec(tfidf)
        # self.get_description_token_onehot_vec(onehot)

    def get_summary_token_tfidf_vec(self, tfidf):
        summary_token_list = self.get_bug_summary_token_list()
        tfidf_vec_list = tfidf.transform(summary_token_list)
        for index, bug in enumerate(self.bugs):
            bug.summary_tfidf_vec = tfidf_vec_list[index]

    def get_description_token_tfidf_vec(self, tfidf):
        description_token_list = self.get_bug_description_token_list()
        tfidf_vec_list = tfidf.transform(description_token_list)
        for index, bug in enumerate(self.bugs):
            bug.description_tfidf_vec = tfidf_vec_list[index]

    def get_summary_token_onehot_vec(self, onehot):
        onehot_vec_list = onehot.transform(self.get_bug_summary_token_list())
        onehot_vec_list = csr_matrix(onehot_vec_list)
        for index, bug in enumerate(self.bugs):
            bug.summary_onehot_vec = onehot_vec_list[index]

    def get_description_token_onehot_vec(self, onehot):
        onehot_vec_list = onehot.transform(self.get_bug_description_token_list())
        onehot_vec_list = csr_matrix(onehot_vec_list)
        for index, bug in enumerate(self.bugs):
            bug.description_onehot_vec = onehot_vec_list[index]

    def get_summary_token_concept_set_onehot_vec(self, concept_set):
        concept_set_vec_list = concept_set.transform(self.get_bug_summary_token_list())
        concept_set_vec_list = csr_matrix(concept_set_vec_list)
        # concept_set_vec_list = csc_matrix(concept_set_vec_list)

        uncommon_concept_set_vec_list = concept_set.transform_uncommon(self.get_bug_summary_token_list())
        uncommon_concept_set_vec_list = csr_matrix(uncommon_concept_set_vec_list)
        # uncommon_concept_set_vec_list = csc_matrix(uncommon_concept_set_vec_list)

        for index, bug in enumerate(self.bugs):
            bug.summary_concept_set_vec = concept_set_vec_list[index]
            bug.summary_uncommon_concept_set_vec = uncommon_concept_set_vec_list[index]

    # 有点问题很奇怪
    def remove(self, bug):
        return self.bugs.remove(bug)

    # 有点问题很奇怪
    def filter_bugs_by_pc(self, product_component_pair_list):
        """
        过滤 被分配的P&C不在P&C list中的bug reports
        :param product_component_pair_list
        :param self: bug reports dataset
        :return: filtered bug reports
        """
        i = 0
        for bug in self.bugs:
            i = i + 1
            print(bug.id)
            print(bug.product_component_pair)
            if bug.product_component_pair not in product_component_pair_list:
                self.bugs.remove(bug)
                print('removed')
            # input()
        print(f'i : {i}')
        input()

    def filter_bugs_by_time(self, time):
        """
        将太早创建的bug report过滤掉
        :param time:
        :return:
        """
        for bug in self.bugs:
            # if bug.creation_time
            pass

    def count_tossing_bugs(self):
        """
        count tossing bugs
        :return: the number of tossing bugs
        """
        count = 0
        for bug in self:
            if bug.tossing_path.length > 1:
                count = count + 1
        return count

    def get_specified_product_component_bugs(self, product_component_pair):
        """
        get specified product&component's bugs from bugs
        :param product_component_pair: specified product&component
        :return: specified product&component's bugs
        """
        specified_bugs = []
        for bug in self.bugs:
            if bug.product_component_pair == product_component_pair:
                specified_bugs.append(bug)
        return Bugs(specified_bugs)

    def classify_bugs_by_product_component_pair_list(self, product_component_pair_list):
        """
        使用product&component_pair_list将bugs分类
        :param product_component_pair_list:
        :return: product_component_pair - bugs dict
        """
        pc_bugs_dict = dict()
        for pc in product_component_pair_list:
            pc_bugs_dict[pc] = self.get_specified_product_component_bugs(pc)

        return pc_bugs_dict

    def get_pc_summary_vec_dict(self, product_component_pair_list, VEC_TYPE):
        pc_summary_vec_dict = dict()
        for pc in product_component_pair_list:
            bugs = self.get_specified_product_component_bugs(pc)
            # print(f'{pc} - {pc_bugs_dict[pc].get_length()}')
            matrix = list()
            if VEC_TYPE == 1:
                for bug in bugs:
                    matrix.append(bug.summary_mean_vec)
            elif VEC_TYPE == 2:
                for bug in bugs:
                    matrix.append(bug.summary_tfidf_vec)
            else:
                for bug in bugs:
                    matrix.append(bug.summary_onehot_vec)

            if len(matrix) == 0:
                if VEC_TYPE != 3:
                    matrix.append(np.zeros(WORD2VEC_DIM))
                else:
                    matrix.append(sp.csr_matrix(np.zeros(ONEHOT_DIM)))  # maybe have some problem
            if VEC_TYPE != 3:
                matrix = np.array(matrix)
            else:
                matrix = sp.vstack(matrix)
            pc_summary_vec_dict[pc] = matrix
        return pc_summary_vec_dict

    def get_pc_bug_description_vec_dict(self, product_component_pair_list, VEC_TYPE):
        pc_bug_description_vec_dict = dict()
        for pc in product_component_pair_list:
            bugs = self.get_specified_product_component_bugs(pc)
            # print(f'{pc} - {pc_bugs_dict[pc].get_length()}')
            matrix = list()
            if VEC_TYPE == 1:
                for bug in bugs:
                    matrix.append(bug.description_mean_vec)
            elif VEC_TYPE == 2:
                for bug in bugs:
                    matrix.append(bug.description_tfidf_vec)
            else:
                for bug in bugs:
                    matrix.append(bug.description_onehot_vec)

            if len(matrix) == 0:
                if VEC_TYPE != 3:
                    matrix.append(np.zeros(WORD2VEC_DIM))
                else:
                    matrix.append(sp.csr_matrix(np.zeros(ONEHOT_DIM)))  # maybe have some problem
            if VEC_TYPE != 3:
                matrix = np.array(matrix)
            else:
                matrix = sp.vstack(matrix)
            pc_bug_description_vec_dict[pc] = matrix
        return pc_bug_description_vec_dict

    def get_pc_mistossed_bug_num(self, product_component_pair_list):
        pc_mistossed_bug_num = dict()
        for bug in self.bugs:
            # print(f'https://bugzilla.mozilla.org/show_bug.cgi?id={bug.id}')
            for pc in bug.tossing_path.product_component_pair_list:
                if pc in product_component_pair_list and pc != bug.product_component_pair:
                    pc_mistossed_bug_num[f"{pc.product}::{pc.component}"] = pc_mistossed_bug_num.get(f"{pc.product}::{pc.component}", 0) + 1

        for pc in product_component_pair_list:
            if f"{pc.product}::{pc.component}" not in pc_mistossed_bug_num.keys():
                pc_mistossed_bug_num[f"{pc.product}::{pc.component}"] = pc_mistossed_bug_num.get(f"{pc.product}::{pc.component}", 0)
        return pc_mistossed_bug_num

    def get_pc_mistossed_bug_dict(self, product_component_pair_list):
        pc_mistossed_bug_dict = dict()
        for bug in self.bugs:
            # print(f'https://bugzilla.mozilla.org/show_bug.cgi?id={bug.id}')
            for pc in bug.tossing_path.product_component_pair_list:
                if pc in product_component_pair_list and pc != bug.product_component_pair:
                    temp = pc_mistossed_bug_dict.get(pc, list())
                    temp.append(bug)
                    pc_mistossed_bug_dict[pc] = temp
            # print(pc_mistossed_bug_dict)
            # input()
        for pc in product_component_pair_list:
            if pc not in pc_mistossed_bug_dict.keys():
                temp = pc_mistossed_bug_dict.get(pc, list())
                pc_mistossed_bug_dict[pc] = temp
        return pc_mistossed_bug_dict

    def get_pc_mistossed_bug_summary_vec_dict(self, product_component_pair_list, VEC_TYPE):
        pc_mistossed_bug_dict = self.get_pc_mistossed_bug_dict(product_component_pair_list)
        pc_mistossed_bug_summary_vec_dict = dict()
        for pc in pc_mistossed_bug_dict.keys():
            matrix = list()

            if VEC_TYPE == 1:
                for bug in pc_mistossed_bug_dict[pc]:
                    matrix.append(bug.summary_mean_vec)
            elif VEC_TYPE == 2:
                for bug in pc_mistossed_bug_dict[pc]:
                    matrix.append(bug.summary_tfidf_vec)
            else:
                for bug in pc_mistossed_bug_dict[pc]:
                    matrix.append(bug.summary_onehot_vec)

            if len(matrix) == 0:
                # matrix.append(np.zeros(WORD2VEC_DIM))
                if VEC_TYPE != 3:
                    matrix.append(np.zeros(WORD2VEC_DIM))
                else:
                    matrix.append(sp.csr_matrix(np.zeros(ONEHOT_DIM)))  # maybe have some problem
            if VEC_TYPE != 3:
                matrix = np.array(matrix)
            else:
                matrix = sp.vstack(matrix)
            pc_mistossed_bug_summary_vec_dict[pc] = matrix

        return pc_mistossed_bug_summary_vec_dict

    def get_pc_mistossed_bug_description_vec_dict(self, product_component_pair_list, VEC_TYPE):
        pc_mistossed_bug_dict = self.get_pc_mistossed_bug_dict(product_component_pair_list)
        pc_mistossed_bug_description_vec_dict = dict()
        for pc in pc_mistossed_bug_dict.keys():
            matrix = list()

            if VEC_TYPE == 1:
                for bug in pc_mistossed_bug_dict[pc]:
                    matrix.append(bug.description_mean_vec)
            elif VEC_TYPE == 2:
                for bug in pc_mistossed_bug_dict[pc]:
                    matrix.append(bug.description_tfidf_vec)
            else:
                for bug in pc_mistossed_bug_dict[pc]:
                    matrix.append(bug.description_onehot_vec)

            if len(matrix) == 0:
                # matrix.append(np.zeros(WORD2VEC_DIM))
                if VEC_TYPE != 3:
                    matrix.append(np.zeros(WORD2VEC_DIM))
                else:
                    matrix.append(sp.csr_matrix(np.zeros(ONEHOT_DIM)))  # maybe have some problem
            if VEC_TYPE != 3:
                matrix = np.array(matrix)
            else:
                matrix = sp.vstack(matrix)
            pc_mistossed_bug_description_vec_dict[pc] = matrix

        return pc_mistossed_bug_description_vec_dict

    def overall_bugs(self):
        """
        统计bugs中每个product&component包含的bug个数、tossing bug个数 及 tossing path数
        :return:
        """
        p_c_pair_list = []
        p_c_pair_framework_list = []

        for bug in self.bugs:
            # bug = Bug.dict_to_object(bug)
            if bug.product_component_pair not in p_c_pair_list:
                p_c_pair_list.append(bug.product_component_pair)

                p_c_pair_framework = ProductComponentPairFramework()
                p_c_pair_framework.product_component_pair = bug.product_component_pair
                p_c_pair_framework.bug_nums = 1

                p_c_pair_framework.tossing_path_framework_list = []
                tossing_path_framework = TossingPathFramework()
                tossing_path_framework.tossing_path = bug.tossing_path
                tossing_path_framework.nums = 1
                tossing_path_framework.bug_id_list = []
                tossing_path_framework.bug_id_list.append(bug.id)
                p_c_pair_framework.tossing_path_framework_list.append(tossing_path_framework)
                p_c_pair_framework_list.append(p_c_pair_framework)
            else:
                for framework in p_c_pair_framework_list:
                    if bug.product_component_pair == framework.product_component_pair:
                        framework.bug_nums = framework.bug_nums + 1
                        i = 0
                        for tossing_path_framework in framework.tossing_path_framework_list:
                            if tossing_path_framework.tossing_path == bug.tossing_path:
                                tossing_path_framework.bug_id_list.append(bug.id)
                                tossing_path_framework.nums = tossing_path_framework.nums + 1
                                break
                            i = i + 1
                        if i == len(framework.tossing_path_framework_list):
                            tossing_path_framework = TossingPathFramework()
                            tossing_path_framework.tossing_path = bug.tossing_path
                            tossing_path_framework.bug_id_list = []
                            tossing_path_framework.bug_id_list.append(bug.id)
                            tossing_path_framework.nums = 1
                            framework.tossing_path_framework_list.append(tossing_path_framework)
                        break
        sum = 0
        sum_tossing = 0
        sum_tossing_path = 0
        for p_c_pair_framework in p_c_pair_framework_list:
            p_c_pair_framework.get_tossing_bug_nums()
            sum = sum + p_c_pair_framework.bug_nums
            sum_tossing = sum_tossing + p_c_pair_framework.tossing_bug_nums
            sum_tossing_path = sum_tossing_path + len(p_c_pair_framework.tossing_path_framework_list)
        # overall
        print(f'bug_nums: {sum}')
        print(f'tossing_bug_nums: {sum_tossing}')  # tossing bug nums
        print(f'tossing_path_nums: {sum_tossing_path}')  # tossing path nums
        print(f'product_component_nums: {len(p_c_pair_framework_list)}')
        for p_c_pair_framework in p_c_pair_framework_list:
            print(p_c_pair_framework.product_component_pair)
            # each of p&c
            print(f'bug_nums: {p_c_pair_framework.bug_nums}')
            print(f'tossing_bug_nums: {p_c_pair_framework.tossing_bug_nums}')
            print(f'tossing_path_nums: {len(p_c_pair_framework.tossing_path_framework_list)}')
        self.product_component_pair_framework_list = p_c_pair_framework_list
        # print(self.product_component_pair_framework_list)

    def get_nodes_edges_for_graph_actual_path(self):
        """
        A->B->C
        A->B
        :return:node_set=set(): A B C
                edge_set=set(): A->B (2) B->C (1)
        """
        pcpair_frequency_dict = dict()
        node_set = set()
        for bug in self.bugs:
            node_set.add(f'{bug.product_component_pair.product}::'
                         f'{bug.product_component_pair.component}')

            pc_list = bug.tossing_path.product_component_pair_list

            for i in range(0, bug.tossing_path.length - 1):
                pcpair = (pc_list[i], pc_list[i + 1])
                # dict中的get(key,0) means if 有这个key值，则返回value，else 返回 0
                pcpair_frequency_dict[pcpair] = pcpair_frequency_dict.get(pcpair, 0) + 1
        edge_set = set()
        for pcpair, freq in pcpair_frequency_dict.items():
            edge = Edge(f'{pcpair[0].product}::{pcpair[0].component}',
                        f'{pcpair[1].product}::{pcpair[1].component}', freq)
            edge_set.add(edge)
        return node_set, edge_set

    def get_nodes_edges_for_graph_goal_oriented_path(self, product_component_pair_list):
        """
        A->B->C
        A->B
        :return:node_set: A B C
                edge_set: A->C(1, 0.5) B->C (1, 1.0) A->B (1, 0.5)
                            (freq, transaction probability)
        """
        pcpair_frequency_dict = dict()
        node_dict = dict()
        for bug in self.bugs:
            pc_name = f'{bug.product_component_pair.product}::{bug.product_component_pair.component}'
            node_dict[pc_name] = node_dict.get(pc_name, 0) + 1

            pc_list = bug.tossing_path.product_component_pair_list
            final = bug.tossing_path.length - 1
            for i in range(0, final):
                if pc_list[i] == pc_list[final]:
                    continue
                pcpair = (pc_list[i], pc_list[final])
                # dict中的get(key,0) means if 有这个key值，则返回value，else 返回 0
                pcpair_frequency_dict[pcpair] = pcpair_frequency_dict.get(pcpair, 0) + 1
        pc_mistossed_bug_num = self.get_pc_mistossed_bug_num(product_component_pair_list)

        node_set = set()
        for node_name, bug_num in node_dict.items():
            node = MyNode(node_name, bug_num, pc_mistossed_bug_num[node_name])
            node_set.add(node)

        edge_set = set()
        for pcpair, freq in pcpair_frequency_dict.items():
            edge = Edge(MyNode(f'{pcpair[0].product}::{pcpair[0].component}'),
                        MyNode(f'{pcpair[1].product}::{pcpair[1].component}'), freq)
            edge_set.add(edge)
        Edge.get_probability(node_set, edge_set)
        return node_set, edge_set

    def get_bug_summary_list(self):
        """
        get bugs' summary
        :return: bug summary list
        """
        summary_list = []
        for bug in self.bugs:
            id_summary = {"id": f'https://bugzilla.mozilla.org/show_bug.cgi?id={bug.id}',
                          "summary": bug.summary}
            summary_list.append(id_summary)
        return summary_list

    def get_bug_summary_token_list(self):
        """
        get bugs' summary token
        :return: bug summary token list
        """
        summary_token_list = []
        for bug in self.bugs:
            summary_token_list.append(bug.summary_token)
        return summary_token_list

    def get_bug_description_token_list(self):
        """
        get bugs' summary token
        :return: bug summary token list
        """
        description_token_list = []
        for bug in self.bugs:
            description_token_list.append(bug.description_token)
        return description_token_list

    def get_pc_summary_token_list(self, product_component_pair_list):
        pc_summary_token_list = list()
        for pc in product_component_pair_list:
            # print(pc)
            bugs = self.get_specified_product_component_bugs(pc)
            one_pc_summary_token_list = list()
            for bug in bugs:
                one_pc_summary_token_list.extend(bug.summary_token)
            pc_summary_token_list.append(one_pc_summary_token_list)
            # print(len(one_pc_summary_token_list))
            # print(one_pc_summary_token_list)
            # input()
        return pc_summary_token_list

    def get_pc_description_token_list(self, product_component_pair_list):
        pc_description_token_list = list()
        for pc in product_component_pair_list:
            # print(pc)
            bugs = self.get_specified_product_component_bugs(pc)
            one_pc_description_token_list = list()
            for bug in bugs:
                # print(bug.summary_token)
                # print(bug.description_token)
                # one_pc_bug_description_token_list.extend(bug.summary_token)
                one_pc_description_token_list.extend(bug.description_token)
                # print(one_pc_summary_description_token_list)
                # input()
            pc_description_token_list.append(one_pc_description_token_list)
            # print(len(one_pc_summary_token_list))
            # print(one_pc_summary_token_list)
            # input()
        return pc_description_token_list

    def get_bug_summary_vec_matrix(self, VEC_TYPE):
        """
        get bugs' summary mean vec matrix
        :return: bug summary mean vec matrix
        """
        matrix = []
        if VEC_TYPE == 1:
            for bug in self.bugs:
                matrix.append(bug.summary_mean_vec)
        elif VEC_TYPE == 2:
            for bug in self.bugs:
                matrix.append(bug.summary_tfidf_vec)
        else:
            for bug in self.bugs:
                matrix.append(bug.summary_onehot_vec)
        if VEC_TYPE != 3:
            return np.array(matrix)
        else:
            return sp.vstack(matrix)

    def get_bug_concept_set_vec_matrix(self):
        matrix = []
        for bug in self.bugs:
            matrix.append(bug.summary_concept_set_vec)
        return sp.vstack(matrix)

    def get_bug_summary_onehot_vec_matrix(self):
        """
        get bugs' summary mean vec matrix
        :return: bug summary mean vec matrix
        """
        matrix = []

        for bug in self.bugs:
            matrix.append(bug.summary_onehot_vec)

        return np.array(matrix)

    def sort_by_creation_time(self):
        self.bugs = sorted(self.bugs, key=lambda x: x.creation_time, reverse=False)

    def split_dataset_by_creation_time(self):
        """
        sort bugs by creation time
        split bugs into
            80% training dataset
            20% testing dataset
        :return:
        """
        self.sort_by_creation_time()

        train_bugs = list()
        test_bugs = list()
        i = 0
        for bug in self.bugs:
            if i < 0.8 * self.get_length():
                train_bugs.append(bug)
            else:
                test_bugs.append(bug)
            i = i + 1
        train_bugs = Bugs(train_bugs)
        # train_bugs.overall_bugs()
        test_bugs = Bugs(test_bugs)
        # test_bugs.overall_bugs()
        return train_bugs, test_bugs

    def split_dataset_by_tossed_and_untossed(self):
        """

        :return:
        """
        tossed_bugs = list()
        untossed_bugs = list()
        for bug in self.bugs:
            if bug.tossing_path.length > 1:
                tossed_bugs.append(bug)
            else:
                untossed_bugs.append(bug)
        tossed_bugs = Bugs(tossed_bugs)
        untossed_bugs = Bugs(untossed_bugs)
        return tossed_bugs, untossed_bugs

    def split_dataset_by_pc(self, product_component_pair_list):
        """
        split bugs according to pc, for each pc: 80% training dataset & 20% testing dataset
        :param product_component_pair_list:
        :return:
        """
        train_bugs = list()
        test_bugs = list()

        for pc in product_component_pair_list:
            bugs = self.get_specified_product_component_bugs(pc)
            train_bugs.extend(list(bugs)[0: int(bugs.get_length() * 0.8)])
            test_bugs.extend(list(bugs)[int(bugs.get_length() * 0.8): bugs.get_length()])
        train_bugs = Bugs(train_bugs)
        # train_bugs.overall_bugs()
        test_bugs = Bugs(test_bugs)
        # test_bugs.overall_bugs()
        return train_bugs, test_bugs

    def split_dataset_by_pc_and_creation_time(self, product_component_pair_list):
        """
        sort bugs by creation time
        split bugs according to pc, for each pc: 80% training dataset & 20% testing dataset
        :param product_component_pair_list:
        :return:
        """
        self.sort_by_creation_time()

        train_bugs = list()
        test_bugs = list()

        for pc in product_component_pair_list:
            bugs = self.get_specified_product_component_bugs(pc)
            train_bugs.extend(list(bugs)[0: int(bugs.get_length() * 0.8)])
            test_bugs.extend(list(bugs)[int(bugs.get_length() * 0.8): bugs.get_length()])
        train_bugs = Bugs(train_bugs)
        # train_bugs.overall_bugs()
        test_bugs = Bugs(test_bugs)
        # test_bugs.overall_bugs()
        return train_bugs, test_bugs
