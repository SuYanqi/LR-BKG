from bug_tossing.utils.nlp_util import TfidfOnehotVectorizer
import numpy as np


class ConceptSet:
    def __init__(self):
        self.max_idf = None
        self.min_idf = 1.0
        self.min_tf = 3
        self.unique_concept_set = dict()
        self.common_concept_set = dict()
        self.controversial_concept_set = dict()
        self.concept_set = dict()
        self.community_concept_set = dict()
        self.word_index_dict = dict()  # unique and controversial only

    def get_word_index_dict(self):
        index = 0
        for word in self.unique_concept_set.keys():
            self.word_index_dict[word] = index
            index = index + 1
        # for word in self.controversial_concept_set.keys():
        #     self.word_index_dict[word] = index
        #     index = index + 1

    @staticmethod
    def get_all_concept_set(corpus):
        onehot = TfidfOnehotVectorizer()
        onehot.fit(corpus)
        all_concept_set = onehot.word2index_weight_pair
        return all_concept_set

    def get_community_concept_set(self, product_component_pair_list):
        for pc in product_component_pair_list:
            self.community_concept_set[pc.community] = dict(
                self.community_concept_set.get(pc.community, dict()),
                **pc.concept_set)

    def extract_concept_set(self, corpus):
        onehot = TfidfOnehotVectorizer()
        onehot.fit(corpus)
        all_concept_set = onehot.word2index_weight_pair
        self.max_idf = onehot.max_idf
        # self.min_tf = onehot.min_tf
        # self.min_idf = onehot.min_idf
        # self.concept_set = {k: v for k, v in all_concept_set.items()
        #                     if v[2] != onehot.min_tf}
        index = 0
        for k, v in all_concept_set.items():
            if v[2] >= self.min_tf and v[1] >= self.min_idf:
                self.concept_set[k] = (index, v[1], v[2])
                index = index + 1
        # print(self.concept_set)
        self.get_unique_concept_set(onehot.max_idf)

        # self.concept_set = sorted(onehot.word2index_weight_pair.items(), key=lambda d: (d[1][1], d[1][2]),
        # reverse=True)

    def get_unique_concept_set(self, max_idf):
        self.unique_concept_set = {k: v for k, v in self.concept_set.items()
                                   if v[1] == max_idf}
        # return self.unique_concept_set

    def get_common_controversial_concept_set(self):
        for word in self.concept_set.keys():
            is_common = 0
            for community in self.community_concept_set.keys():
                if word in self.community_concept_set[community].keys():
                    is_common = is_common + 1
                if is_common == 2:
                    self.common_concept_set[word] = self.concept_set[word]
                    break
            if is_common == 1 and word not in self.unique_concept_set.keys():
                self.controversial_concept_set[word] = self.concept_set[word]

    def transform(self, corpus):
        words_list = list()
        for words in corpus:
            word_array = np.zeros(len(self.concept_set))
            for w in words:
                if w in self.concept_set.keys():
                    word_array[self.concept_set[w][0]] = self.concept_set[w][1]  # pc unit idf
            words_list.append(word_array)
        return np.array(words_list)

    def transform_uncommon(self, corpus):
        words_list = list()
        for words in corpus:
            word_array = np.zeros(len(self.word_index_dict))
            for w in words:
                if w in self.word_index_dict.keys():
                    word_array[self.word_index_dict[w]] = self.concept_set[w][1]  # pc unit idf
            words_list.append(word_array)
        return np.array(words_list)
