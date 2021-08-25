import string
from pathlib import Path
from re import finditer
import nltk
from gensim.summarization import bm25
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import gensim.downloader
from gensim.models import FastText

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local
# issuer certificate (_ssl.c:1076)
import ssl

from bug_tossing.utils.path_util import PathUtil
from config import WORD2VEC_DIM


class NLPUtil:
    @staticmethod
    def sentence_tokenize(paragraph):
        """
        分句
        :param paragraph:
        :return: sentences
        """
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenizer.tokenize(paragraph)
        return sentences

    @staticmethod
    def lemmatize(sentence):
        """
        分词 词性标注 词形还原
        :param sentence:
        :return:
        """
        wnl = nltk.WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag.startswith('NN'):
                yield wnl.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                yield wnl.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                yield wnl.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                yield wnl.lemmatize(word, pos='r')
            else:
                yield word
                # yield wnl.lemmatize(word)
        # print(word,tag)
        # for word in word_tokenize(sentence):
        #     yield wnl.lemmatize(word)

    @staticmethod
    def remove_stopwords(words):
        """
        去除停用词
        :param words:
        :return:
        """
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        return filtered_words

    @staticmethod
    def remove_punctuation(sentence):
        """
        去除标点符号
        :param sentence:
        :return:
        """
        # sentence_p = "".join([char for char in sentence if char not in string.punctuation])
        sentence_p = ""
        for char in sentence:
            if char not in string.punctuation:
                sentence_p = sentence_p + char
            else:
                sentence_p = sentence_p + ' '
        return sentence_p

    @staticmethod
    def remove_number(token_list):
        token_list = list(filter(lambda x: not str(x).isdigit(), token_list))
        return token_list

    @staticmethod
    def camel_case_split(identifier):
        """
        驼峰分词
        :param identifier:
        :return:
        """
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        words = ""
        for m in matches:
            words = words + " " + m.group(0)
        return words

    @staticmethod
    def filter_by_pos_tag(words):
        """
        only keep noun and verb
        :param words:
        :return:
        """
        filtered_words = []
        for word, tag in pos_tag(words):
            if tag.startswith('NN') or tag.startswith('VB'):
                filtered_words.append(word)
        return filtered_words

    @staticmethod
    def filter_paragraph_by_pos_tag(paragraph):
        """
        in a paragraph, only keep noun and verb
        :param paragraph:
        :return:
        """
        filtered_paragraph = list()
        for sentence in paragraph:
            filtered_paragraph.append(NLPUtil.filter_by_pos_tag(sentence))
        return filtered_paragraph

    @staticmethod
    def preprocess(paragraph):
        """
        预处理
        1. 驼峰
        2. caselower
        3. sentence split
        4. remove punctuations
        5. 分词 词性标注 词形还原
        6. remove stopword
        7. remove number
        :param paragraph:
        :return:
        """
        # 去掉回车，换成空格
        paragraph = paragraph.replace('\n', ' ')

        # 驼峰
        paragraph = NLPUtil.camel_case_split(paragraph)

        # 变成小写表示
        paragraph = paragraph.lower()
        # 分句
        sentences = NLPUtil.sentence_tokenize(paragraph)
        filtered_words_list = []
        for sentence in sentences:
            # print(sentence)
            # 去标点符号
            sentence_p = NLPUtil.remove_punctuation(sentence)
            # print(sentence_p)
            # 分词 词性标注 词形还原
            words = NLPUtil.lemmatize(sentence_p)
            # 去除停词
            filtered_words = NLPUtil.remove_stopwords(words)
            filtered_words = NLPUtil.remove_number(filtered_words)
            for fword in filtered_words:
                # print(fword)
                filtered_words_list.append(fword)
        return filtered_words_list

    @staticmethod
    def load_word2vec_model(model_name):
        """
        https://radimrehurek.com/gensim/models/word2vec.html

        How to deal with multi-word phrases(or n-grams) while building a custom embedding?
        https://suyashkhare619.medium.com/how-to-deal-with-multi-word-phrases-or-n-grams-while-building-a-custom-embedding-eec547d1ab45
        :param model_name: word2vec-google-news-300
        :return:
        """
        # ssl._create_default_https_context = ssl._create_unverified_context
        wv = gensim.downloader.load(model_name)  # 'word2vec-google-news-300'
        return wv

    @staticmethod
    def load_fasttext_model(model_name):
        """
        https://fasttext.cc/docs/en/pretrained-vectors.html
        :param model_name:
        :return:
        """
        model = FastText.load_fasttext_format(model_name)  # "wiki.en.bin"
        return model

    @staticmethod
    def convert_word_to_vector(model, word):
        """

        :param word:
        :param model:
        :return:
        """
        try:
            word_vec = model[word]
        except Exception:
            word_vec = np.zeros(WORD2VEC_DIM)
        return word_vec
        # if word in model:
        #     word_vec = model[word]
        # else:
        #     word_vec = np.zeros(WORD2VEC_DIM)
        # return word_vec

    # @staticmethod
    # def get_tfidf_vectorizer(corpus):
    #     """
    #     corpus = [
    #         'This is the first document.',
    #         'This document is the second document.',
    #         'And this is the third one.',
    #         'Is this the first document?',
    #     ]
    #     :param corpus:
    #     :return:
    #     """
    #     vectorizer = TfidfVectorizer()
    #     X = vectorizer.fit_transform(corpus)
    #     return vectorizer, X

    @staticmethod
    def get_text_similarity_by_word_mover_distance(model, sentence1, sentence2):
        distance = model.wmdistance(sentence1, sentence2)
        return distance

    @staticmethod
    def count_word_in_sentence(words):
        word_count_dict = dict()
        for word in words:
            word_count_dict[word] = word_count_dict.get(word, 0) + 1
        return word_count_dict

    @staticmethod
    def get_word_idf_dict(corpus):
        # word_idf = dict()
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(corpus)
        max_idf = max(tfidf.idf_)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        word_idf = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])

        return word_idf

    @staticmethod
    def get_word_tfidf_dict(document, word_idf):

        word_tfidf = dict()

        word_count_dict = NLPUtil.count_word_in_sentence(document)
        sent_len = len(document)
        tfidf_list = list()
        for w in word_count_dict.keys():
            tfidf = word_idf[w] * word_count_dict[w] / sent_len
            tfidf_list.append(tfidf)
        tfidf_array = np.array(tfidf_list)
        square = np.einsum('i, i->i', tfidf_array, tfidf_array)
        sum_all = np.einsum('i->', square)
        # print(sum)
        norms = np.sqrt(sum_all)

        for index, w in enumerate(word_count_dict.keys()):
            word_tfidf[w] = tfidf_array[index] / norms
        return word_tfidf


# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb
# class MeanEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         # if a text is empty we should return a vector of zeros with the same dimensionality as all the other vectors
#         # self.dim = len(word2vec.itervalues().next()) #AttributeError: 'Word2VecKeyedVectors' object has no
#         # attribute 'itervalues'
#         self.dim = WORD2VEC_DIM  # ******需要根据模型变化 动态更改******
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X):
#         return np.array([
#             np.mean([self.word2vec[w] for w in words if w in self.word2vec]
#                     or [np.zeros(self.dim)], axis=0)
#             for words in X
#         ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None  # use smooth idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1 ps: 以2为底
        self.dim = WORD2VEC_DIM

    def __repr__(self):
        return f'{self.word2vec}::{self.word2weight}'

    def __str__(self):
        return f'{self.word2vec}::{self.word2weight}'

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    # def transform(self, X):
    #     """
    #     sentence归一化
    #     tfidf值标准化，norm=‘l2’，它的意思是，把我们计算的结果做个调整，标准化
    #     :param X:
    #     :return:
    #     """
    #     sent_vec_list = list()
    #     for words in X:
    #         word_count_dict = NLPUtil.count_word_in_sentence(words)
    #         sent_len = len(words)
    #         weight_sum = 0
    #         sent_vec = np.zeros(self.dim)
    #         # print(words)
    #         # print(word_count_dict)
    #         tfidf_list = list()
    #         for w in word_count_dict.keys():
    #             tfidf = self.word2weight[w] * word_count_dict[w]/sent_len
    #             tfidf_list.append(tfidf)
    #         tfidf_array = np.array(tfidf_list)
    #         square = np.einsum('i, i->i', tfidf_array, tfidf_array)
    #         sum_all = np.einsum('i->', square)
    #         # print(sum)
    #         norms = np.sqrt(sum_all)
    #
    #         for index, w in enumerate(word_count_dict.keys()):
    #             tfidf = tfidf_array[index] / norms
    #             # print(f'{w} idf:{self.word2weight[w]} tf:{word_count_dict[w]/sent_len} '
    #             #       f'tfidf:{tfidf} word2vec: ')
    #
    #             try:
    #                 word_vec = self.word2vec[w]
    #             except Exception:
    #                 word_vec = np.zeros(WORD2VEC_DIM)
    #             # print(word_vec)
    #             sent_vec = sent_vec + word_vec * tfidf
    #
    #             weight_sum = weight_sum + tfidf
    #         if weight_sum != 0:
    #             sent_vec = sent_vec / weight_sum
    #         sent_vec_list.append(sent_vec)
    #     return np.array(sent_vec_list)

    # def transform(self, X):
    #     """
    #     归一化
    #     未标准化：norm=‘l2’，它的意思是，把我们计算的结果做个调整，标准化，
    #                 只影响word的tfidf值，对sentence的word2vec tfidf加权表示 无影响
    #     :param X:
    #     :return:
    #     """
    #     sent_vec_list = list()
    #     for words in X:
    #         word_count_dict = NLPUtil.count_word_in_sentence(words)
    #         sent_len = len(words)
    #         weight_sum = 0
    #         sent_vec = np.zeros(self.dim)
    #         # print(words)
    #         # print(word_count_dict)
    #
    #         for w in word_count_dict.keys():
    #             tfidf = self.word2weight[w] * word_count_dict[w]/sent_len
    #             # print(f'{w} idf:{self.word2weight[w]} tf:{word_count_dict[w]/sent_len} '
    #             #       f'tfidf:{tfidf} word2vec: ')
    #
    #             try:
    #                 word_vec = self.word2vec[w]
    #             except Exception:
    #                 word_vec = np.zeros(WORD2VEC_DIM)
    #             # print(word_vec)
    #             sent_vec = sent_vec + word_vec * tfidf
    #             weight_sum = weight_sum + tfidf
    #         if weight_sum != 0:
    #             sent_vec = sent_vec / weight_sum
    #         sent_vec_list.append(sent_vec)
    #     return np.array(sent_vec_list)

    def transform(self, corpus):
        # 未归一化
        # 未标准化
        # return np.array([
        #     np.mean([self.word2vec[w] * self.word2weight[w]
        #              for w in words if w in self.word2vec] or
        #             [np.zeros(self.dim)], axis=0)
        #     for words in X
        # ])
        words_list = list()
        # i = 0
        for words in corpus:
            word_list = list()
            for w in words:
                try:
                    word_vec = self.word2vec[w]
                except Exception:
                    word_vec = np.zeros(WORD2VEC_DIM)
                word_list.append(word_vec * self.word2weight[w])

                # if i == 0 or i == 9:
                #     print(f'{w}: {self.word2weight[w]}')
                #     print(word_vec)

            if len(words) == 0:
                word_array = np.zeros(WORD2VEC_DIM)
            else:
                word_array = np.mean(np.array(word_list), axis=0)
            # i = i + 1

            words_list.append(word_array)
        return np.array(words_list)


class TfidfOnehotVectorizer():
    def __init__(self):
        # key=word, value=(index, idf, tf in all documents(单词在所有文档中出现的总次数))
        self.max_idf = None
        self.min_idf = None
        self.min_tf = 1
        self.word2index_weight_pair = None  # use smooth idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1 ps: 以2为底
        self.dim = None

    def __repr__(self):
        return f'{self.word2index_weight_pair}'

    def __str__(self):
        return f'{self.word2index_weight_pair}'

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        merge_X = list()
        for one in X:
            merge_X.extend(one)
        word_count_dict = NLPUtil.count_word_in_sentence(merge_X)

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        self.dim = len(tfidf.vocabulary_.items())
        self.max_idf = max(tfidf.idf_)
        self.min_idf = min(tfidf.idf_)
        self.word2index_weight_pair = defaultdict(
            lambda: self.max_idf,
            [(w, (i, tfidf.idf_[i], word_count_dict.get(w, 1))) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, corpus):
        words_list = list()
        for words in corpus:
            word_array = np.zeros(self.dim)
            for w in words:
                if w in self.word2index_weight_pair.keys():
                    word_array[self.word2index_weight_pair[w][0]] = self.word2index_weight_pair[w][1]

                # if i == 0 or i == 9:
                #     print(f'{w}: {self.word2weight[w]}')
                #     print(word_vec)

            # i = i + 1

            words_list.append(word_array)
        return np.array(words_list)


class BM25:
    def __init__(self, corpus=None):
        self.bm25_model = bm25.BM25(corpus)

    def fit(self, corpus):
        self.bm25_model = bm25.BM25(corpus)

    def transform(self, query):
        scores = self.bm25_model.get_scores(query)

        return scores
