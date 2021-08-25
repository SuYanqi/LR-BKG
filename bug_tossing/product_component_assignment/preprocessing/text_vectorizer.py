# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# class MeanEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         # if a text is empty we should return a vector of zeros with the same dimensionality as all the other vectors
#         # self.dim = len(word2vec.itervalues().next()) #AttributeError: 'Word2VecKeyedVectors' object has no
#         # attribute 'itervalues'
#         self.dim = 300  # ******需要根据模型变化 动态更改******
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
#
#
# class TfidfEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         self.word2weight = None
#         self.dim = len(word2vec.itervalues().next())
#
#     def fit(self, X, y):
#         tfidf = TfidfVectorizer(analyzer=lambda x: x)
#         tfidf.fit(X)
#         # if a word was never seen - it must be at least as infrequent
#         # as any of the known words - so the default idf is the max of
#         # known idf's
#         max_idf = max(tfidf.idf_)
#         self.word2weight = defaultdict(
#             lambda: max_idf,
#             [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
#
#         return self
#
#     def transform(self, X):
#         return np.array([
#             np.mean([self.word2vec[w] * self.word2weight[w]
#                      for w in words if w in self.word2vec] or
#                     [np.zeros(self.dim)], axis=0)
#             for words in X
#         ])
