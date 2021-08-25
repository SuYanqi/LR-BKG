import numpy as np
from numpy import sqrt, sum
from sklearn.metrics import pairwise_distances

# import tensorflow as tf
from numpy.dual import norm
import scipy.spatial as sp
from sklearn.metrics.pairwise import cosine_similarity


class MatrixUtil:

    @staticmethod
    def get_3_d_matrix(vec_list):
        matrix = []
        for vec in vec_list:
            vec = np.mean(vec, 0)  # Summary_vec可能由多个向量构成时，求平均
            # print(vec.shape)
            matrix.append(vec.reshape(1, 300))
        return np.array(matrix)

    @staticmethod
    def get_summary_vec_matrix(bug_list):
        """
        将bug_list中的summary向量取出构成 矩阵，以便用于 矩阵加速
        :param bug_list:
        :return:
        """
        summary_vec_list = []
        for bug in bug_list:
            vec = np.mean(bug.summary_mean_vec, 0)  # Summary_vec可能由多个向量构成时，求平均
            summary_vec_list.append(vec)
        return np.array(summary_vec_list)  # 用mat得到的也是一个矩阵，还需要降维成array

    @staticmethod
    def get_pro_com_pair_vec_matrix(pair_list):
        pair_vec_list = []
        for i in range(0, min(10, len(pair_list))):
            vec = np.mean(pair_list[i].product_component_pair_mean_vec, 0)  # Summary_vec可能由多个向量构成时，求平均
            # print(vec.shape)
            pair_vec_list.append(vec)
            i = i + 1
        while (i < 10):
            pair_vec_list.append(np.zeros(300))
            i = i + 1
        pair_vec_list = np.array(pair_vec_list)
        pair_vec_list.reshape(10, 300)
        return pair_vec_list  # 用mat得到的也是一个矩阵，还需要降维成array

    @staticmethod
    def get_vec_matrix(vec_list):
        """
        :param vec_list: 向量列表
        :return: 矩阵
        """
        return np.array(vec_list)  # 用mat得到的也是一个矩阵，还需要降维成array

    @staticmethod
    def cos_vector_vector(vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denom == 0:
            return 0.0
        sim = num / denom
        return sim

    @staticmethod
    def cos_vector_matrix(arr, brr):
        """
        向量与矩阵的余弦相似度, 返回一个matrix，需要使用.flatten()降维成vector
        :param arr:
        :param brr:
        :return:
        """
        cos = arr.dot(brr.T) / (sqrt(sum(arr * arr)) * sqrt(sum(brr * brr, axis=1)))
        cos[np.isnan(cos)] = 0
        return cos

    @staticmethod
    def cos_matrix_matrix(a, b):
        """
        both sparse matrix and dense matrix are OK
        :param a:
        :param b:
        :return:
        """
        # cos = 1 - sp.distance.cdist(a, b, 'cosine')
        # # 将区间从[-1，1] -> [0, 1]
        # # cos = (cos + 1)/2
        cos = 1 - pairwise_distances(a, b, metric="cosine")
        return cos

    # @staticmethod
    # def cos_tensor_tensor(a, b):
    #     """
    #     Note that it is a number between -1 and 1.
    #     When it is a negative number between -1 and 0,
    #     0 indicates orthogonality and values closer to -1 indicate greater similarity.
    #     The values closer to 1 indicate greater dissimilarity.
    #     :param b:
    #     :param a:
    #     :return:
    #     """
    #     cos = tf.keras.losses.cosine_similarity(a, b)
    #     # cos = tf.matmul(a, b)
    #     # cos[np.isnan(cos)] = 0
    #     return -cos.numpy()  # Get value out of torch.cuda.float tensor
