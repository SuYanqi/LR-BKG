import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


class MetricsUtil:

    @staticmethod
    def accuracy(result_list):
        # print(len(result_list))
        accuracy = dict()
        accuracy[1] = accuracy.get(1, 0)
        accuracy[3] = accuracy.get(3, 0)
        accuracy[5] = accuracy.get(5, 0)
        accuracy[10] = accuracy.get(10, 0)
        for result in result_list:
            # print(result)
            indexs = np.nonzero(result)[0]

            for index in indexs:
                # print(index)
                if index == 0:
                    accuracy[1] = accuracy.get(1, 0) + 1
                    accuracy[3] = accuracy.get(3, 0) + 1
                    accuracy[5] = accuracy.get(5, 0) + 1
                    accuracy[10] = accuracy.get(10, 0) + 1
                    break
                elif 0 < index < 3:
                    accuracy[3] = accuracy.get(3, 0) + 1
                    accuracy[5] = accuracy.get(5, 0) + 1
                    accuracy[10] = accuracy.get(10, 0) + 1
                    break
                elif 2 < index < 5:
                    accuracy[5] = accuracy.get(5, 0) + 1
                    accuracy[10] = accuracy.get(10, 0) + 1
                    break
                elif 4 < index < 10:
                    accuracy[10] = accuracy.get(10, 0) + 1
                    break
        for key in accuracy.keys():
            if len(result_list) == 0:
                accuracy[key] = 0
            else:
                accuracy[key] = accuracy[key] / len(result_list)
        return accuracy

    @staticmethod
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.

    @staticmethod
    def ndcg_at_k(r, k):
        # r1 = [1, 1, 1, 1, 1]
        idcg = MetricsUtil.dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 0.
        return MetricsUtil.dcg_at_k(r, k) / idcg

    @staticmethod
    def ndcg(result_list):
        average_ndcg = dict()
        for result in result_list:
            result = list(result)
            ndcg = dict()
            ndcg[1] = ndcg.get(1, MetricsUtil.ndcg_at_k(result, k=1))
            ndcg[3] = ndcg.get(3, MetricsUtil.ndcg_at_k(result, k=3))
            ndcg[5] = ndcg.get(5, MetricsUtil.ndcg_at_k(result, k=5))
            ndcg[10] = ndcg.get(10, MetricsUtil.ndcg_at_k(result, k=10))
            for key in ndcg.keys():
                average_ndcg[key] = average_ndcg.get(key, 0) + ndcg[key]
        if len(result_list) == 0:
            average_ndcg[1] = average_ndcg.get(1, 0)
            average_ndcg[3] = average_ndcg.get(3, 0)
            average_ndcg[5] = average_ndcg.get(5, 0)
            average_ndcg[10] = average_ndcg.get(10, 0)

        for key in average_ndcg.keys():
            if len(result_list) == 0:
                average_ndcg[key] = 0
            else:
                average_ndcg[key] = average_ndcg[key]/len(result_list)

        return average_ndcg

    @staticmethod
    def mrr(y_true, y_score):
        """
        If there is exactly one relevant label per sample,
        label ranking average precision is equivalent to the mean reciprocal rank

        y_true is the ground truth
        y_score is the score predicted
        """
        return label_ranking_average_precision_score(y_true, y_score)



