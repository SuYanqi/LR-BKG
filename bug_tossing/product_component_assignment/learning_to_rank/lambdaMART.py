import xgboost as xgb
import joblib
from matplotlib import pyplot
from pathlib import Path

from sklearn.datasets import load_svmlight_file
from tqdm import tqdm

from bug_tossing.utils.path_util import PathUtil
from config import FEATURE_VECTOR_DIR


class LambdaMART:
    def __init__(self):
        pass

    # @staticmethod
    # def train_model(train_dir):
    #     lambda_mart_model = None
    #     param = {'max_depth': 6, 'eta': 0.3,
    #              # 'silent': 0,
    #              'objective': 'rank:pairwise',
    #              # 'verbosity': 2,
    #              # 'eval_metric': 'ndcg@1-'
    #              }
    #
    #     # valid_sets = [(xgb.DMatrix(str(Path(FEATURE_VECTOR_DIR, "test_top_30_tfidf_onehot_percentage") /
    #     #                                "feature_vector_0.test")), 'ndcg@1-')]
    #     for train_file in tqdm(train_dir.glob("*.train"), ascii=True):
    #         x_train, y_train = load_svmlight_file(str(train_file))
    #         group_train = []
    #         with open(f"{str(train_file)}.group", "r") as f:
    #             data = f.readlines()
    #             for line in data:
    #                 group_train.append(int(line.split("\n")[0]))
    #
    #         train_dmatrix = DMatrix(x_train, y_train)
    #         train_dmatrix.set_group(group_train)
    #
    #         lambda_mart_model = xgb.train(param, train_dmatrix,
    #                                       num_boost_round=10, xgb_model=lambda_mart_model,
    #                                       )
    #                                       # evals=valid_sets)
    #
    #     lambda_mart_model.save_model(PathUtil.load_lambdaMart_model_filepath())
    #     # save model
    #     # lambda_mart_model.save_model(PathUtil.load_lambdaMart_model_filepath())
    #     # joblib.dump(lambda_mart_model, PathUtil.load_lambdaMart_model_filepath())
    #     return lambda_mart_model

    @staticmethod
    def train_model(train_dir):
        lambda_mart_model = None
        param = {'max_depth': 6, 'eta': 0.3,
                 # 'silent': 0,
                 'objective': 'rank:pairwise',
                 # 'verbosity': 2,
                 # 'eval_metric': 'ndcg@1-'
                 }

        # valid_sets = [(xgb.DMatrix(str(Path(FEATURE_VECTOR_DIR, "test_top_30_tfidf_onehot_percentage") /
        #                                "feature_vector_0.txt")), 'ndcg@1-')]
        # for _ in tqdm(range(100), ascii=True):
        for train_file in tqdm(train_dir.glob("*.txt"), ascii=True):
            # print(train_file)
            lambda_mart_model = xgb.train(param, xgb.DMatrix(str(train_file)),
                                          num_boost_round=10, xgb_model=lambda_mart_model,
                                          )
                                          # evals=valid_sets)

        lambda_mart_model.save_model(PathUtil.load_lambdaMart_model_filepath())
        # save model
        # lambda_mart_model.save_model(PathUtil.load_lambdaMart_model_filepath())
        # joblib.dump(lambda_mart_model, PathUtil.load_lambdaMart_model_filepath())
        return lambda_mart_model

    # @staticmethod
    # def test_model(model_filepath, test_dir):
    #     """
    #
    #     :param model_filepath:
    #     :param test_dir:
    #     :return:
    #     """
    #     lambda_mart_model = Booster()
    #     # load saved model
    #     lambda_mart_model.load_model(model_filepath)
    #     preds = list()
    #     n = len(list(test_dir.glob("*.test")))
    #     for k in tqdm(range(0, n), ascii=True):
    #         test_file = Path(test_dir, f'feature_vector_{k}.test')
    #         x_test, y_test = load_svmlight_file(str(test_file))
    #         # group_test = []
    #         # with open(f"{test_file}.group", "r") as f:
    #         #     data = f.readlines()
    #         #     for line in data:
    #         #         group_test.append(int(line.split("\n")[0]))
    #         test_dmatrix = DMatrix(x_test)
    #         # test_dmatrix.set_group(group_test)
    #         preds.extend(lambda_mart_model.predict(test_dmatrix))
    #     return preds

    @staticmethod
    def test_model(model_filepath, test_dir):
        """

        :param model_filepath:
        :param test_dir:
        :return:
        """
        lambda_mart_model = xgb.Booster()
        # load saved model
        lambda_mart_model.load_model(model_filepath)
        preds = list()
        n = len(list(test_dir.glob("*.txt")))
        for k in tqdm(range(0, n), ascii=True):
            test_file = Path(test_dir, f'feature_vector_{k}.txt')
            # print(test_file)
            testing_data = xgb.DMatrix(test_file)
            preds.extend(lambda_mart_model.predict(testing_data))
        return preds

    @staticmethod
    def plot_model(model_filepath):
        # load saved model
        model = joblib.load(model_filepath)
        # xgb.plot_tree(model, num_trees=3)
        # fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(150, 100)

        ax = xgb.plot_importance(model, color='red')
        fig = ax.figure
        fig.set_size_inches(10, 10)
        pyplot.show()

# testing_data = xgb.DMatrix(PathUtil.get_feature_vector_test_filepath())
#
# xgb.plot_tree(model, num_trees=3)
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(150, 100)
#
# ax = xgb.plot_importance(model, color='red')
# fig = ax.figure
# fig.set_size_inches(20, 20)
#
# preds = model.predict(testing_data)
#
# print(preds)
