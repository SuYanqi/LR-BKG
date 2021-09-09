import re
from pathlib import Path

from scipy import sparse

from bug_tossing.product_component_assignment.learning_to_rank.lambdaMART import LambdaMART
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil

import xgboost as xgb
from config import FEATURE_VECTOR_DIR

# def get_bug_id(line):
#     res = re.findall(r'qid:\d+', line)
#     res = re.findall(r'\d+', res[0])
#     return res[0]


# 18328
if __name__ == "__main__":
    model = LambdaMART.train_model(Path(FEATURE_VECTOR_DIR, "train_graph_feature_vector_for_ablation"))

    # model = LambdaMART.train_model(Path(FEATURE_VECTOR_DIR, "train_top_30_tfidf_onehot_percentage_graph"))
    # model = LambdaMART.train_model(Path(FEATURE_VECTOR_DIR, "train_top_30_tfidf_onehot_percentage_concept_set_onehot_tfidf"))
    print('model done')
    # output_feature_vector_dir = Path(FEATURE_VECTOR_DIR, "train_concept_set")
    # x_train_sparse = sparse.load_npz(Path(str(output_feature_vector_dir), "x_sparse.npz"))  # 读取
    # y_train = FileUtil.load_pickle(Path(str(output_feature_vector_dir), "y.npz"))  # 读取
    # param = {'max_depth': 8, 'eta': 0.3,
    #          # 'silent': 0,
    #          'objective': 'rank:pairwise',
    #          'num_boost_round': 10
    #          # 'verbosity': 2,
    #          # 'eval_metric': 'ndcg@1'
    #          }
    # lambda_mart_model = xgb.train(param, xgb.DMatrix(x_train_sparse, y_train))
    #
    # lambda_mart_model.save_model(PathUtil.load_lambdaMart_model_filepath())
    # print('model done')
