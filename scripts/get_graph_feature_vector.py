from pathlib import Path

from bug_tossing.product_component_assignment.feature_extraction.feature_vector import FeatureExtractor
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil
from config import FEATURE_VECTOR_DIR

if __name__ == "__main__":
    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())

    test_bugs = FileUtil.load_pickle(PathUtil.get_test_bugs_filepath())

    FeatureExtractor.get_graph_feature_vector(pc_list, train_bugs,
                                              Path(FEATURE_VECTOR_DIR,
                                                   f"train_graph_feature_vector"),
                                              )

    FeatureExtractor.get_graph_feature_vector(pc_list, test_bugs,
                                              Path(FEATURE_VECTOR_DIR,
                                                   f"test_graph_feature_vector"),
                                              )
