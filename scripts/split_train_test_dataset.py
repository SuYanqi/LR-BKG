from bug_tossing.types.bugs import Bugs
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil


if __name__ == "__main__":
    """
    sort bugs by creation time
    and split bugs into training set and testing set by different ways
    80% training dataset
    20% testing dataset
    """
    filtered_bugs_filepath = PathUtil.get_filtered_bugs_filepath()
    bugs = FileUtil.load_pickle(filtered_bugs_filepath)

    train_bugs, test_bugs = bugs.split_dataset_by_creation_time()

    # pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    # train_bugs, test_bugs = bugs.split_dataset_by_pc_and_creation_time(pc_list)
    # train_bugs, test_bugs = bugs.split_dataset_by_pc(pc_list)

    train_bugs.overall_bugs()
    test_bugs.overall_bugs()
    print(train_bugs.get_length())
    print(test_bugs.get_length())

    train_bugs_filepath = PathUtil.get_train_bugs_filepath()
    FileUtil.dump_pickle(train_bugs_filepath, train_bugs)
    test_bugs_filepath = PathUtil.get_test_bugs_filepath()
    FileUtil.dump_pickle(test_bugs_filepath, test_bugs)


