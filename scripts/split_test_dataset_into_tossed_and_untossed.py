from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil

if __name__ == "__main__":

    test_bugs_filepath = PathUtil.get_test_bugs_filepath()
    test_bugs = FileUtil.load_pickle(test_bugs_filepath)
    tossed_bugs, untossed_bugs = test_bugs.split_dataset_by_tossed_and_untossed()
    # tossed_bugs.overall_bugs()
    # untossed_bugs.overall_bugs()
    FileUtil.dump_pickle(PathUtil.get_tossed_test_bugs_filepath(), tossed_bugs)
    FileUtil.dump_pickle(PathUtil.get_untossed_test_bugs_filepath(), untossed_bugs)

