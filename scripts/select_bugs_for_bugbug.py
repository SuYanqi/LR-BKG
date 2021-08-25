from pathlib import Path

from jsonlines import jsonlines
from tqdm import tqdm

from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil
from bugbug import bugzilla
from config import DATA_DIR


def select_same_with_ours(our_bugs, filename):
    """
    1. 由于bugbug需要使用dict[]数据，因此，从原本的数据中split出bugs的字典
    2. bugbug使用的训练集是使用jsonlines存储的，因此使用jsonlines存入文件
    """

    # 从 BUGS_DB = "data/bugs.json"中读取bugs -> Iterator[BugDict]
    bugs = bugzilla.get_bugs()

    bug_list = []

    for bug in tqdm(bugs, ascii=True):
        # k = 0
        for our_bug in our_bugs:
            if bug['id'] == our_bug.id:
                bug_list.append(bug)
                # k = 1
                break

    print(len(bug_list))

    with jsonlines.open(Path(DATA_DIR, "bugbug", filename), mode='w') as writer:
        writer.write_all(bug_list)


def select_train_bugs_for_bugbug():
    """
    由于bugbug的一些预处理方面的问题，因此我们只除去test dataset中的bugs，剩余部分均作为 training dataset
    1. 由于bugbug需要使用dict[]数据，因此，从原本的数据中split出train_bugs的字典
    2. bugbug使用的训练集是使用jsonlines存储的，因此使用jsonlines存入文件
    """

    # 从 BUGS_DB = "data/bugs.json"中读取bugs -> Iterator[BugDict]
    bugs = bugzilla.get_bugs()

    test_bugs = FileUtil.load_pickle(Path(DATA_DIR, "without_description", "test_bugs.json"))

    train_bug_list = []

    for bug in tqdm(bugs, ascii=True):
        k = 0
        for test_bug in test_bugs:
            if bug['id'] == test_bug.id:
                k = 1
                break
        if k == 0:
            train_bug_list.append(bug)

    print(len(train_bug_list))

    with jsonlines.open(Path(DATA_DIR, "bugbug", "train_bugs.json"), mode='w') as writer:
        # writer.write(...)
        writer.write_all(train_bug_list)


if __name__ == "__main__":
    test_bugs = FileUtil.load_pickle(PathUtil.get_test_bugs_filepath())
    select_same_with_ours(test_bugs, "test_bugs.json")
    # tossed_test_bugs = FileUtil.load_pickle(PathUtil.get_tossed_test_bugs_filepath())
    # select_same_with_ours(tossed_test_bugs, "tossed_test_bugs.json")
    # untossed_test_bugs = FileUtil.load_pickle(PathUtil.get_untossed_test_bugs_filepath())
    # select_same_with_ours(untossed_test_bugs, "untossed_test_bugs.json")
