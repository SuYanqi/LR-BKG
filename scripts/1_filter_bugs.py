from bugbug import bugzilla
from tqdm import tqdm

from bug_tossing.types.bug import Bug
from bug_tossing.types.bugs import Bugs
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil


def filter_bugs_by_pc():
    product_component_pair_filepath = PathUtil.get_pc_filepath()
    pc_dataset = FileUtil.load_pickle(product_component_pair_filepath)
    print(len(pc_dataset))
    print(pc_dataset)

    # 从 BUGS_DB = "data/bugs.json"中读取bugs -> Iterator[BugDict]
    bugs = bugzilla.get_bugs()

    bug_list = []

    for bug in bugs:
        bug = Bug.dict_to_object(bug)
        if bug.product_component_pair in pc_dataset:
            bug_list.append(bug)
            print(f"https://bugzilla.mozilla.org/show_bug.cgi?id={bug.id} "
                  f"+ {bug.summary} + {bug.summary_token} + {bug.description} + {bug.description_token}")

    print(len(bug_list))

    filtered_bugs_filepath = PathUtil.get_filtered_bugs_filepath()
    FileUtil.dump_pickle(filtered_bugs_filepath, Bugs(bug_list))


if __name__ == "__main__":
    """
    从原始数据集中 1. 过滤掉不是product&component set中的bug
                2. 保留status = "CLOSED, RESOLVED, VERIFIED"
    """
    product_component_pair_filepath = PathUtil.get_pc_filepath()
    pc_dataset = FileUtil.load_pickle(product_component_pair_filepath)

    # 从 BUGS_DB = "data/bugs.json"中读取bugs -> Iterator[BugDict]
    bugs = bugzilla.get_bugs()

    bug_list = []

    for bug in tqdm(bugs, ascii=True):
        bug = Bug.dict_to_object(bug)
        if bug.product_component_pair in pc_dataset:
            if bug.status == 'CLOSED' or bug.status == 'RESOLVED' or bug.status == 'VERIFIED':
                bug_list.append(bug)
            # print(f"https://bugzilla.mozilla.org/show_bug.cgi?id={bug.id} "
            #       f"- {bug.summary} - {bug.description} - {bug.status} - {bug.summary_token}")
            # input()

    # print(len(bug_list))
    filtered_bugs = Bugs(bug_list)
    filtered_bugs.overall_bugs()

    filtered_bugs_filepath = PathUtil.get_filtered_bugs_filepath()
    FileUtil.dump_pickle(filtered_bugs_filepath, filtered_bugs)
