from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil

if __name__ == "__main__":
    filtered_bugs_filepath = PathUtil.get_filtered_bugs_filepath()
    bugs = FileUtil.load_pickle(filtered_bugs_filepath)
    bugs.sort_by_creation_time()
    for bug in bugs:
        print(bug.creation_time)
