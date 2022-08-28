from pathlib import Path

from tqdm import tqdm

from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.list_util import ListUtil
from bug_tossing.utils.path_util import PathUtil
from config import FEATURE_VECTOR_DIR, PRODUCT_COMPONENT_PAIR_NUM


def split_test_file(tossed_test_bugs):
    train_or_test = "test"
    data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector")

    tossed_data_out_dir = Path(FEATURE_VECTOR_DIR,
                               f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector_tossed")
    tossed_data_out_dir.mkdir(exist_ok=True, parents=True)

    untossed_data_out_dir = Path(FEATURE_VECTOR_DIR,
                                 f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector_untossed")
    untossed_data_out_dir.mkdir(exist_ok=True, parents=True)

    for index, data_file in tqdm(enumerate(data_dir.glob(f"*.{train_or_test}")), ascii=True):
        tossed_lines = list()
        untossed_lines = list()
        lines = list()
        data_txt_name = str(data_file).split("/")[len(str(data_file).split("/")) - 1]
        data_txt_name_without = data_txt_name.split(".")[0]

        for add_file in data_dir.glob(f"*.txt"):
            add_file_name = str(add_file).split("/")[len(str(add_file).split("/")) - 1]
            add_file_name_without = add_file_name.split(".")[0]
            if data_txt_name_without == add_file_name_without:
                with open(data_file, "r") as f:
                    lines.extend(f.readlines())
                    after_lines = list()
                    for line in lines:
                        after_lines.append(line.replace("\n", ""))
                    lines = after_lines
                    lines_block = ListUtil.list_of_groups(lines, PRODUCT_COMPONENT_PAIR_NUM)
                with open(add_file, "r") as f:
                    bug_lines = f.readlines()
                    bug_lines_block = ListUtil.list_of_groups(bug_lines, PRODUCT_COMPONENT_PAIR_NUM)
                    for block_index, bug_lines in enumerate(bug_lines_block):
                        scores = bug_lines[0].split(' ')
                        bug_id = int(scores[1].split(":")[1])
                        tossed = 0
                        for bug in tossed_test_bugs:
                            if bug_id == bug.id:
                                tossed_lines.extend(lines_block[block_index])
                                tossed = 1
                                break
                        if tossed == 0:
                            untossed_lines.extend(lines_block[block_index])
                break

        with open(Path(str(tossed_data_out_dir), f"{data_txt_name}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(tossed_lines))
            print(f"tossed_lines: {len(tossed_lines)}")

        with open(Path(str(untossed_data_out_dir), f"{data_txt_name}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(untossed_lines))
            print(f"untossed_lines: {len(untossed_lines)}")


def split_txt_file(tossed_test_bugs):
    train_or_test = "test"

    data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector")

    tossed_data_out_dir = Path(FEATURE_VECTOR_DIR,
                               f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector_tossed")
    tossed_data_out_dir.mkdir(exist_ok=True, parents=True)

    untossed_data_out_dir = Path(FEATURE_VECTOR_DIR,
                                 f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector_untossed")
    untossed_data_out_dir.mkdir(exist_ok=True, parents=True)

    for add_file in tqdm(data_dir.glob(f"*.txt"), ascii=True):
        add_file_name = str(add_file).split("/")[len(str(add_file).split("/")) - 1]

        tossed_lines = list()
        untossed_lines = list()
        with open(add_file, "r") as f:
            lines = f.readlines()
            after_lines = list()
            for line in lines:
                after_lines.append(line.replace("\n", ""))
            lines = after_lines
            lines_block = ListUtil.list_of_groups(lines, PRODUCT_COMPONENT_PAIR_NUM)

            for block_index, lines in enumerate(lines_block):
                scores = lines[0].split(' ')
                bug_id = int(scores[1].split(":")[1])
                tossed = 0
                for bug in tossed_test_bugs:
                    if bug_id == bug.id:
                        tossed_lines.extend(lines_block[block_index])
                        tossed = 1
                        break
                if tossed == 0:
                    untossed_lines.extend(lines_block[block_index])

        with open(Path(str(tossed_data_out_dir), f"{add_file_name}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(tossed_lines))
            print(f"tossed_lines: {len(tossed_lines)}")

        with open(Path(str(untossed_data_out_dir), f"{add_file_name}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(untossed_lines))
            print(f"untossed_lines: {len(untossed_lines)}")


if __name__ == '__main__':
    tossed_test_bugs = FileUtil.load_pickle(PathUtil.get_tossed_test_bugs_filepath())
    # split_test_file(tossed_test_bugs)
    split_txt_file(tossed_test_bugs)
