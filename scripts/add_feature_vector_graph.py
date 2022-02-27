from pathlib import Path

from tqdm import tqdm

from config import FEATURE_VECTOR_DIR

if __name__ == '__main__':
    # for adding train graph feature vector
    train_or_test = "train"
    data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage")
    # data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_concept_set_onehot_tfidf")

    add_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_graph_feature_vector")

    data_out_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector")
    data_out_dir.mkdir(exist_ok=True, parents=True)
    for index, data_file in tqdm(enumerate(data_dir.glob(f"*.txt")), ascii=True):
        lines = list()
        data_txt_name = str(data_file).split("/")[len(str(data_file).split("/"))-1]

        for add_file in add_dir.glob(f"*.txt"):
            if data_txt_name == str(add_file).split("/")[len(str(add_file).split("/")) - 1]:
                with open(data_file, "r") as f:
                    lines.extend(f.readlines())
                with open(add_file, "r") as f:
                    for line_index, scores in enumerate(f.readlines()):
                        lines[line_index] = lines[line_index].replace("\n", "")
                        scores = scores.replace("\n", "")
                        scores = scores.split(' ')
                        for score_index, score in enumerate(scores):
                            if score_index == 0 or score_index == 1:  # because the first column is relevance label and the second column is bug_id (qid)
                                continue
                            if score != "":
                                score = score.split(":")
                                num = 128 + int(score[0])  # 有时候需要+1，因为第一次计算时，feature index从0开始
                                lines[line_index] = f"{lines[line_index]} {num}:{score[1]}"

                break

        with open(Path(str(data_out_dir), f"{data_txt_name}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(lines))

    # for adding test graph feature vector
    train_or_test = "test"
    data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage")
    # data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_concept_set_onehot_tfidf")

    add_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_graph_feature_vector")

    data_out_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector")
    data_out_dir.mkdir(exist_ok=True, parents=True)
    for index, data_file in tqdm(enumerate(data_dir.glob(f"*.txt")), ascii=True):
        lines = list()
        data_txt_name = str(data_file).split("/")[len(str(data_file).split("/")) - 1]

        for add_file in add_dir.glob(f"*.txt"):
            if data_txt_name == str(add_file).split("/")[len(str(add_file).split("/")) - 1]:
                with open(data_file, "r") as f:
                    lines.extend(f.readlines())
                with open(add_file, "r") as f:
                    for line_index, scores in enumerate(f.readlines()):
                        lines[line_index] = lines[line_index].replace("\n", "")
                        scores = scores.replace("\n", "")
                        scores = scores.split(' ')
                        for score_index, score in enumerate(scores):
                            if score_index == 0 or score_index == 1:  # because the first column is relevance label and the second column is bug_id (qid)
                                continue
                            if score != "":
                                score = score.split(":")
                                num = 128 + int(score[0])  # 有时候需要+1，因为第一次计算时，feature index从0开始
                                lines[line_index] = f"{lines[line_index]} {num}:{score[1]}"

                break

        with open(Path(str(data_out_dir), f"{data_txt_name}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(lines))
