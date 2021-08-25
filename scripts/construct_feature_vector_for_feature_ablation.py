from pathlib import Path
from tqdm import tqdm

from config import FEATURE_VECTOR_DIR

if __name__ == '__main__':
    train_or_test = "test"
    data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector")
    feature_ablation = "bug"
    # feature_ablation = "community"
    # feature_ablation = "degree"
    # feature_ablation = "resolver_bystander"
    # feature_ablation = "pc_name_descripion"
    # feature_ablation = "top_30"
    # feature_ablation = "mistossed_top_30"

    data_out_dir = Path(FEATURE_VECTOR_DIR,
                        f"{train_or_test}_top_30_tfidf_onehot_percentage_graph_feature_vector_{feature_ablation}")
    data_out_dir.mkdir(exist_ok=True, parents=True)

    for index, data_file in tqdm(enumerate(data_dir.glob(f"*.{train_or_test}")), ascii=True):
        lines = list()
        with open(data_file, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                features = line.split(' ')
                for feature_index, feature in enumerate(features):

                    if feature_index == 0:
                        line = feature
                        continue
                    if feature != "":
                        index_feature = feature.split(":")

                        index_feature[0] = int(index_feature[0])
                        if feature_ablation == "bug" and index_feature[0] == 129:
                            continue
                        # if feature_ablation == "community" and index_feature[0] == 130:
                        #     continue
                        # if feature_ablation == "degree" and 133 <= index_feature[0] <= 138:
                        #     continue
                        # if feature_ablation == "resolver_bystander" and 131 <= index_feature[0] <= 132:
                        #     continue
                        # if feature_ablation == "pc_name_description" and 1 <= index_feature[0] <= 4:
                        #     continue
                        # if feature_ablation == "top_30" and (5 <= index_feature[0] <= 35 or 67 <= index_feature[0] <= 97):
                        #     continue
                        # if feature_ablation == "mistossed_top_30" and (36 <= index_feature[0] <= 66 or 98 <= index_feature[0] <= 128):
                        #     continue

                        line = line + " " + feature
                lines.append(line)

        with open(Path(str(data_out_dir), f"feature_vector_{index}.{train_or_test}"), "w") as f:
            # 利用追加模式,参数从w替换为a即可
            f.write("\n".join(lines))
