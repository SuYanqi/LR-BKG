import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import FEATURE_VECTOR_DIR

if __name__ == '__main__':
    train_or_test = "test"
#     ablation = "bug"
#     ablation = "top30"
#     ablation = "top30_mistossed"
#     ablation = "pc_name_description"
#     ablation = "community"
#     ablation = "degree"
#     ablation = "resolver_bystander"
    ablation = "bug_bug_component"
    data_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_graph")
    data_out_dir = Path(FEATURE_VECTOR_DIR, f"{train_or_test}_top_30_tfidf_onehot_percentage_{ablation}")
    data_out_dir.mkdir(exist_ok=True, parents=True)
    for index, data_file in enumerate(data_dir.glob(f"*.txt")):
        data_txt_name = str(data_file).split("/")[len(str(data_file).split("/")) - 1]
        # if data_txt_name == "feature_vector_1":
        #     continue
        with open(data_file, "r") as f:
            info = []
            matrix = []
            output = open(Path(str(data_out_dir), f"{data_txt_name}"), "w")
            counter = 0
            for line in tqdm(f, ascii=True):
                strs = re.split(r"[ :]", line.strip())
                info.append(strs[:3])
                matrix.append(strs[3:])
                counter += 1
                if counter >= 500000:
                    info = np.array(info)
                    matrix = np.array(matrix)
                    indices = list(range(1, matrix.shape[1], 2))
                    matrix = matrix[:, indices]
                    matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :129]], -1)

                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :128], matrix[:, 129:]], -1)
                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :129], matrix[:, 130:]], -1)
                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :133]], -1)
                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :130], matrix[:, 132:]], -1)
                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :4], matrix[:, 35:66], matrix[:, 97:]], -1)
                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :35], matrix[:, 66:97], matrix[:, 128:]], -1)
                    # matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, 4:]], -1)
                    template = "{} qid:{} " + " ".join(str(f_idx) + ":{}" for f_idx in range(1, matrix.shape[1] - 1, 1))
                    lines = []
                    for vec in matrix:
                        lines.append(template.format(*vec))
                    output.write("\n".join(lines))
                    output.write("\n")
                    info = []
                    matrix = []
                    counter = 0
            if counter > 0:
                info = np.array(info)
                matrix = np.array(matrix)
                indices = list(range(1, matrix.shape[1], 2))
                matrix = matrix[:, indices]
                matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :129]], -1)
                #                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :128], matrix[:, 129:]], -1) bug
#                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :129], matrix[:, 130:]], -1)
#                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :132]], -1)
#                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :130], matrix[:, 132:]], -1)
                #                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :4], matrix[:, 35:66], matrix[:, 97:]], -1)
#                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, :35], matrix[:, 66:97], matrix[:, 128:]], -1)
#                 matrix = np.concatenate([info[:, :1], info[:, 2:], matrix[:, 4:]], -1)
                template = "{} qid:{} " + " ".join(str(f_idx) + ":{}" for f_idx in range(1, matrix.shape[1] - 1, 1))
                lines = []
                for vec in matrix:
                    lines.append(template.format(*vec))
                output.write("\n".join(lines))
