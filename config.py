import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root Directory

DATA_DIR = str(Path(ROOT_DIR) / "data")

OUTPUT_DIR = str(Path(ROOT_DIR) / "output")

FEATURE_VECTOR_DIR = str(Path(OUTPUT_DIR) / "feature_vector")

METRICS_DIR = str(Path(ROOT_DIR) / "metrics")

MODEL_DIR = str(Path(ROOT_DIR) / "model")

WORD2VEC_MODEL_NAME = 'word2vec-google-news-300'

FASTTEXT_MODEL_NAME = "wiki.en.bin"

WORD2VEC_DIM = 300

ONEHOT_DIM = 20047  # ONE_HOT_DIM = onehot.dim, you can get it from get_vec

TOPIC_KEYWORDS_NUM = 911

TOP_N_BUG_SUMMARY_FEATURE_VECTOR_NUM = 30

TOP_M_MISTOSSED_BUG_SUMMARY_FEATURE_VECTOR_NUM = 30

FEATURE_VECTOR_NUMS_PER_FILE = 1860000  # (the number of product::component) * 10,000
                                        # or FEATURE_VECTOR_NUMS_PER_FILE % (the number of product::component) == 0

BLOCK_SIZE = 10000

PRODUCT_COMPONENT_PAIR_NUM = 186

IS_MEAN = 0

IS_TFIDF = 0
