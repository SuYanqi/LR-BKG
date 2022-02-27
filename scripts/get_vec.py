from tqdm import tqdm

from bug_tossing.types.product_component_pair import ProductComponentPairs
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.nlp_util import NLPUtil, TfidfEmbeddingVectorizer, TfidfOnehotVectorizer
from bug_tossing.utils.path_util import PathUtil

if __name__ == "__main__":
    # nlp_model = NLPUtil.load_word2vec_model(WORD2VEC_MODEL_NAME)
    nlp_model = NLPUtil.load_fasttext_model(PathUtil.load_fasttext_model_filepath())

    # pc_topics_list_filepath = PathUtil.get_pc_with_topics_filepath()
    # pc_topics_list = FileUtil.load_pickle(pc_topics_list_filepath)
    # for pc in tqdm(pc_topics_list, ascii=True):
    #     # print(pc)
    #     pc.get_topics_keywords_vec(nlp_model)
    # FileUtil.dump_pickle(pc_topics_list_filepath, pc_topics_list)

    # print('***************************************************************************************')

    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())

    train_bugs_filepath = PathUtil.get_train_bugs_filepath()
    train_bugs = FileUtil.load_pickle(train_bugs_filepath)

    tfidf = TfidfEmbeddingVectorizer(nlp_model)
    # 以一个bug summary为单位
    # tfidf.fit(train_bugs.get_bug_summary_token_list())

    # 以一个pc的所有summary为单位
    tfidf.fit(train_bugs.get_pc_summary_token_list(pc_list))
    # 以一个pc的所有summary和description为单位
    # tfidf.fit(train_bugs.get_pc_summary_description_token_list(pc_list))

    onehot = TfidfOnehotVectorizer()
    # 目前用的idf, 以pc为unit
    onehot.fit(train_bugs.get_pc_summary_token_list(pc_list))
    print(f"You need to change the value of ONE_HOT_DIM in config.py to {onehot.dim}!!!")
    # onehot.fit(train_bugs.get_pc_description_token_list(pc_list))
    # 目前用的idf, 以pc为unit（summary description）
    # onehot.fit(train_bugs.get_pc_summary_description_token_list(pc_list))

    train_bugs.get_vec(nlp_model, tfidf, onehot)
    FileUtil.dump_pickle(train_bugs_filepath, train_bugs)

    # print('***************************************************************************************')

    test_bugs_filepath = PathUtil.get_test_bugs_filepath()
    test_bugs = FileUtil.load_pickle(test_bugs_filepath)
    test_bugs.get_vec(nlp_model, tfidf, onehot)
    FileUtil.dump_pickle(test_bugs_filepath, test_bugs)
    # print('***************************************************************************************')

    pc_list.get_vec(nlp_model, tfidf, onehot)
    FileUtil.dump_pickle(PathUtil.get_pc_filepath(), pc_list)
