from bug_tossing.tossing_graph_construction.concept_extraction import extract_concept
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil

if __name__ == "__main__":

    train_bugs = FileUtil.load_pickle(PathUtil.get_train_bugs_filepath())
    pc_list = FileUtil.load_pickle(PathUtil.get_pc_filepath())
    corpus = train_bugs.get_pc_summary_token_list(pc_list)
    #     concept_set, unique_concept_set, common_concept_set, controversial_concept_set, community_concept_set
    #     pc: concept_set, unique_concept_set, common_concept_set, controversial_concept_set
    concept_set, pc_list = extract_concept(corpus, pc_list)
    # print(concept_set.concept_set)

    FileUtil.dump_pickle(PathUtil.get_concept_set_filepath(), concept_set)
    FileUtil.dump_pickle(PathUtil.get_pc_filepath(), pc_list)
