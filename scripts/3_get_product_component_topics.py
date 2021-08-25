from tqdm import tqdm

from bug_tossing.tossing_graph_construction.topic_extraction import extract_topic
from bug_tossing.types.bugs import Bugs
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.lda_util import LDAUtil
from bug_tossing.utils.path_util import PathUtil

if __name__ == "__main__":
    train_bugs_filepath = PathUtil.get_train_bugs_filepath()
    train_bugs = FileUtil.load_pickle(train_bugs_filepath)
    # print(train_bugs.overall_bugs())
    product_component_pair_filepath = PathUtil.get_pc_filepath()
    pc_dataset = FileUtil.load_pickle(product_component_pair_filepath)
    i = 0
    pc_topics_list = list()
    for pc in tqdm(pc_dataset, ascii=True):
        bug_list = list()
        # print(pc)
        for bug in train_bugs:
            if bug.product_component_pair == pc:
                bug_list.append(bug)
                # train_bugs.remove(bug)
        if len(bug_list) != 0:
            # Bugs(bug_list).overall_bugs()
            topics = extract_topic(bug_list)
            # print(topics)
            pc.topics = LDAUtil.transform_topics(topics)
            pc_topics_list.append(pc)
            # print(pc.topics)

    # print(len(pc_topics_list))
    pc_topics_list_filepath = PathUtil.get_pc_with_topics_filepath()
    FileUtil.dump_pickle(pc_topics_list_filepath, pc_topics_list)
    # print(train_bugs.get_length())
    # i = i + 1
    # if i == 2:
    #     break
    # 删除多余的5个product&component
    # for pc in pc_dataset:
    #     print(pc)
    #     print(pc.topic)
