from tqdm import tqdm

from bug_tossing.types.concept_set import ConceptSet
from bug_tossing.utils.nlp_util import NLPUtil


def extract_concept(corpus, product_component_pair_list):
    """
    concept_set, unique_concept_set, common_concept_set, controversial_concept_set, community_concept_set
    pc: concept_set, unique_concept_set, common_concept_set, controversial_concept_set
    :param corpus:
    :param product_component_pair_list:
    :return:
    """
    concept_set = ConceptSet()
    # concept_set and unique_concept_set
    # only keep noun and verb
    # corpus = NLPUtil.filter_paragraph_by_pos_tag(corpus)
    concept_set.extract_concept_set(corpus)

    product_component_pair_list.set_concept_set(corpus, concept_set)
    concept_set.get_community_concept_set(product_component_pair_list)
    concept_set.get_common_controversial_concept_set()
    concept_set.get_word_index_dict()
    # print(concept_set.word_index_dict)
    # print(len(concept_set.word_index_dict))
    for pc in tqdm(product_component_pair_list):
        pc.set_unique_common_controversial_concept_set(concept_set)
    return concept_set, product_component_pair_list
