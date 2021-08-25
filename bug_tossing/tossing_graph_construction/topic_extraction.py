from bug_tossing.utils.lda_util import LDAUtil


def extract_topic(bugs):
    """
    extract topics from bug.summary_token
    :param bugs:
    :return:
    """
    texts = list()
    for bug in bugs:
        texts.append(bug.summary_token)

    topics = LDAUtil.train_lda(texts)
    return topics
