import gensim
from gensim import corpora

from bug_tossing.types.product_component_pair import Topic
from bug_tossing.utils.path_util import PathUtil


class LDAUtil:
    @staticmethod
    def train_lda(texts):
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)

        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]

        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=20)
        print("model Done")
        topics = ldamodel.print_topics(num_topics=1, num_words=20)
        # print(topics)
        return topics

    @staticmethod
    def save_lda(ldamodel):
        """
        模型的保存
        :param ldamodel:
        :return:
        """
        ldamodel_filepath = PathUtil.load_lda_model_filepath()
        ldamodel.save(ldamodel_filepath)
        print("model Saved")

    @staticmethod
    def load_lda():
        """
        lda model loading
        :return:
        """
        ldamodel_filepath = PathUtil.load_lda_model_filepath()
        ldamodel = gensim.models.ldamodel.LdaModel.load(ldamodel_filepath)
        print("model Loaded")
        return ldamodel

    @staticmethod
    def transform_topics(topics):
        """
        topics = ldamodel.print_topics(num_topics=1, num_words=20)
        将topics转换成Topic的list
        :param topics:
        :return:
        """
        topic_list = list()
        keyword_weight_pairs = topics[0][1].split('+')
        for pair in keyword_weight_pairs:
            ones = pair.split('*')
            topic = Topic(eval(ones[1].strip()), float(ones[0]))
            topic_list.append(topic)
        return topic_list
