class Result:
    def __init__(self, product_component_pair=None):
        self.product_component_pair = product_component_pair
        self.bug_num = None
        self.accuracy_1 = None
        self.accuracy_3 = None
        self.accuracy_5 = None
        self.accuracy_10 = None
        self.community_accuracy_1 = None
        self.community_accuracy_3 = None
        self.community_accuracy_5 = None
        self.community_accuracy_10 = None
        self.community_ndcg_1 = None
        self.community_ndcg_3 = None
        self.community_ndcg_5 = None
        self.community_ndcg_10 = None
        self.direct_link_accuracy_1 = None
        self.direct_link_accuracy_3 = None
        self.direct_link_accuracy_5 = None
        self.direct_link_accuracy_10 = None
        self.direct_link_ndcg_1 = None
        self.direct_link_ndcg_3 = None
        self.direct_link_ndcg_5 = None
        self.direct_link_ndcg_10 = None

    def __repr__(self):
        return f'{self.product_component_pair} - {self.bug_num}' \
               f'\n\taccuracy@1-3-5-10:\t{self.accuracy_1} - {self.accuracy_3} - {self.accuracy_5} - {self.accuracy_10}' \
               f'\n\tcommunity_accuracy@1-3-5-10:\t{self.community_accuracy_1} - {self.community_accuracy_3} - {self.community_accuracy_5} - {self.community_accuracy_10}' \
               f'\n\tcommunity_ndcg@1-3-5-10:\t{self.community_ndcg_1} - {self.community_ndcg_3} - {self.community_ndcg_5} - {self.community_ndcg_10}' \
               f'\n\tdirect_link_accuracy@1-3-5-10:\t{self.direct_link_accuracy_1} - {self.direct_link_accuracy_3} - {self.direct_link_accuracy_5} - {self.direct_link_accuracy_10}' \
               f'\n\tdirect_link_ndcg@1-3-5-10:\t{self.direct_link_ndcg_1} - {self.direct_link_ndcg_3} - {self.direct_link_ndcg_5} - {self.direct_link_ndcg_10}'

    def __str__(self):
        return f'{self.product_component_pair} - {self.bug_num}' \
               f'\n\taccuracy@1-3-5-10:\t{self.accuracy_1} - {self.accuracy_3} - {self.accuracy_5} - {self.accuracy_10}' \
               f'\n\tcommunity_accuracy@1-3-5-10:\t{self.community_accuracy_1} - {self.community_accuracy_3} - {self.community_accuracy_5} - {self.community_accuracy_10}' \
               f'\n\tcommunity_ndcg@1-3-5-10:\t{self.community_ndcg_1} - {self.community_ndcg_3} - {self.community_ndcg_5} - {self.community_ndcg_10}' \
               f'\n\tdirect_link_accuracy@1-3-5-10:\t{self.direct_link_accuracy_1} - {self.direct_link_accuracy_3} - {self.direct_link_accuracy_5} - {self.direct_link_accuracy_10}' \
               f'\n\tdirect_link_ndcg@1-3-5-10:\t{self.direct_link_ndcg_1} - {self.direct_link_ndcg_3} - {self.direct_link_ndcg_5} - {self.direct_link_ndcg_10}'
