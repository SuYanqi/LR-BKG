from bug_tossing.types.bug import Bug
from bug_tossing.types.product_component_pair import ProductComponentPair


class RelevanceLabel:
    def __init__(self):
        self.cache = {}

    def get_relevance_label(self, bug: Bug, product_component_pair: ProductComponentPair) -> int:
        """
        Relevance Label
        Tossing Path： A    ->   B    ->    C
                    (1/2)^2   (1/2)^1    (1/2)^0
                Else：
                       0
        :param bug:
        :param product_component_pair:
        :return:
        """
        # n = len(bug.tossing_path.product_component_pair_list)
        # dict.get(key, default=None)
        # 参数
        # key -- 字典中要查找的键。
        # default -- 如果指定键的值不存在时，返回该默认值。

        # 先查看cache中是否包含该bug和pair字典的键值，如果有，则返回对应key，没有则计算，并存入
        # print(product_component_pair)
        key = str(bug.id) + ': ' + product_component_pair.product + '::' + product_component_pair.component
        if key not in self.cache:
            '''
            Relevance Label
            Tossing Path： A -> B -> C
                          1/3 2/3 3/3
                Else：
                          0
            # value = (bug.tossing_path.product_component_pair2idx.get(product_component_pair,
            #                                                          -1) + 1) / bug.tossing_path.length
            '''

            index = bug.tossing_path.product_component_pair2idx.get(product_component_pair, -1)
            if index == -1:
                value = 0.0
            else:
                value = pow((1.0 / 2), (bug.tossing_path.length - (index + 1)))
            self.cache[key] = value
        return self.cache.get(key)

        # return (bug.tossing_path.product_component_pair2idx.get(product_component_pair,
        #                                                         -1) + 1) / bug.tossing_path.length
