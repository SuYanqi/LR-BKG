from bug_tossing.types.product_component_pair import ProductComponentPair, ProductComponentPairs
from bug_tossing.utils.file_util import FileUtil
from bug_tossing.utils.path_util import PathUtil
import csv

if __name__ == "__main__":
    '''
    Note that need to adjust column number according to the data table
    '''
    with open('/Users/suyanqi/Desktop/product_component.txt') as file_obj:
        pairlist = []
        content = file_obj.readline()
        while content:
            print(content)
            list = content.replace("\n", "").split("::", 1)
            print(f'{list}\n\n')
            product_component_pair = ProductComponentPair()
            product_component_pair.product = list[0]
            product_component_pair.component = list[1]
            pairlist.append(product_component_pair)
            content = file_obj.readline()

    file_obj.close()

    with open('/Users/suyanqi/Desktop/product_component_description_information.csv') as csvfile:
        reader = csv.reader(csvfile)
        # 这里不需要readlines
        for index, line in enumerate(reader):
            print(line[5])
            if index == 0:
                continue
            pairlist[index-1].description = line[5]
            # print(type(line))

    with open('/Users/suyanqi/Desktop/frequency>=2_graph/product_component_community.csv') as csvfile:
        reader = csv.reader(csvfile)
        # 这里不需要readlines
        for index, line in enumerate(reader):
            print(line[6])
            if index == 0:
                continue

            for pc in pairlist:
                if f'{pc.product}::{pc.component}' == line[1]:
                    pc.community = line[6]
                    break

    product_component_pair_filepath = PathUtil.get_pc_filepath()
    FileUtil.dump_pickle(product_component_pair_filepath, ProductComponentPairs(pairlist))
