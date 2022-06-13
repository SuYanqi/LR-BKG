# BugTossing

Dataset link: https://drive.google.com/file/d/17XXG75zmR3_bDeWKDNeXYNVL8ptAzUo3/view?usp=sharing
Fasttext Model link: https://fasttext.cc/docs/en/pretrained-vectors.html (a. download the English wiki.en.bin model b. Under root directory, construct model\wiki.en directory and c. put the wiki.en.bin model into it)

Software prepare
1. Neo4j: as the database to save the Bug Tossing Graph
2. Gephi: Use the community detection algorithm to get the modularity class of the product::component (The parameters of the community detection algorithm we used are ''Randomize'' is On, ''Use edge weights'' is On and ''Resolution'' is 1.0.)

Directory prepare
1. Construct the data directory under the root directory
2. Construct the data directory under the scripts directory

Dataset prepare
1. Run get_product_component.py to get product_component.json (adjust the filepath according to where you put the product_component_files)
2. Run filter_bugs.py to get filtered_bugs.json
3. Run split_train_test_dataset.py to get train_bugs.json and test_bugs.json
4. Run generate_tossing_graph_goal_oriented_path.py to get Bug Tossing Graph (a. need to connect with Neo4j b. train_bugs only)
5. Run get_vec.py to get vector for text information 
   (Note that after step 4, change the ONEHOT_DIM in config.py according to the onehot.dim from onehot = TfidfOnehotVectorizer()）
6. Run get_graph_feature_for_pc.py for graph features of product components

Feature vector
1. Change FEATURE_VECTOR_NUMS_PER_FILE in config.py to (the number of product::component) * 10,000 or FEATURE_VECTOR_NUMS_PER_FILE % (the number of product::component) == 0
2. Run get_feature_vector.py to get the relevance label and features about text information
3. Run get_graph_feature_vector.py to get bug feature and features about graph
4. Run add_feature_vector_graph.py to merge features from step2 and step3

Model
1. Run train_lambdaMart.py to train the learning to rank model
2. Run test_lambdaMart.py to test the model (change PRODUCT_COMPONENT_PAIR_NUM in config.py to the number of product::component)

Note that LR-BKG needs amount of memory and disk storage!!!
