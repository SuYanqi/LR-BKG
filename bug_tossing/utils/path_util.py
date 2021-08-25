from pathlib import Path

from config import DATA_DIR, METRICS_DIR, OUTPUT_DIR, MODEL_DIR, FASTTEXT_MODEL_NAME


class PathUtil:

    @staticmethod
    def get_specified_product_component_bugs_filepath(product_component_pair):
        return DATA_DIR / Path("product_component_bugs") / \
               Path(product_component_pair.product+"_"+product_component_pair.component+"_"+"bugs.json")

    @staticmethod
    def get_bugs_filepath():
        return DATA_DIR / Path("bugs.json")

    @staticmethod
    def get_bugs_object_filepath():
        return DATA_DIR / Path("bugs_object.json")

    @staticmethod
    def get_test_bugs_filepath():
        return DATA_DIR / Path("test_bugs.json")

    @staticmethod
    def get_tossed_test_bugs_filepath():
        return DATA_DIR / Path("tossed_test_bugs.json")

    @staticmethod
    def get_untossed_test_bugs_filepath():
        return DATA_DIR / Path("untossed_test_bugs.json")

    @staticmethod
    def get_train_bugs_filepath():
        return DATA_DIR / Path("train_bugs.json")

    @staticmethod
    def get_filtered_bugs_filepath():
        return DATA_DIR / Path("filtered_bugs.json")

    @staticmethod
    def get_pc_filepath():
        return DATA_DIR / Path("product_component.json")

    @staticmethod
    def get_pc_adjacency_matrix_filepath():
        return DATA_DIR / Path("product_component_adjacency_matrix.json")

    @staticmethod
    def get_pc_with_topics_filepath():
        return DATA_DIR / Path("product_component_with_topics.json")

    @staticmethod
    def get_concept_set_filepath():
        return DATA_DIR / Path("concept_set.json")

    @staticmethod
    def get_bugbug_pair_outputs_filepath():
        return METRICS_DIR / Path("bugbug", "test_bugs_top10_pair_outputs.json")

    @staticmethod
    def get_bugbug_top10_outputs_filepath():
        return METRICS_DIR / Path("bugbug", "test_bugs_metrics_top10.json")


    @staticmethod
    def get_bugbug_metrics_filepath():
        return METRICS_DIR / Path("bugbug", "test_bugs_metrics.json")

    @staticmethod
    def get_bugbug_tossed_metrics_filepath():
        return METRICS_DIR / Path("bugbug", "tossed_test_bugs_metrics.json")

    @staticmethod
    def get_bugbug_untossed_metrics_filepath():
        return METRICS_DIR / Path("bugbug", "untossed_test_bugs_metrics.json")

    @staticmethod
    def get_bugbug_result_filepath():
        return METRICS_DIR / Path("bugbug", "result.json")

    @staticmethod
    def get_bugbug_tossed_result_filepath():
        return METRICS_DIR / Path("bugbug", "tossed_result.json")

    @staticmethod
    def get_bugbug_untossed_result_filepath():
        return METRICS_DIR / Path("bugbug", "untossed_result.json")

    @staticmethod
    def get_bugbug_with_tossing_graph_metrics_filepath():
        return METRICS_DIR / Path("bugbug_with_tossing_graph", "metrics_with_tossing_graph.json")

    @staticmethod
    def get_bugbug_with_tossing_graph_tossed_metrics_filepath():
        return METRICS_DIR / Path("bugbug_with_tossing_graph", "tossed_metrics_with_tossing_graph.json")

    @staticmethod
    def get_bugbug_with_tossing_graph_untossed_metrics_filepath():
        return METRICS_DIR / Path("bugbug_with_tossing_graph", "untossed_metrics_with_tossing_graph.json")

    @staticmethod
    def get_bugbug_with_tossing_graph_result_filepath():
        return METRICS_DIR / Path("bugbug_with_tossing_graph", "result.json")

    @staticmethod
    def get_bugbug_with_tossing_graph_tossed_result_filepath():
        return METRICS_DIR / Path("bugbug_with_tossing_graph", "tossed_result.json")

    @staticmethod
    def get_bugbug_with_tossing_graph_untossed_result_filepath():
        return METRICS_DIR / Path("bugbug_with_tossing_graph", "untossed_result.json")

    @staticmethod
    def get_our_metrics_ablation_filepath(ablation):
        return METRICS_DIR / Path("our_approach", f"metrics_{ablation}.json")

    @staticmethod
    def get_our_metrics_filepath():
        return METRICS_DIR / Path("our_approach", "metrics.json")

    @staticmethod
    def get_our_tossed_metrics_filepath():
        return METRICS_DIR / Path("our_approach", "tossed_metrics.json")

    @staticmethod
    def get_our_untossed_metrics_filepath():
        return METRICS_DIR / Path("our_approach", "untossed_metrics.json")

    @staticmethod
    def get_our_result_ablation_filepath(ablation):
        return METRICS_DIR / Path("our_approach", f"result_{ablation}.json")

    @staticmethod
    def get_our_result_filepath():
        return METRICS_DIR / Path("our_approach", "result.json")

    @staticmethod
    def get_our_untossed_result_filepath():
        return METRICS_DIR / Path("our_approach", "untossed_result.json")

    @staticmethod
    def get_our_tossed_result_filepath():
        return METRICS_DIR / Path("our_approach", "tossed_result.json")

    @staticmethod
    def get_our_approach_metrics_with_tossing_graph_filepath():
        return METRICS_DIR / Path("our_approach", "metrics_with_tossing_graph.json")

    @staticmethod
    def get_feature_vector_train_filepath():
        return OUTPUT_DIR / Path("feature_vector_train.txt")

    @staticmethod
    def get_feature_vector_test_filepath():
        return OUTPUT_DIR / Path("feature_vector_test.txt")

    @staticmethod
    def get_feature_vector_score_filepath():
        return OUTPUT_DIR / Path("feature_vector_score.json")

    @staticmethod
    def load_lambdaMart_model_filepath():
        return MODEL_DIR / Path("lambdaMart_model.json")

    @staticmethod
    def load_lda_model_filepath():
        return MODEL_DIR / Path("lda.model")

    @staticmethod
    def load_fasttext_model_filepath():
        return MODEL_DIR / Path("wiki.en", FASTTEXT_MODEL_NAME)
