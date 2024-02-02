class NaturalLanguageInterface:
    def __init__(self, inference_engine, explanation_engine):
        self.inference_engine = inference_engine
        self.explanation_engine = explanation_engine

    def process_input(self, input):
        pass
        # Implemente processamento de entrada