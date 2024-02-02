from knowledge_base import KnowledgeBase
from inference_engine import InferenceEngine
from explanation_engine import ExplanationEngine
from interface_user import NaturalLanguageInterface

kb = KnowledgeBase(file_path='IA-gerente', target='target')

inference_engine = InferenceEngine(kb)
explanation_engine = ExplanationEngine(kb)

nli = NaturalLanguageInterface(inference_engine, explanation_engine)



user_input = input('Digite sua renda: ')
nli.process_input(user_input)
# Output: New fact: renda: alta
# Output: New fact: target: sim

user_input = input('Digite sua garantia: ')


