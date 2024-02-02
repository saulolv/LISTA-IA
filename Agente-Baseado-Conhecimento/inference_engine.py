class InferenceEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def forward_chain(self):
        while True:
            new_facts = self.knowledge_base.get_new_facts()
            if not new_facts:
                break
            for fact in new_facts:
                self.knowledge_base.add_fact(fact)
                print(f'New fact: {fact}')
            self.knowledge_base.remove_rule_fired_facts()

    def backward_chain(self, goal):
        if self.knowledge_base.is_fact_true(goal):
            return True
        for rule in self.knowledge_base.rules:
            if rule.consequent == goal:
                if all(self.backward_chain(antecedent) for antecedent in rule.antecedents):
                    return True
        return False

    def get_new_facts(self):
        return self.knowledge_base.get_new_facts()

    def add_fact(self, fact):
        self.knowledge_base.add_fact(fact)

    def remove_rule_fired_facts(self):
        self.knowledge_base.remove_rule_fired_facts()

    def is_fact_true(self, fact):
        return self.knowledge_base.is_fact_true(fact)

    def is_goal_true(self, goal):
        return self.backward_chain(goal)
    
    def backward_chaining(self, goal):
        return self.backward_chain(goal)
    
    def forward_chaining(self):
        return self.forward_chain()
    
