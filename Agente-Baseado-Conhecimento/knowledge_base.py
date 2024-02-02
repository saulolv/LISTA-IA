import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class KnowledgeBase:
    def __init__(self, file_path, target):
        self.df = pd.read_excel(file_path)
        self.target = target
        self.model = self.train_model()
    
    def train_model(self):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        model = DecisionTreeClassifier()
        model.fit(X, y)
        return model