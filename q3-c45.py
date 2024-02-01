import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def encadeamento_para_tras(model, exemplo, feature_names):
  """
  Realiza o encadeamento para trás em uma árvore de decisão, 
  gerando regras "Se Então".

  Argumentos:
    model: Modelo de árvore de decisão.
    exemplo: Lista com os valores das features do exemplo a ser analisado.
    feature_names: Lista com os nomes das features.

  Retorno:
    Regras "Se Então" em formato textual.
  """

  node = 0  # Começamos no nó raiz
  features = model.tree_.feature
  thresholds = model.tree_.threshold
  left_children = model.tree_.children_left
  right_children = model.tree_.children_right
  classes = model.classes_

  regras = []  # Lista para armazenar as regras

  while node != -2:  # Percorre a árvore até chegar a um nó folha
    feature_name = feature_names[features[node]]
    threshold = thresholds[node]

    # Se for um nó de divisão
    if features[node] != -2:
      if exemplo[features[node]] <= threshold:
        regras.append(f"SE {feature_name} <= {threshold}")
        node = left_children[node]
      else:
        regras.append(f"SE {feature_name} > {threshold}")
        node = right_children[node]
    else:  # Se for um nó folha
      classe_predita = classes[model.tree_.value[node].argmax()]
      regras.append(f"ENTÃO Risco predito = {classe_predita}")

  # Inverte a ordem da lista para que as regras sejam lidas da forma correta
  regras.reverse()

  # Concatena as regras em uma string
  return " ".join(regras)

def encadeamento_para_frente(model, exemplo, feature_names):
    node = 0  # Começamos no nó raiz
    features = model.tree_.feature
    thresholds = model.tree_.threshold
    classes = model.classes_

    print("Encadeamento para frente:")
    while True:
        # Verifica a condição no nó atual
        if features[node] != -2:  # -2 indica um nó folha
            feature_name = feature_names[features[node]]
            threshold = thresholds[node]

            print(f"SE {feature_name} <= {threshold} ENTÃO")

            if exemplo[features[node]] <= threshold:
                node = model.tree_.children_left[node]
            else:
                node = model.tree_.children_right[node]
        else:
            # Chegamos a um nó folha, retorna a classe correspondente
            classe_predita = classes[model.tree_.value[node].argmax()]
            print(f"CONCLUSÃO: Risco predito = {classe_predita}")
            return classe_predita

def inferencer(model, user_data):
    user_data = le.transform(user_data)
    return le.inverse_transform(model.predict([user_data]))[0]

# Carrega os dados do arquivo Excel
df = pd.read_excel('IA-gerente.xlsx')

# Exemplos adicionais para ampliar a base
novos_exemplos = pd.DataFrame({
    'historia_credito': ['Boa', 'Boa', 'Ruim', 'Ruim', 'Desconhecida', 'Desconhecida'],
    'divida': ['Baixa', 'Alta', 'Baixa', 'Alta', 'Baixa', 'Alta'],
    'garantia': ['Nenhuma', 'Adequada', 'Nenhuma', 'Adequada', 'Adequada', 'Nenhuma'],
    'renda': ['$15k - $35k', 'Acima de $35k', '$0 - $15k', 'Acima de $35k', 'Acima de $35k', '$15k - $35k'],
    'risco': ['Baixo', 'Moderado', 'Baixo', 'Moderado', 'Moderado', 'Baixo']
})

# Adiciona os novos exemplos ao DataFrame original
df = pd.concat([df, novos_exemplos], ignore_index=True)

# Converte as variáveis categóricas para numéricas
le = LabelEncoder()
df['historia_credito'] = le.fit_transform(df['historia_credito'])
df['divida'] = le.fit_transform(df['divida'])
df['garantia'] = le.fit_transform(df['garantia'])
df['renda'] = le.fit_transform(df['renda'])
df['risco'] = le.fit_transform(df['risco'])

X = df[['historia_credito', 'divida', 'garantia', 'renda']]
y = df['risco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
# Treinamento do modelo
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Avaliação do modelo
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('Acurácia (treino):', (y_train_pred == y_train).mean())
print('Acurácia (teste):', (y_test_pred == y_test).mean())


tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)

# Exemplo de teste
exemplo1 = [1, 1, 0, 0]  # história_credito=Desconhecida, divida=Alta, garantia=Nenhuma, renda=$15k - $35k
exemplo2 = [0, 1, 0, 1]  # história_credito=Boa, divida=Alta, garantia=Adequada, renda=Acima de $35k
exemplo3 = [1, 0, 1, 1]  # história_credito=Desconhecida, divida=Baixa, garantia=Adequada, renda=Acima de $35k


for exemplo in [exemplo1, exemplo2, exemplo3]:
    print(f"Encademento para frente: {encadeamento_para_frente(model, exemplo, list(X.columns))}")
    print(f"Encadeamento para trás: {encadeamento_para_tras(model, exemplo, list(X.columns))}")
    print("\n")
    