import seaborn as sns
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predição de Obesidade",
                   page_icon="🏥", layout="centered")
st.title("🏥 Predição de Obesidade — App")

ART_DIR = "model"


def find_model_and_meta():
    if not os.path.isdir(ART_DIR):
        return None, None
    joblibs = [os.path.join(ART_DIR, f)
               for f in os.listdir(ART_DIR) if f.endswith(".joblib")]
    if not joblibs:
        return None, None
    model_path = max(joblibs, key=os.path.getmtime)
    finfo_path = os.path.join(ART_DIR, "feature_info.json")
    if not os.path.exists(finfo_path):
        finfo_path = None
    return model_path, finfo_path


MODEL_PATH, FEATURE_INFO_PATH = find_model_and_meta()

if MODEL_PATH is None or FEATURE_INFO_PATH is None:
    st.error("Nao encontrei os artefatos em 'model/'. Preciso de 'best_model_*.joblib' e 'feature_info.json'.")
    st.stop()


@st.cache_resource
def load_artifacts(model_path, finfo_path):
    model = joblib.load(model_path)
    with open(finfo_path, "r", encoding="utf-8") as f:
        finfo = json.load(f)
    return model, finfo


model, finfo = load_artifacts(MODEL_PATH, FEATURE_INFO_PATH)


def getFieldName(value):
    # Definimos os correspondentes em um dicionário
    correspondencias = {
        'Height': 'Altura',
        'Age': 'Idade',
        'Weight': 'Peso',
        'Gender': 'Gênero. Valores: Female, Male',
        'family_history': 'Existe histórico familiar de excesso de peso? Valores: yes (Sim), no (Não)',
        'FAVC': 'Consume frequentemente alimentos muito calóricos? Valores: yes (Sim), no (Não)',
        'FCVC': 'Consome com frequência vegetais nas refeições? Valores (escala 1–3): 1 raramente, 2 às vezes, 3 sempre.',
        'NCP': 'Número de refeições principais por dia. Valores (escala 1–4): 1 uma refeição, 2 duas, 3 três, 4 quatro ou mais.',
        'CAEC': 'Consume lanches/comes entre as refeições? Valores: no (não consome), Sometimes (às vezes), Frequently (frequentemente), Always (sempre).',
        'SMOKE': 'Hábito de fumar. Valores: yes (fuma), no (não fuma).',
        'CH2O': 'Consumo diário de água. Valores (escala 1–3): 1 < 1 L/dia, 2 1–2 L/dia, 3 > 2 L/dia.',
        'SCC': 'Monitora a ingestão calórica diária? Valores: yes (sim), no (não).',
        'FAF': 'Frequência semanal de atividade física. Valores (escala 0–3): 0 nenhuma, 1 ~1–2×/sem, 2 ~3–4×/sem, 3 5×/sem ou mais.',
        'TUE': 'Tempo diário usando dispositivos eletrônicos. Valores (escala 0–2): 0 ~0–2 h/dia, 1 ~3–5 h/dia, 2 > 5 h/dia.',
        'CALC': 'Consumo de bebida alcoólica. Valores: no (não bebe), Sometimes (às vezes), Frequently (frequentemente), Always (sempre).',
        'MTRANS': 'Qual o Meio de transporte habitual? Valores: Automobile (carro), Motorbike (moto), Bike (bicicleta), Public_Transportation (transporte público), Walking (a pé).',
        'Obesity': 'Classe de peso corporal. Valores: Insufficient_Weight (abaixo do peso), Normal_Weight (peso normal), Overweight_Level_I (sobrepeso I), Overweight_Level_II (sobrepeso II), Obesity_Type',
    }
    return correspondencias.get(value, "Nome não encontrado no mapeamento: " + value)


def getResultValue(value):
    correspondencias = {
        'Insufficient_Weight': 'Abaixo do peso',
        'Normal_Weight': 'Peso normal',
        'Overweight_Level_I': 'Sobrepeso I',
        'Overweight_Level_II': 'Sobrepeso II',
        'Obesity_Type_I': 'Obesidade I',
        'Obesity_Type_II': 'Obesidade II',
        'Obesity_Type_III': 'Obesidade III'
    }
    return correspondencias.get(value, "Resultado não encontrado no mapeamento: " + value)


num_cols = finfo.get("num_cols", [])
cat_cols = finfo.get("cat_cols", [])
all_features = num_cols+cat_cols

st.caption(f"Modelo: {os.path.basename(MODEL_PATH)}")

tab_pred, tab_ins = st.tabs(["🔮 Predição", "📊 Insights"])

with tab_pred:
    st.subheader("Insira seus dados pessoais abaixo")
    cols = st.columns(1)
    vals = {}

    for i, c in enumerate(num_cols):
        with cols[0]:
            default = 0
            if c == "Age":
                default = 30
            if c == "Height":
                default = 1.70
            if c == "Weight":
                default = 70.0
            if c == "BMI":
                continue
            vals[c] = st.number_input(getFieldName(c), value=default)

    for i, c in enumerate(cat_cols):
        with cols[0]:
            vals[c] = st.text_input(getFieldName(c), value="")

    if st.button("Efetuar análise preditiva"):
        x = pd.DataFrame([vals], columns=all_features)
        try:
            y_pred = model.predict(x)[0]
            st.success(
                f"Grau de obesidade previsto: **{getResultValue(y_pred)}**")
            try:
                proba = model.predict_proba(x)[0]
                classes = model.classes_
                st.write("Confiança (top 3):")
                top = np.argsort(proba)[::-1][:3]
                for i in top:
                    st.write(f"- {classes[i]}: {proba[i]:.2%}")
            except Exception:
                pass
        except Exception as e:
            st.error(f"Erro ao prever: {e}")

with tab_ins:
    st.subheader("Análises realizadas no conjunto de dados coletados\n\n")

    df = pd.read_csv('data/dados_tratados.csv', sep=';')

    obesity_counts = df['Obesity'].value_counts()
    labels = obesity_counts.index
    sizes = obesity_counts.values
    percentages = (sizes / sizes.sum()) * 100

    # 1. Capture a figura em uma variável (fig)
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.viridis(np.linspace(0, 1, len(labels))),
        wedgeprops={'edgecolor': 'black', 'linewidth': 1}
    )

    ax.set_title('Distribuição dos Níveis de Obesidade (Gráfico de Pizza)')
    ax.axis('equal')

    # 2. Em vez de plt.show(), use o comando do streamlit
    st.pyplot(fig)

    # Criando a figura e os eixos
    fig, ax = plt.subplots(figsize=(8, 6))

    # Gerando o boxplot (passando o 'ax' para o Seaborn saber onde desenhar)
    sns.boxplot(data=df, x='Gender', y='Weight', hue='Obesity', ax=ax)

    # Configurando título e labels
    ax.set_title('Distribuição de Peso por Gênero e Nível de Obesidade')

    # Exibindo no Streamlit
    st.pyplot(fig)
