import seaborn as sns
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predição de Risco de Defasagem",
                   page_icon="🎓", layout="centered")

st.title("🎓 Análise preditiva de Defasagem Educacional")

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
        'Idade': 'Idade',
        'Gênero': 'Gênero',
        'Fase': 'Fase Atual',
        'INDE': 'INDE Atual',
        'IAN': 'Ind. Adequação de Nível (IAN)',
        'Mat': 'Matemática (MAT)',
        'Por': 'Português (POR)',
        'Ing': 'Inglês (ING)',
        'IAA': 'Ind. Autoavaliação (IAA)',
        'IEG': 'Ind. Engajamento (IEG)',
        'IPS': 'Ind. Psicossocial (IPS)',
        'IPP': 'Ind. Psicopedagógico (IPP)',
        'IDA': 'Indicador de Desempenho Acad. (IDA)',
        'IPV': 'Indicador de Ponto de Virada (IPV)'
    }
    return correspondencias.get(value, "Nome não encontrado no mapeamento: " + value)


def getResultValue(value):
    correspondencias = {
        '0': 'Aluno com baixo risco de defasagem',
        '1': 'Aluno com alto risco de defasagem',
    }
    return correspondencias.get(value, "Resultado não encontrado no mapeamento: " + value)


num_cols = finfo.get("num_cols", [])
cat_cols = finfo.get("cat_cols", [])
all_features = num_cols+cat_cols

st.caption(f"Modelo: {os.path.basename(MODEL_PATH)}")

st.subheader("Insira os dados do aluno para análise preditiva:")
cols = st.columns(1)
vals = {}

for i, c in enumerate(num_cols):
    with cols[0]:
        default = 0
        if c == "Idade":
            default = 10
        elif c in ["Mat", "Por", "Ing"]:
            default = 5
        elif c in ["IAA", "IEG", "IPS", "IPP", "IDA", "IPV"]:
            default = 6.0
    vals[c] = st.number_input(getFieldName(c), value=default)

# No loop de colunas categóricas (cat_cols)
for c in cat_cols:
    if c == "Gênero":
        vals[c] = st.selectbox(getFieldName(c), options=[
                               "Masculino", "Feminino"])
    elif c == "Fase":
        vals[c] = st.selectbox(getFieldName(c), options=[
                               "ALFA", "Fase 1", "Fase 2", "Fase 3", "Fase 4", "Fase 5", "Fase 6", "Fase 7", "Fase 8"])
    else:
        vals[c] = st.text_input(getFieldName(c), value="")

with st.expander("ℹ️ Entenda os Indicadores"):
    st.write("""
    - **IAN:** Indicador de Adequação de Nível.
    - **IDA:** Indicador de Desempenho Acadêmico.
    - **IEG:** Indicador de Engajamento.
    - **IPV:** Ponto de Virada (indicador de maturidade e esforço).
    """)


if st.button("Efetuar análise preditiva"):
    x = pd.DataFrame([vals], columns=all_features)
    try:
        # Pega a probabilidade da classe 1 (alto risco)
        y_prob = model.predict_proba(x)[0][1]
        y_pred = model.predict(x)[0]

        # Interface visual com métricas
        st.subheader("Resultado da Análise")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Nível de Risco", f"{y_prob:.1%}")

        with col2:
            status = "🔴 ALTO RISCO" if y_prob > 5 else "🟢 BAIXO RISCO"
            st.write(f"**Status:** {status}")

        if y_prob > 7:
            st.warning(
                "⚠️ Este aluno apresenta indicadores críticos de defasagem. Recomenda-se intervenção psicopedagógica imediata.")

    except Exception as e:
        st.error(f"Erro ao prever: {e}")
    
