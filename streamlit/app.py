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
st.title("🎓 Predição de Risco de Defasagem")

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
        'Height': 'Altura'
    }
    return correspondencias.get(value, "Nome não encontrado no mapeamento: " + value)


def getResultValue(value):
    correspondencias = {
        '0': 'Defasagem leve',
        '1': 'Defasagem moderada',
    }
    return correspondencias.get(value, "Resultado não encontrado no mapeamento: " + value)


num_cols = finfo.get("num_cols", [])
cat_cols = finfo.get("cat_cols", [])
all_features = num_cols+cat_cols

st.caption(f"Modelo: {os.path.basename(MODEL_PATH)}")

tab_pred, tab_ins = st.tabs(["🔮 Predição"])

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
            if y_pred > 0.7:
                st.error("⚠️ Aluno em alto risco de defasagem!")
            else:
                st.success(
                    f"Risco de Defasagem: **{getResultValue(y_pred)}**")
        except Exception as e:
            st.error(f"Erro ao prever: {e}")
