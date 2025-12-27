import os,json,numpy as np,pandas as pd,streamlit as st,joblib

st.set_page_config(page_title="Predição de Obesidade",page_icon="🏥",layout="centered")
st.title("🏥 Predição de Obesidade — App")

ART_DIR="model"

def find_model_and_meta():
    if not os.path.isdir(ART_DIR): return None,None
    joblibs=[os.path.join(ART_DIR,f) for f in os.listdir(ART_DIR) if f.endswith(".joblib")]
    if not joblibs: return None,None
    model_path=max(joblibs,key=os.path.getmtime)
    finfo_path=os.path.join(ART_DIR,"feature_info.json")
    if not os.path.exists(finfo_path): finfo_path=None
    return model_path,finfo_path

MODEL_PATH,FEATURE_INFO_PATH=find_model_and_meta()

if MODEL_PATH is None or FEATURE_INFO_PATH is None:
    st.error("Nao encontrei os artefatos em 'model/'. Preciso de 'best_model_*.joblib' e 'feature_info.json'."); st.stop()
@st.cache_resource

def load_artifacts(model_path,finfo_path):
    model=joblib.load(model_path)
    with open(finfo_path,"r",encoding="utf-8") as f: finfo=json.load(f)
    return model,finfo

model,finfo=load_artifacts(MODEL_PATH,FEATURE_INFO_PATH)

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

num_cols=finfo.get("num_cols",[]); cat_cols=finfo.get("cat_cols",[]); all_features=num_cols+cat_cols

st.caption(f"Modelo: {os.path.basename(MODEL_PATH)}")

tab_pred,tab_ins=st.tabs(["🔮 Predição","📊 Insights"])

with tab_pred:
    st.subheader("Insira seus dados pessoais abaixo")
    cols=st.columns(1); vals={}

    for i,c in enumerate(num_cols):
        with cols[0]:
            default=0
            if c=="Age": default=30
            if c=="Height": default=1.70
            if c=="Weight": default=70.0
            if c=="BMI": continue
            vals[c]=st.number_input(getFieldName(c),value=default)

    for i,c in enumerate(cat_cols):
        with cols[0]:
            vals[c]=st.text_input(getFieldName(c),value="")

    if st.button("Efetuar análise preditiva"):
        x=pd.DataFrame([vals],columns=all_features)
        try:
            y_pred=model.predict(x)[0]
            st.success(f"Predição: **{y_pred}**")
            try:
                proba=model.predict_proba(x)[0]; classes=model.classes_
                st.write("Confiança (top 3):")
                top=np.argsort(proba)[::-1][:3]
                for i in top: st.write(f"- {classes[i]}: {proba[i]:.2%}")
            except Exception: pass
        except Exception as e:
            st.error(f"Erro ao prever: {e}")

    st.markdown("---")
    st.subheader("Lote (CSV)")
    up=st.file_uploader("Envie CSV com as colunas de entrada",type=["csv"])

    if up is not None:
        df_in=pd.read_csv(up); missing=[c for c in all_features if c not in df_in.columns]
        if missing: st.warning(f"Colunas faltantes no CSV: {missing}")
        try:
            cols_used=[c for c in all_features if c in df_in.columns]
            preds=model.predict(df_in[cols_used]); out=df_in.copy(); out["prediction"]=preds
            st.dataframe(out.head(50))
            st.download_button("Baixar CSV", out.to_csv(index=False).encode("utf-8"), file_name="predicoes.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erro no lote: {e}")

with tab_ins:
    st.subheader("Features do modelo")
    st.write("Numericas:", num_cols)
    st.write("Categoricas:", cat_cols)
    st.info("Para graficos detalhados e SHAP utilize este notebook.")