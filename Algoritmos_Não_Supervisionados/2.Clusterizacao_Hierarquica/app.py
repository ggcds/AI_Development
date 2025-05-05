import streamlit as st
import pandas as pd

# carregar dados e colocar no cache do streamlit
@st.cache_data
def carregar_dados():
    return pd.read_csv('./datasets/clusterizacao_laptops.csv')

df = carregar_dados()

# sidebar para filtro
st.sidebar.header('Filtros')

# selecionar modelos
model = st.sidebar.selectbox('Selecionar Modelo', df['model'].unique())

# filtrar modelo
df_laptops_modelo = df[df['model'] == model]

# filtrar cluster do model escolhido
df_laptops_final = df[df['cluster'] == df_laptops_modelo.iloc[0]['cluster']]

# visualizar modelos
st.write('Recomendações de Modelos')
st.table(df_laptops_final)