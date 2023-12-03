import streamlit as st
import hotmodel
st.title("Descrição do problema")


st.markdown(
    """
    Sua tarefa:

    1. Baseado no dataset disponibilizado, identifique qual a melhor variante do
    experimento para todos os usuários. Em outras palavras, se tivesse que
    escolher apenas uma versão do experimento, qual delas seria?
    2. Crie um modelo que recomendará uma variante do teste para um usuário,
    a fim de maximizar sua monetização, mas sem haver perda de engajamento.
    """
)


