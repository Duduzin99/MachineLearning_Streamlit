import streamlit as st
import pandas as pd
import joblib
import os


st.title("Hallo Kaiser")

#bosta = st.text_input("qual o teu nome pvt", vakue="digite aqui")
#if st.button(label="clique aqui"):
#    st.success(f"seja bem vindo {bosta}")

#ETAPA 1 - definição de features
FEATURES_NAMES = [ 
    'Nota_P1',
    'Nota_P2',
    'Media_Trabalhos',
    'Frequencia',
    'Reprovacoes_Anteriores',
    'Acessos_Plataforma_Mes'
]

COLUNAS_HISTORICO = FEATURES_NAMES + ["Previsao_Resultado", 'Prob_Aprovado', 'Prob_Reprovado']

#criar uma sessão
if 'historico_previsoes' not in st.session_state:
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)


#ETAPA 2 - carregamento do moledo para o nosso front end
#st.cache_resource para carregar o modelo apenas uma vez
#otimizando o desempenho do aplicativo

@st.cache_resource
def carregar_modelo(caminho_modelo = "modelo_previsão_desempenho.joblib"):
    try:
        if os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo
        else:
            st.error(f"erro epico do {caminho_modelo}")
            st.warning("por favor, execute o script 'modelo_treinamento.py' para gerar o modelo " )
    except Exception as e:
        st.error("erro exotico pprt")

pipeline_modelo = carregar_modelo()

st.set_page_config(layout='centered', page_title='previsão de notas')

st.title("previsor de desempenho academico")
st.markdown( """
    essa ferramenta usa inteligencia artifical para prever o status final (aprovado ou reprovado de um aluno com base em seu desempenho parcial
""")

#ETAPA 3 - formulario de entrada

if pipeline_modelo is not None:
    #utilizar um formulario para agrupar as entradas e o botao
    with st.form('formulario_previsao'):
        st.subheader('insira as notas e metricas do aluno')

        col1, col2 = st.columns(2)

        with col1:
            nota_p1 = st.slider("nota da prova 1 (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            media_trabalho = st.slider("media dos trabalhos ( 0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            numero_reprovacoes = st.number_input('reprovações anteriores', min_value=0.0, max_value=10.0, value=5.0, step=1.0)
        with col2:
            nota_p2 = st.slider("nota da prova 2 (0 a 10", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            frequencia = st.slider("frequencia (%)", min_value=0.0, max_value=100.0, value=75.0, step=5.0)
            acesso_mes = st.number_input("media de acesso a plataforma (por mes)", min_value=0.0, max_value=100.0, value=1.0, step=1.0)

        submitted = st.form_submit_button("realizar previsão")
    
    if submitted:

        features_name = [
            'Nota_P1',
            'Nota_P2',
            'Media_Trabalhos',
            'Frequencia',
            'Reprovacoes_Anteriores',
            'Acessos_Plataforma_Mes'
        ]

        #criação de um dataframe a partir dos dados inseridos
        dados_alunos = pd.DataFrame(
            [(nota_p1, nota_p2, media_trabalho, frequencia, numero_reprovacoes, acesso_mes)]
        )

        st.info("processando dados e realizando a previsão...")

        try:

        
            #realizar a previsão ([0] ou [1])
            previsao = pipeline_modelo.predict(dados_alunos)

            #obter a probabilidade ()
            probabilidade = pipeline_modelo.predict_proba(dados_alunos)

            prob_reprovados = probabilidade[0][0]
            prob_aprovados = probabilidade[0][1]
            resultado_texto = "APROVADO" if previsao[0] == 1 else "REPROVADO"

            #EXIBIR OS RESULTADO
            st.subheader("resultado da previsão")
            if previsao[0] == 1:
                st.success("previsão: Aprovado")
                st.markdown(f"""
                            com base nos resultados fornecidos, o modelo prevê que o aluno tem: {prob_aprovados*100:.2f}% de chance de ser aprovado
                            
                            Chance de reprovação: {prob_reprovados*100:.2f}
                            """)
            else:
                st.error("previsão: Reprovado (Area de Risco)")
                st.markdown(f"""
                            com base nos resultados fornecidos, o modelo prevê que o aluno tem: {prob_reprovados*100:.2f}% de chance de ser reprovado
                            
                            Chance de aprovação: {prob_aprovados*100:.2f}%"
                """)

            nova_linha_dict = { 
                'Nota_P1': nota_p1,
                'Nota_P2': nota_p2,
                'Media_Trabalhos': media_trabalho,
                'Frequencia':frequencia,
                'Reprovacoes_Anteriores': numero_reprovacoes,
                'Acessos_Plataforma_Mes': acesso_mes,
                'Previsao_Resultado': resultado_texto,
                'Prob_Aprovado': prob_aprovados,
                'Prob_Reprovado': prob_reprovados
            }
        
            nova_linha_df = pd.DataFrame([nova_linha_dict], columns=nova_linha_dict)

            st.session_state.historico_previsoes = pd.concat([st.session_state.historico_previsoes, nova_linha_df],
                                                             ignore_index=True
            )                                                 

        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}")
            st.error("verifique se os nomes da colunas correspondem aos nomes utilizados nos treinos")

    st.subheader("Historico de previsões realizadas na seção: ")
    if st.session_state.historico_previsoes. empty:
        st.write("não foi possível exibir os valores!")
    else:
        st.dataframe(st.session_state.historico_previsoes, use_container_width=True)

        if st.button("Limpar historico"):
            st.session_state.historico_previsoes = pd.DataFrame(columns= COLUNAS_HISTORICO)

            st.rerun()

else:    
    st.warning("O aplicativo não pode fazer previsões porque o modelo não foi carregado!")
