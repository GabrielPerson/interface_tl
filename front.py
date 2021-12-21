import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import random
import shap 
shap.initjs()
import joblib

from  statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rcP = {'figure.figsize':(8, 5),
            'grid.color': 'grey',
            'axes.labelsize': 12,
            'font.size': 9}

plt.rcParams.update(rcP)

TITLE = 'Aprendizagem Descritiva Turboleads'
st.set_page_config(page_title=TITLE, layout='wide')

@st.cache
def Shap(keep_na):
    
    ## Modelo treinado (oct-18) + dados input modelo
    #dict_modelo = joblib.load('modelo.pkl')
    #modelo = dict_modelo['modelo']
    #treino_turboleads = pd.read_csv('amostra_turboleads.csv')

    treino_turboleads.loc[:, ['cliente_antigo_rac', 'cliente_antigo_sn','cliente_antigo_gf']] = treino_turboleads[['cliente_antigo_rac','cliente_antigo_sn','cliente_antigo_gf']].fillna('False')
    treino_turboleads.loc[:, ['n_vendas_rac_m', 'n_vendas_rac_d', 'vlr_gasto_rac', 'vlr_gasto_sn',
               'ca_rac_tempo', 'ca_sn_tempo', 'ca_gf_tempo',
               'ca_rac_ult_venda_tempo', 'ca_sn_ult_venda_tempo', 'ca_gf_ult_venda_tempo']] = treino_turboleads[['n_vendas_rac_m', 'n_vendas_rac_d', 'vlr_gasto_rac', 
                                                                                                                 'vlr_gasto_sn',
               'ca_rac_tempo', 'ca_sn_tempo', 'ca_gf_tempo',
               'ca_rac_ult_venda_tempo', 'ca_sn_ult_venda_tempo', 'ca_gf_ult_venda_tempo']].fillna(0)
    
    explainer = shap.TreeExplainer(modelo)
    
    if keep_na == 'Não': data = treino_turboleads.dropna(axis=0)    
    else: data = treino_turboleads
    shap_values = explainer.shap_values(data)
    shap_obj = explainer(data)
    
    data.columns = ['Tipo Pessoa','Evento','Telefone Valido','Email Valido',
                'Descricao Preenchida','Cliente Antigo RAC', 'Cliente Antigo SN',
                'Cliente Antigo GF', 'Vendas Rac Por Mes', 'Vendas RAC Por Dia', 
                'Valor Gasto RAC', 'Valor Gasto SN', 'Tempo Cliente RAC', 'Tempo Cliente SN',
                'Tempo Cliente GF','Tempo Ultima Venda RAC', 'Tempo Ultima Venda SN', 'Tempo Ultima Venda GF',
                'Quantidade de Carros na Proposta','Valor Medio da Proposta','Diferenca Valor Medio da Proposta',
                'QAP Proposta','Diferenca QAP Prosposta','Estado','Dia da Semana', 'Final de Semana', 'Indicacao',
                'Numero de Leads','Numero de Propostas','Numero de Canais','Numero de Dias do Primeiro Lead',
                'Numero de Dias do Ultimo Lead','Numero de Dias da Abertura de Conta','Numero de Dias da Primeira Proposta',
                'Numero de Dias da Ultima Proposta','Numero de Dias do Ultimo Evento', 'Pre Analise de Credito',
                'Valor Pre Aprovado','Descricao do Canal','Como encontrou', 'Origem Lead', 'Sistema Operacional - GA',
                'Page Views - GA', 'Tempo no Site - GA', 'Tipo de Dispositivo - GA', 'Canal - GA', 'Fonte - GA']
    
    
    return explainer, shap_values, shap_obj, data

@st.cache
def ScoreDiv():
    
    ## Dados predição turboleads + dados modelo (pré tuplas)
    df_pred = pd.read_csv('pred_leads_oct.csv')
    df_turbo = pd.read_csv('base_treino_turboleads.csv')
    merge_pred = df_turbo.merge(df_pred[['id_lead', 'vlr_previsao_conversao']], on='id_lead', how='inner')

    ## FILL NA
    merge_pred.loc[:, ['cliente_antigo_rac', 'cliente_antigo_sn','cliente_antigo_gf']] = merge_pred[['cliente_antigo_rac', 
                                                                                'cliente_antigo_sn','cliente_antigo_gf']].fillna('False')
    merge_pred.loc[:, ['n_vendas_rac_m', 'n_vendas_rac_d', 
                       'vlr_gasto_rac', 'vlr_gasto_sn']] = merge_pred[['n_vendas_rac_m', 'n_vendas_rac_d', 
                                                                       'vlr_gasto_rac', 'vlr_gasto_sn']].fillna(0)
    return merge_pred.drop_duplicates()

@st.cache
def ShapPlot(values, data):
    return shap.summary_plot(values, data)

st.title(TITLE)

## Sidebar -----------

#st.sidebar.markdown('### Seletores Dados Shap')
#keep_na = st.sidebar.selectbox('Utilizar base com valores nulos', options=['Sim', 'Não'])
keep_na = 'Sim'

## -------------------
#explainer, shap_values, shap_object, shap_data = Shap(keep_na)
df_pred = ScoreDiv()

cols_shapdata = ['Vendas Rac Por Mes', 'Vendas RAC Por Dia', 'Valor Gasto RAC', 'Valor Gasto SN', 
                 'Tempo Cliente RAC', 'Tempo Cliente SN', 'Tempo Cliente GF','Tempo Ultima Venda RAC', 'Tempo Ultima Venda SN', 'Tempo Ultima Venda GF',
                 'Quantidade de Carros na Proposta','Valor Medio da Proposta','Diferenca Valor Medio da Proposta',
                 'QAP Proposta','Diferenca QAP Prosposta', 'Numero de Dias do Primeiro Lead',
                 'Numero de Dias do Ultimo Lead','Numero de Dias da Abertura de Conta','Numero de Dias da Primeira Proposta',
                 'Numero de Dias da Ultima Proposta','Numero de Dias do Ultimo Evento',
                  'Valor Pre Aprovado','Page Views - GA', 'Tempo no Site - GA']
#cont_col_shapdata = st.sidebar.selectbox('Coluna Contínua Gráfico de Distribuição', options=cols_shapdata, index=1)

## Suporte nativo para linguagem markdown
'''
---
## Utilizando SHAP para Interpretabilidade do Modelo Treinado
O SHAP (SHapley Additive exPlanations) é uma ferramenta que utiliza de estratégias de Teoria dos Jogos para explicar os resultados de modelos de Aprendizado de Máquina.

O SHAP utiliza tanto o modelo treinado assim como os dados utilizados já formatados para identificar como cada _feature_ gera impacto na váriação no resultado esperado do modelo, tanto positiva quanto negativamente. Essas análises nos permitem identificar como a variação de valores das principais _features_ do modelo podem impactar os resultados, muitas destas _features_ ja identificadas na métrica _feature importance_.
'''


#f'''### Total de Linhas - {shap_data.shape[0]}'''
#st.write(shap_data.sample(20,replace=True))

'''
No gráfico a seguir temos um exemplo de aplicação do SHAP ao utilizar os dados do modelo TurboLeads junto ao modelo treinado. O Eixo X representa o grau de importancia que determinada _feature_ tem sobre o score do modelo, equanto o Eixo Y representa o valor absoluto dessa _feature_ (somente para features de caráter contínuo)
'''

#st.pyplot(shap.summary_plot(shap_values, shap_data))
#st.pyplot(ShapPlot(shap_values, shap_data))


"""row1_1, row1_2 = st.columns(2)
with row1_1:    
    #st.pyplot(shap.summary_plot(shap_values, shap_data))
    st.write('pato')

with row1_2:
    ecdf = ECDF(shap_data[cont_col_shapdata])
    fig, ax = plt.subplots()
    ax = plt.plot(ecdf.x, ecdf.y)
    plt.xlabel('Valor')
    plt.ylabel('Proporção')
    plt.title(f'Distribuição Cumulativa de {cont_col_shapdata}')
    st.pyplot(fig)"""

'''
No gráfico a seguir temos o impacto das principais _features_ sobre uma observação aleatória da base de treino.
'''

#rand_num = random.randrange(len(shap_object))
#lead = df.loc
#st.write(shap_data.loc[rand_num, ''])
#with row1_1:
#st.pyplot(shap.plots.waterfall(shap_object[rand_num]))

df_pred = ScoreDiv()

'''
---
## Seletor de Lead
'''


lead_select = st.sidebar.select_slider('Seletor de Lead', options = np.sort(df_pred.id_lead.unique()))

lead_filter = df_pred[df_pred.id_lead == lead_select].reset_index()


f'''#### Dados de Conversão do Lead {lead_select}'''

f'''
Número de Predições - {lead_filter.shape[0]}

Mínimo - {round(lead_filter.vlr_previsao_conversao.min(), 3)}

Média - {round(lead_filter.vlr_previsao_conversao.mean(), 3)}

Mediana - {round(lead_filter.vlr_previsao_conversao.median(), 3)}

Máximo - {round(lead_filter.vlr_previsao_conversao.max(), 3)}
'''

st.write(lead_filter)



'''
---
## Segmentação de Predições em Faixas de Score

Uma das estratégias adotadas para a análise das _features_ dos leads junto às predições do modelo é a segmentação dos leads em quantils de score de tamanhos semelhantes. Assim é possível analisar a frequência e o possível impacto (isolado) de algumas features no score dos leads.
'''
## DADOS PRED + TURBO


col_info = ['pessoa', 'evento']
col_plat = ['cliente_antigo_rac', 'cliente_antigo_sn','cliente_antigo_gf', 'n_vendas_rac_m', 'n_vendas_rac_d','vlr_gasto_rac', 'vlr_gasto_sn']
col_prop = ['qtd_carros_proposta','vlr_medio_proposta', 'qap_proposta','resultado_preanalise', 'valor_preaprovado']
col_desc = ['desc_canal', 'desc_como_encontrou', 'desc_origem_lead']
col_ga = ['operatingSystem', 'pageViews', 'timeOnSite', 'deviceCategory', 'channelGrouping', 'source']

#row1_1, row1_2 = st.columns((4,4))
#with row1_1:
f'''### Total de Linhas - {df_pred.shape[0]}'''
#st.write(shap_data.sample(20,replace=True))
#st.write(df_pred.head(10))

colunas_uteis = col_info + col_plat + col_prop + col_desc + col_ga + ['vlr_previsao_conversao']
df_util = df_pred.loc[:, colunas_uteis]
df_util = df_util.dropna(axis=0).drop_duplicates().sort_values('vlr_previsao_conversao').reset_index(drop=True)

## Sidebar -----------

st.sidebar.markdown('---')
st.sidebar.markdown('### Seletores Dados Predição + Base')

n_faixas = st.sidebar.selectbox('Numero de quantils para se dividir os Dados', options=[4, 5, 7, 10])
tipo_pessoa = st.sidebar.selectbox('Filtrar Base por Pessoa Física ou Jurídica', options=list(df_util['pessoa'].unique()) + ['Ambos'], index=2)
cont_col_preddata = st.sidebar.selectbox('Coluna Contínua Gráfico de Distribuição', options=df_util.select_dtypes(exclude=['object']).columns,index=1)
cat_col_preddata = st.sidebar.selectbox('Coluna Categórica Gráfico de Porcentagem', options=df_util.select_dtypes(include=['object']).columns,index=1)

## -------------------
if tipo_pessoa != 'Ambos':
    df_util = df_util[df_util.pessoa == tipo_pessoa]

pred_qcut = pd.qcut(df_util['vlr_previsao_conversao'], q=n_faixas).sort_values().reset_index(drop=True).unique() ## divisoes de tamanho "igual"
last_quantile = pred_qcut[-1] ## quartil de maior score

## Divisão da pagina em colunas para organização das visualizações
row_chart1, row_chart2 = st.columns(2)
with row_chart1:
    filt = df_util[df_util.vlr_previsao_conversao.between(last_quantile.left, last_quantile.right)]
    ecdf = ECDF(filt[cont_col_preddata])
    fig, ax = plt.subplots()
    ax = plt.plot(ecdf.x, ecdf.y)
    plt.xlabel('Valor')
    plt.ylabel('Proporção')
    plt.title(f'Distribuição Cumulativa de {cont_col_preddata} - Faixa de Score: {last_quantile}')
    st.pyplot(fig)
    
with row_chart2:
    filt = df_util[df_util.vlr_previsao_conversao.between(pred_qcut[-2].left, pred_qcut[-2].right)]
    ecdf = ECDF(filt[cont_col_preddata])
    fig, ax = plt.subplots()
    ax = plt.plot(ecdf.x, ecdf.y)
    plt.xlabel('Valor')
    plt.ylabel('Proporção')
    plt.title(f'Distribuição Cumulativa de {cont_col_preddata} - Faixa de Score: {pred_qcut[-2]}')
    st.pyplot(fig)
    
with row_chart1:
    filt = df_util[df_util.vlr_previsao_conversao.between(last_quantile.left, last_quantile.right)]
    fig, ax = plt.subplots()
    ax = (filt[cat_col_preddata].value_counts()/filt.shape[0]*100)[:10].plot(kind='barh', title='faixa de score -- ' + str(last_quantile)+' -- '+str(filt.shape[0])+' obs')
    #ax = sns.countplot(y = cat_col_preddata, data = filt,order = filt[cat_col_preddata].value_counts().index)
    ax.set_title(f'Gráfico de Frequência de {cat_col_preddata} - Faixa de Score: {last_quantile}')
    #ax.set_ylabel(cat_col_preddata)
    ax.set_xlabel('Porcentagem')
    st.pyplot(fig)

with row_chart2:
    filt = df_util[df_util.vlr_previsao_conversao.between(pred_qcut[-2].left, pred_qcut[-2].right)]
    fig, ax = plt.subplots()
    ax = (filt[cat_col_preddata].value_counts()/filt.shape[0]*100)[:10].plot(kind='barh')
    #ax = sns.countplot(y = cat_col_preddata, data = filt, order = filt[cat_col_preddata].value_counts().index)
    ax.set_title(f'Gráfico de Frequência de {cat_col_preddata} - Faixa de Score: {pred_qcut[-2]}')
    #ax.set_ylabel(cat_col_preddata)
    ax.set_xlabel('Porcentagem')
    st.pyplot(fig)