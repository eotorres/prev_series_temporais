import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.set_page_config(page_title="Forecast",
                   initial_sidebar_state="collapsed",
                   page_icon="🔮")

st.markdown("<h1 style='text-align: center;'>📈 Previsão automatizada de séries temporais</h1>"
	        ,unsafe_allow_html=True)

"""
Este aplicativo de dados usa a biblioteca de código aberto do Facebook, Prophet, para gerar automaticamente valores de previsão futura a partir de um conjunto de dados importado.
Você poderá importar seus dados de um arquivo CSV e o separador precisa ser por ';', visualizar tendências e recursos, analisar o desempenho da previsão e, finalmente, baixar a previsão criada 😵
"""

"""
### Etapa 1: Importar dados 🏋️
"""
df = st.file_uploader('Importe o arquivo csv da série temporal aqui. As colunas devem ser rotuladas como ds e y. A entrada para o Prophet é sempre um dataframe com duas colunas: ds e y. A coluna ds (datestamp) deve ter um formato esperado pelo Pandas, idealmente AAAA-MM-DD para uma data ou AAAA-MM-DD HH:MM:SS para um carimbo de data/hora. A coluna y deve ser numérica e representa a medição que desejamos prever.', type='csv')
st.info(
            f"""
                👆 Carregue primeiro um arquivo .csv. Exemplo para experimentar:  
                [peyton_manning_wiki_ts.csv](https://raw.githubusercontent.com/zachrenwick/streamlit_forecasting_app/master/example_data/example_wp_log_peyton_manning.csv)
                """
        )
if df is not None:
    data = pd.read_csv(df,sep=';')
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')     
    st.write(data)    
    max_date = data['ds'].max()    
    if st.checkbox('Chart data', key='show'):
        with st.spinner('Plotting data..'):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(data)
            with col2:
                st.write("Dataframe description:")
                st.write(data.describe())

    try:
        line_chart = alt.Chart(data).mark_line().encode(
          x='ds:T', y="y:Q",
          tooltip=['ds:T',
                   'y']).properties(title="Time series preview").interactive()
        st.altair_chart(line_chart, use_container_width=True)
    except:
        st.line_chart(data['y'], use_container_width=True, height=300)

"""
### Etapa 2: Selecione o Período/horizonte de previsão 🛠️ 
Lembre-se de que as previsões se tornam menos precisas com horizontes de previsão maiores 
"""

period_input = st.number_input('Quantos períodos você gostaria de prever no futuro?',
min_value = 1, max_value = 365)
initial_input = st.number_input('Período inicial (em dias)', min_value=1, max_value=365, value=30)
horizon_input = st.number_input('Horizonte (em dias)', min_value=1, max_value=365, value=365)

"""
### Etapa3: Validação do modelo 🧪
"""
if df is not None:
    m = Prophet()
    m.fit(data)
    cv_results = cross_validation(m, 
                                  initial=f"{initial_input} days",
                                  period=f"{period_input} days",
                                  horizon=f"{horizon_input} days")
    metrics = performance_metrics(cv_results)        
    st.write("MSE: ", metrics['mse'].mean())
    st.write("MAE: ", metrics['mae'].mean())
    st.write("MAPE: ", metrics['mape'].mean())
    st.write("MdAPE: ", metrics['mdape'].mean())
    st.write(plot_cross_validation_metric(cv_results, metric='mape'))
    st.write(plot_cross_validation_metric(cv_results, metric='mse'))
    st.write(plot_cross_validation_metric(cv_results, metric='mae'))
    st.write(plot_cross_validation_metric(cv_results, metric='mdape'))    
"""
### Etapa 4: Visualizar dados de previsão 🧙🏻

O visual abaixo mostra valores futuros previstos. 
"yhat" é o valor previsto e os limites superior e 
inferior são (por padrão) intervalos de confiança de 80%.
"""
if df is not None:
    future = m.make_future_dataframe(periods=period_input)    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)    
    """
    O próximo visual mostra os valores reais (pontos pretos) e 
    previstos (linha azul) ao longo do tempo.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)
    """
    Os próximos visuais mostram uma tendência de alto nível de valores previstos, 
    tendências de dia da semana e tendências anuais (se o conjunto de dados cobrir vários anos). 
    A área sombreada em azul representa os intervalos de confiança superior e inferior.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)

"""
### Passo 5: Baixar os dados de previsão ✨
O link abaixo permite que você baixe a previsão recém-criada para o seu computador 
para posterior análise e uso.
"""
if df is not None:    
    def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(fcst_filtered)
    st.download_button("Download",
                       csv,
                       f"forecast_data.csv",
                       "text/csv",
                       key='download-csv')            
            
    #csv_exp = fcst_filtered.to_csv(index=False)    
    # When no file name is given, pandas returns the CSV as a string, nice.
    #b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (clique com o botão direito do mouse e salvar link como ** <forecast_name>.csv**)'
    #st.markdown(href, unsafe_allow_html=True)
