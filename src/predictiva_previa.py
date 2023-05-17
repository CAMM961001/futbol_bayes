import os
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import warnings
import BFG3000 as bfg
warnings.filterwarnings('ignore')

def ajuste_previa_predictiva(modelo, data, cadenas, muestras, burnin):
    """
    Genera simulaciones de la predictiva previa
    Args: 
    archivo (string): nombre del archivo stan en el que se guardara el modelo
    modelstring (strin): string del modelo en codigo de stan
    data(dictionary): dictionnary containing data to fit stan model
    Returns:
    summary(pandas dataframe): dataframe con resultados de diagnostico de las simulaciones
    y1_sim_df(pandas dataframe), y2_sim_df(pandas dataframe): dataframes con las simulaciones generadas
    column_means(array): promedio de cada una de las simulaciones
    column_max: observaciones maximas de cada una de las simulaciones
    column_min: observaciones minimas de cada una de las simulaciones
    pct_95: percentil 95 de cada una de las simulaciones
    pct_99: percentil 99 de cada una de las simulaciones
    """
    compile_ = CmdStanModel(
    stan_file=modelo
    ,compile=True)
    # Hago el ajuste, corro las cadenas
    ajuste = compile_.sample(
    data=data, 
    show_progress=False, 
    chains=cadenas,
    iter_warmup= burnin,
    iter_sampling=muestras)
    #Saco el resumen de la cadena
    summary=ajuste.summary().round(2)
    # Extraigo las simulaciones
    y1_sim = ajuste.stan_variable("y1_sim")
    y2_sim = ajuste.stan_variable("y2_sim")
    #Hago un dataframe
    y1_sim_df = pd.DataFrame(y1_sim)
    y2_sim_df = pd.DataFrame(y2_sim)
    # Asignar nombres de columna basados en los índices
    column_names = [f"y1_sim[{i+1}]" for i in range(y1_sim.shape[1])]
    y1_sim_df.columns = column_names
    column_names = [f"y2_sim[{i+1}]" for i in range(y2_sim.shape[1])]
    y2_sim_df.columns = column_names
    # Calculo los promedios de las simulaciones
    column_means_y1 = y1_sim_df.mean(axis=0)
    column_means_y2 = y2_sim_df.mean(axis=0)
    # Column max 
    column_max_y1 = y1_sim_df.max(axis=0)
    column_max_y2 = y2_sim_df.max(axis=0)
    # Column min
    column_min_y1 = y1_sim_df.min(axis=0)
    column_min_y2 = y2_sim_df.min(axis=0)
    # Percentil 95
    pct_95_y1 = y1_sim_df.apply(lambda x: x.quantile(0.95), axis=0)
    pct_95_y2 = y2_sim_df.apply(lambda x: x.quantile(0.95), axis=0)
    # Percentil 99
    pct_99_y1 = y1_sim_df.apply(lambda x: x.quantile(0.99), axis=0)
    pct_99_y2 = y2_sim_df.apply(lambda x: x.quantile(0.99), axis=0)
    
    return summary, y1_sim_df, y2_sim_df, column_means_y1, column_means_y2, column_max_y1, column_max_y2, column_min_y1, column_min_y2, pct_95_y1, pct_95_y2, pct_99_y1, pct_99_y2

    
def plot_histogram(data, title, ylog_scale=False, xlog_scale=False, data_lim=False):
    """
    Graficas las estadisticas requeridas de la predictiva previa
    Args:
    data(array): estadistica de las simulaciones de la previa
    title(string): titulo de la grafica
    log_scale(boolean): indicador para graficar el y en escala logaritmica
    """
    if data_lim:
        data = data[data < 100]
    # Crear el histograma
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, alpha=0.75, color="teal", edgecolor="white")
    # Personalizar el gráfico
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Valor",fontsize=15)
    ax.set_ylabel("Frecuencia", fontsize=15)
    
    # Establecer la escala de los ejes
    if ylog_scale:
        ax.set_yscale('log')
    
    # Establecer la escala de los ejes
    if xlog_scale:
        ax.set_xscale('log')