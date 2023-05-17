"""
Este modulo de python ajusta la predictiva posterior y genera N=978 * 1000 simulaciones y establece una funcion para graficar elementos de comparacion relevantes para los diagnosticos posteriores
predictive checks
"""
import os
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from outils import load_config
import matplotlib.patches as patches
import arviz as az
import xarray as xr
import matplotlib.ticker as ticker
import seaborn as sns


CURRENT = os.getcwd()
ROOT = os.path.dirname(CURRENT)

# Load config file calling load_config function
config_f = load_config("config.yml")



def ajuste_posterior_predictiva(archivo, modelstring, data, datos_originales, n_samples, n_chains):
    """
    Genera simulaciones de la predictiva posterior
    Args: 
    archivo (string): nombre del archivo stan en el que se guardara el modelo
    modelstring (strin): string del modelo en codigo de stan
    data(dictionary): dictionnary containing data to fit stan model
    datos_originales(pandas dataframe): dataframe de pandas con los datos
    parametros(list): Lista de parametros sobre los cuales hace inferencia el modelo (acepta regex).
    Returns:
    parametros(pandas dataframe): dataframe con las simulaciones de los parametros del modelo.
    predicciones_natural (pandas dataframe): dataframe con las 900 y pico simulaciones generadas 
    en escala natural (pico-curies por litro)
    estadisticas_replicaciones(pandas dataframe): dataframe con una columna de medias de 
    predicciones y de desviaciones estandar. 
    summary_modelo(pandas dataframe): dataframe con estadisticas de salida del summary.
    muestras_aleatorias(pandas dataframe): subset de simulaciones para graficar
    
    """
    # Creo el archivo de Stan
    modelo = os.path.join(ROOT,config_f["models_directory"], archivo)
    with open(modelo, 'w') as f:
        f.write(modelstring)
    # Compilo el modelo
    compilacion=CmdStanModel(stan_file=os.path.join(ROOT, config_f["models_directory"], archivo))
    # Hago el ajuste, corro las cadenas
    ajuste = compilacion.sample(
    data=data, 
    show_progress=False, 
    chains=n_chains,
    iter_warmup= 1000,
    iter_sampling=n_samples)
    predicciones_locales = ajuste.draws_pd(vars=['y1_sim'])
    muestras_aleatorias_locales = predicciones_locales\
                   .sample(n=15, random_state=34)\
                   .transpose()\
                   .set_index(datos_originales.index)
    muestras_aleatorias_locales.insert(0, "datos_originales", datos_originales['home_team_score'])
    media_locales=predicciones_locales.transpose().mean()
    desviacion_estandar_locales = predicciones_locales.transpose().std()
    # Crear un nuevo DataFrame con las columnas 'Media' y 'Desviación Estándar'
    estadisticas_replicaciones_locales = pd.DataFrame({'Media': media_locales, 
                                                       'Desviación Estándar':desviacion_estandar_locales})
    predicciones_visitantes = ajuste.draws_pd(vars=['y2_sim'])
    muestras_aleatorias_visitantes = predicciones_visitantes\
                   .sample(n=15, random_state=34)\
                   .transpose()\
                   .set_index(datos_originales.index)
    muestras_aleatorias_visitantes.insert(0, "datos_originales", datos_originales['away_team_score'])
    media_visitantes=predicciones_visitantes.transpose().mean()
    desviacion_estandar_visitantes = predicciones_visitantes.transpose().std()
    # Crear un nuevo DataFrame con las columnas 'Media' y 'Desviación Estándar'
    estadisticas_replicaciones_visitantes = pd.DataFrame(
        {'Media': media_visitantes, 
        'Desviación Estándar': desviacion_estandar_visitantes})
    return estadisticas_replicaciones_locales, muestras_aleatorias_locales, ajuste, estadisticas_replicaciones_visitantes, muestras_aleatorias_visitantes


def grafica_barras_replicaciones_observados(muestras_aleatorias):
    """
    Grafica los datos observados del problema contra 14 muestras aleatorias de las replicaciones realizadas
    """
    # Crear un panel único para los gráficos de barras
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

    # Iterar sobre cada columna y crear un gráfico de barras en el panel
    for i, column in enumerate(muestras_aleatorias.columns):
        ax = axes[i // 4, i % 4]

        # Graficar el primer gráfico de barras con un color diferente
        if i == 0:
            sns.countplot(data=muestras_aleatorias, x=column, color="red", alpha=0.8, ax=ax)
            ax.set_title("Datos observados", color="red")
        else:
            sns.countplot(data=muestras_aleatorias, x=column, color="teal", alpha=0.8, ax=ax)

        # Ajustar las etiquetas del eje x
        if i < 12:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Goles")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

    fig.suptitle("Gráficos de barras de datos observados\n y replicaciones", fontsize=15)
    plt.tight_layout()


def grafica_media_std_replicaciones(estadisticas_replicaciones, datos_originales, modo):
    if modo=="local":
        MODO=datos_originales["home_team_score"]
    else:
        MODO=datos_originales["away_team_score"]

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))     
    fig.suptitle('Media y desviación estándar de replicaciones\n y datos observados', fontsize=15)
    plt.subplots_adjust(top=0.80)
    ax1.hist(estadisticas_replicaciones["Media"], bins=40, alpha=0.50, color="teal", edgecolor="white")
    ax1.set_title("Medias")
    ax1.axvline(MODO.mean(), color="red")
    ax1.legend([f"Media de datos\n observados"], loc='upper right', frameon=False, framealpha=1, fontsize=9)
    ax1.text(0.70, 0.80, "Replicaciones", transform=ax1.transAxes, color="black",fontsize=9)
    rect1 = patches.Rectangle((0.60, 0.78), 0.07, 0.07, alpha=0.50, 
                                  facecolor="teal", edgecolor="none", transform=ax1.transAxes)
    ax1.add_patch(rect1)
    ax2.hist(estadisticas_replicaciones["Desviación Estándar"],bins=40, alpha=0.50, color="teal", edgecolor="white")
    ax2.set_title("Desviación estándar")
    ax2.axvline(MODO.std(), color="red")
    ax2.legend([f"Desviación estándar\n de datos observados"], loc='upper right', 
                   frameon=False, framealpha=1,fontsize=9)
    ax2.text(0.63, 0.80, "Replicaciones", transform=ax2.transAxes, color="black",fontsize=9)
    rect2 = patches.Rectangle((0.55, 0.78), 0.07, 0.07, alpha=0.50,
                                  facecolor="teal", edgecolor="none", transform=ax2.transAxes)
    ax2.add_patch(rect2);
    
def grafica_scatter_estadisticas(modo, estadisticas_replicaciones, datos_originales):
    if modo=="local":
        MODO=datos_originales["home_team_score"]
    else:
        MODO=datos_originales["away_team_score"]

    plt.scatter(estadisticas_replicaciones["Media"], estadisticas_replicaciones["Desviación Estándar"], 
                alpha=0.1, edgecolors="white", s=30, color="teal")

    # Señalar la primera entrada con una flecha y texto
    plt.annotate("Datos observados", xy=(MODO.mean(), 
                                         MODO.std()), xytext=(10, -20),
                 textcoords="offset points", arrowprops=dict(arrowstyle="->",
                 connectionstyle="arc3,rad=0.5"), fontsize=10, color="red")
    plt.scatter(MODO.mean(), MODO.std(), color="red")
    plt.xlabel("Media")
    plt.ylabel("Desviación Estándar")
    plt.title("Distribución de estadísticas:\n replicaciones de las observaciones")

def calcula_metricas(ajuste, parametros, n_chains, n_draws_per_chain):
    
    # Extraer la log-verosimilitud
    log_lik = ajuste.draws_pd(vars=['log_lik'])
    log_lik_np = log_lik.to_numpy()
    log_lik_np = log_lik_np.reshape((n_chains, n_draws_per_chain, -1))  
    log_lik_xr = xr.DataArray(log_lik_np, dims=["chain", "draw", "log_lik_dim"])
    
    # Extraer parámetros
    posterior_dict = {}
    for param in parametros:
        param_values = ajuste.stan_variable(param)
        param_dim = len(param_values.shape)  # Obtener la dimensión del parámetro
        if param_dim > 1:
            # Si el parámetro tiene más de una dimensión (es un vector o matriz)
            param_np = param_values.reshape((n_chains, n_draws_per_chain, -1))
            param_xr = xr.DataArray(param_np, dims=["chain", "draw", f"{param}_dim"])
        else:
            param_np = param_values.reshape((n_chains, n_draws_per_chain))
            param_xr = xr.DataArray(param_np, dims=["chain", "draw"])
        posterior_dict[param] = param_xr

    # Crear objeto InferenceData
    idata = az.from_dict(posterior=posterior_dict, log_likelihood={"y": log_lik_xr})

    # Calcular WAIC y LOO
    waic = az.waic(idata)
    loo = az.loo(idata)
    
    return waic, loo



def loglik_posterior_predictiva(archivo, modelstring, data, datos_originales, parametros, n_samples, n_chains):
    """
    Genera simulaciones de la predictiva posterior
    Args: 
    archivo (string): nombre del archivo stan en el que se guardara el modelo
    modelstring (strin): string del modelo en codigo de stan
    data(dictionary): dictionnary containing data to fit stan model
    datos_originales(pandas dataframe): dataframe de pandas con los datos
    parametros(list): Lista de parametros sobre los cuales hace inferencia el modelo (acepta regex).
    Returns:
    parametros(pandas dataframe): dataframe con las simulaciones de los parametros del modelo.
    predicciones_natural (pandas dataframe): dataframe con las 900 y pico simulaciones generadas 
    en escala natural (pico-curies por litro)
    estadisticas_replicaciones(pandas dataframe): dataframe con una columna de medias de 
    predicciones y de desviaciones estandar. 
    summary_modelo(pandas dataframe): dataframe con estadisticas de salida del summary.
    muestras_aleatorias(pandas dataframe): subset de simulaciones para graficar
    
    """
    # Creo el archivo de Stan
    modelo = os.path.join(config_f["models_directory"], archivo)
    with open(modelo, 'w') as f:
        f.write(modelstring)
    # Compilo el modelo
    compilacion=CmdStanModel(stan_file=os.path.join(config_f["models_directory"], archivo))
    # Hago el ajuste, corro las cadenas
    ajuste = compilacion.sample(
    data=data, 
    show_progress=False, 
    chains=n_chains,
    iter_warmup= 1000,
    iter_sampling=n_samples)
    parametros = ajuste.draws_pd(vars=parametros)

    return parametros, ajuste