"""
This python module defines some useful functions
"""
import os
import yaml
from IPython.core.magic import register_cell_magic
import time

# folder to load config file
CONFIG_PATH = "../"

# Function to load yaml configuration file
def load_config(config_name):
    """
    Sets the configuration file path
    Args:
    config_name: Name of the configuration file in the directory
    Returns:
    Configuration file
    """
    with open(os.path.join(CONFIG_PATH, config_name), encoding="utf-8") as conf:
        config = yaml.safe_load(conf)
    return config

def prior_predictive_check(model):
    """
    Returns histogram of means of prior distribution
    Args:
    Model(fitted stan model)
    Returns:
    y_sim: prior distribution simulations
    fig: histogram 
    """
    complete_pooling_model_previa_fit.stan_variable("y_sim")
    

config_f = load_config("config.yml")
