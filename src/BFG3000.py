import os
import yaml
import numpy as np
import pandas as pd

from shutil import copy
from cmdstanpy import CmdStanModel


class StanProject:
    def __init__(self, name) -> None:
        # Inicializar variables estáticas
        self.NAME = name
        self.CURRENT = os.path.dirname(__file__)
        self.ROOT = os.path.dirname(self.CURRENT)

        # Archivo de configuración
        with open(os.path.join(self.ROOT, 'config.yml'), 'r') as file_:
            self.config = yaml.safe_load(file_)
        file_.close()


    def create_stan_project(self, dir):
        """
        Función para crear un proyecto de stan en un directoria
        especificado.

        Este método genera una plantilla de stan en un directorio
        con el mismo nombre del proyecto
        """
        dir = os.path.join(dir, self.NAME)
        dir = os.path.join(self.ROOT, dir)
        new_ = os.path.join(dir, f'{self.NAME}.stan')

        try:
            # Se crea directorio de archivos
            os.mkdir(dir)

            # Cargar plantilla stan en memoria
            template = os.path.join(self.CURRENT, '__template__.stan')

            # Crear plantilla de stan en directorio
            copy(template, dir)

            # Cambio de nombre de archivo
            old_ = os.path.join(dir, '__template__.stan')
            os.rename(old_, new_)

            print('Proyecto creado')

        except FileExistsError:
            print(f'Ya existe un proyecto con el nombre {self.NAME}')

        return dir, new_


if __name__ == '__main__':
    name = 'iter2_jera'
    sp_ = StanProject(name)
    dir_, model_ = sp_.create_stan_project(dir='modelos')
