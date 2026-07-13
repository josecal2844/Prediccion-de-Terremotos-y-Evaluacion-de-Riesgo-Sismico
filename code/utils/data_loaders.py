"""
Funciones y clases de carga de datos. Incluye los DataLoaders personalizados
para iterar sobre el HDF5 y el CSV de STEAD sin saturar la memoria.
"""

import os
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class STEADDataset(Dataset):
    """
    Dataset de PyTorch que carga las formas de onda desde un archivo HDF5 a partir
    de un DataFrame de metadatos (CSV filtrado). Lectura bajo demanda (lazy loading)
    para no saturar la RAM.
    """

    # Rutas por defecto al dataset STEAD (relativas a la carpeta `code/`)
    CSV_PATH = os.path.join("..", "STEAD", "features.csv")
    HDF5_PATH = os.path.join("..", "STEAD", "waveforms.hdf5")

    def __init__(self, df, hdf5_path = None, transform = None):
        """
        :param df: DataFrame de pandas filtrado con los eventos seleccionados.
        :param hdf5_path: Ruta al .hdf5. Si es None, usa STEADDataset.HDF5_PATH.
        :param transform: Opcional. Funciones de transformación de datos.
        """
        self.df = df.reset_index(drop = True)
        self.hdf5_path = hdf5_path if hdf5_path is not None else self.HDF5_PATH
        self.transform = transform
        self.file = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Apertura del HDF5 en el primer acceso, necesaria para la compatibilidad
        # con múltiples procesos de carga (num_workers > 0).
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')

        # Nombre de la traza correspondiente a la fila
        trace_name = self.df.iloc[idx]['trace_name']

        # Matriz de la traza (ruta interna de STEAD: 'data/<nombre>')
        dataset = self.file.get('data/' + str(trace_name))
        data = np.array(dataset)

        # STEAD almacena las trazas como (6000, 3). Las convoluciones 1D de PyTorch
        # esperan el formato (canales, longitud) = (3, 6000).
        data = data.T

        # Conversión a tensor float32
        tensor_data = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            tensor_data = self.transform(tensor_data)

        return tensor_data, trace_name

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
