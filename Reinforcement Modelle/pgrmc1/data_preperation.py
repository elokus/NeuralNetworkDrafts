import numpy as np
import pandas as pd

class data_preperation():
    def __init__(self, meta_file, prote_file):
        self.metab = pd.read_csv(meta_file)
        self.prote = pd.read_csv(prote_file)
        self.df_names = ["Proteomics", "Metabolomics"]
        self.data_array = [self.prote, self.metab]
        self.print_shape()
        #self.expre = pd.read_csv(expre_file)

    def filter_by_column(self, column="lineage_1", filter="Breast"):
        self.metab = self.metab[self.metab[column]==filter]
        self.prote = self.prote[self.prote[column]==filter]
        for name in self.df_names:
            print(f"Dataset {name} filtered column: {column} for {filter}")
        self.print_shape()

    def print_shape(self):
        print(f"Metabolomics Data Shape: {self.metab.shape}  |  Proteomics Data Shape: {self.prote.shape}")

    def matching_celllines(self):
        matching_cells = []
        for id in self.metab["depmap_id"].values:
            if id in self.prote["depmap_id"].values:
                matching_cells.append(id)
        self.metab = self.metab[self.metab.depmap_id.isin(matching_cells)]
        self.prote = self.prote[self.prote.depmap_id.isin(matching_cells)]
        print("Celllines matched........")
        self.print_shape()







filename_metabolomics ="data/Metabolomics.csv"
filename_proteomics = "data/Proteomics.csv"

data = data_preperation(filename_metabolomics, filename_proteomics)
data.filter_by_column()
data.matching_celllines()
