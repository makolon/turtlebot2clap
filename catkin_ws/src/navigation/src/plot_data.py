import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = sorted(glob.glob("csvfile/*.csv"))
for f in file_name:
    data = pd.read_csv(f, engine="python")
    data = data.drop(columns="Unnamed: 0")
    data.plot()
    plt.show()
