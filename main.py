import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QListWidget, QLabel, QPushButton, QCheckBox, QGridLayout, QMdiArea
import matplotlib.pyplot as plt
import xarray as xr
from odbind.survey import Survey
from odbind.seismic3d import Seismic3D
from odbind.well import Well
from odbind.horizon3d import Horizon3D
import pandas as pd
from PyQt5.QtWidgets import QApplication, QListWidget, QVBoxLayout, QLabel, QPushButton, QWidget, QAbstractItemView, QCheckBox, QGridLayout
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QUrl, QObject, pyqtSlot
from plotly.subplots import make_subplots
from arb18_fin_plt_log import SeismicHorizonPlotter 
from scipy.interpolate import interp1d


import matplotlib.cm as cm
from matplotlib.colors import Normalize


from corl_p3_final1_seis_fin_da1_fin9 import MainWindow1
#from arb18_fin_plt_d import MainWindow2




# Run the application
app = QApplication(sys.argv)

window = MainWindow1()
window.show()
sys.exit(app.exec_())
