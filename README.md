# Seismic Data Visualization Application

This project provides a PyQt5-based GUI and Dash web application for visualizing and analyzing seismic data, wells, and horizons using `SeismicPlotter`. The application integrates Python libraries such as Plotly, Matplotlib, and Dash for interactive plotting and visualization.

## Features

- **Seismic Visualization**: Plot inline, crossline, and time-slice seismic data using Plotly.
- **Interactive GUI**: A PyQt5 interface with dropdowns, checklists, and sub-windows for seismic, well, and horizon visualization.
- **Dash Integration**: A Dash web application for advanced visualization and interaction.
- **Flexible Layout**: Supports tiled sub-windows and standalone Dash apps for detailed data analysis.

## Requirements

- Python 3.10+
- Libraries:
  - PyQt5
  - Plotly
  - Dash
  - Matplotlib
  - NumPy
  - Pandas
  - SciPy
  - xarray
  - `odbind` package for seismic data handling

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/seismic-visualization.git
   cd seismic-visualization
