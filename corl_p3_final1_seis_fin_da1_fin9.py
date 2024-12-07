import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QListWidget, QLabel, QPushButton, QCheckBox, QGridLayout, QMdiArea, QMdiSubWindow
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QListWidget, QVBoxLayout, QLabel, QPushButton, QWidget, QAbstractItemView, QCheckBox, QGridLayout, QComboBox

import matplotlib.cm as cm
from matplotlib.colors import Normalize
from dash import Dash, dcc, html, State
from dash import no_update

import xarray as xr
import threading
from dash.dependencies import Input, Output
from arb18_fin_plt_log import SeismicHorizonPlotter 
from multishp6_fin_sub import create_dash_app2
import dash_table


import sys
import os
import matplotlib.pyplot as plt
import threading
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMdiArea
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go





class SeismicPlotter:
    def __init__(self, survey_name, seismic_name):
        self.survey = Survey(survey_name)
        self.seismic = Seismic3D(self.survey, seismic_name)
        

        # Get the range of inlines, crosslines, and Z slices
        ranges = self.seismic.ranges
        self.inline_range = list(range(ranges.inlrg[0], ranges.inlrg[1] + 1, 20))
        self.crossline_range = list(range(ranges.crlrg[0], ranges.crlrg[1] + 1, 20))
        self.time_slices = list(range(0, 2000, 100))  # Example: 0 to 2000 ms


    def fetch_inline_data(self, inline_number):
        """Fetch seismic data for a specific inline."""
        try:
            iline_slice = self.seismic.iline[inline_number]
            
            # Extract x-coordinates (crossline), y-coordinates (inline), and z-coordinates (TWT or depth)
            x_coords = iline_slice['x'].values  # Crossline coordinates
            y_coords = iline_slice['y'].values  # Inline coordinates
            z_coords = iline_slice.coords['twt'].values  # Time or Depth

            if self.seismic.comp_names:
                firstcomp = self.seismic.comp_names[0]
                seismic_inline = iline_slice[firstcomp].values  # Extract the seismic data
                return x_coords, y_coords, z_coords, seismic_inline
        except Exception as e:
            print(f"Error fetching inline data for {inline_number}: {e}")
            return None, None, None, None

    def fetch_crossline_data(self, crossline_number):
        """Fetch seismic data for a specific crossline."""
        try:
            xline_slice = self.seismic.xline[crossline_number]
            
            # Extract x-coordinates (inline), y-coordinates (crossline), and z-coordinates (TWT or depth)
            x_coords = xline_slice['x'].values  # Inline coordinates
            y_coords = xline_slice['y'].values  # Crossline coordinates
            z_coords = xline_slice.coords['twt'].values  # Time or Depth

            if self.seismic.comp_names:
                firstcomp = self.seismic.comp_names[0]
                seismic_crossline = xline_slice[firstcomp].values  # Extract the seismic data
                return x_coords, y_coords, z_coords, seismic_crossline
        except Exception as e:
            print(f"Error fetching crossline data for {crossline_number}: {e}")
            return None, None, None, None

    def plot_seismic_data(self, inline_number, crossline_number,xy_map):
        """Plot the seismic data using Plotly."""
        x_coords_inline, y_coords_inline, z_coords_inline, seismic_inline = self.fetch_inline_data(inline_number)
        x_coords_crossline, y_coords_crossline, z_coords_crossline, seismic_crossline = self.fetch_crossline_data(crossline_number)

        if seismic_inline is not None and seismic_crossline is not None:
            # Define the color scale limits based on the maximum and minimum values
            zmin = min(np.min(seismic_inline), np.min(seismic_crossline))
            zmax = max(np.max(seismic_inline), np.max(seismic_crossline))

            # Create heatmap for inline
            fig_inline = go.Figure(data=go.Heatmap(
                z=np.transpose(seismic_inline),
                x=x_coords_inline,
                y=z_coords_inline,
                colorscale='RdGy',
                zmin=zmin,
                zmax=zmax
            ))

            # Create heatmap for crossline
            fig_crossline = go.Figure(data=go.Heatmap(
                z=np.transpose(seismic_crossline),
                x=x_coords_crossline,
                y=z_coords_crossline,
                colorscale='RdGy',
                zmin=zmin,
                zmax=zmax
            ))

            # Create XY map for inline and crossline data
            xy_map.add_trace(go.Scatter(
                x=x_coords_inline,
                y=y_coords_inline,
                mode='markers',
                marker=dict(
                    size=5,
                    color='blue',
                    opacity=0.6
                ),
                name=f'Inline {inline_number}'
            ))

            xy_map.add_trace(go.Scatter(
                x=x_coords_crossline,
                y=y_coords_crossline,
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.6
                ),
                name=f'Crossline {crossline_number}'
            ))

            # Update layout for inline
            fig_inline.update_yaxes(autorange='reversed')
            fig_inline.update_layout(
                title=f"Seismic Data for Inline {inline_number}",
                xaxis_title='Crossline Coordinate',
                yaxis_title='Time/Depth',
                width=600,
                height=600,
                margin=dict(l=40, r=40, t=40, b=40)
            )

            # Update layout for crossline
            fig_crossline.update_yaxes(autorange='reversed')
            fig_crossline.update_layout(
                title=f"Seismic Data for Crossline {crossline_number}",
                xaxis_title='Inline Coordinate',
                yaxis_title='Time/Depth',
                width=600,
                height=600,
                margin=dict(l=40, r=40, t=40, b=40)
            )

            # Update layout for XY map
            xy_map.update_layout(
                title="Inline and Crossline XY Map",
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                width=1200,
                height=1200,
                margin=dict(l=40, r=40, t=40, b=40)
            )

            return fig_inline, fig_crossline, xy_map
        else:
            print("No data to plot.")
            return None, None, None


    def plot_time_slice(self, time_slice):
        """Get the data for the time slice."""
        # Get the inline and crossline ranges
        inl_range = self.seismic_volume.ranges.inlrg
        crl_range = self.seismic_volume.ranges.crlrg

        # Convert the time slice to a z index (if necessary)
        z_range = [time_slice, time_slice, 1]  # Start, stop, step

        # Fetch the data using getdata
        try:
            data = self.seismic_volume.getdata(inl_range, crl_range, z_range)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, None

        # Check the type of the returned result
        if isinstance(data, xr.Dataset):
            data_array = data.to_array().squeeze()  # Convert to DataArray and remove singleton dimensions
            info = {
                'x': data.coords['x'].values,
                'y': data.coords['y'].values
            }
            return info['x'], info['y'], data_array
        else:
            raise ValueError("Unexpected return format from getdata.")









#plotter = SeismicPlotter(selected_survey, selected_seismic)


# Load surveys and return as options
def load_surveys():
    try:
        surveys = Survey.names()
        return [{'label': survey, 'value': survey} for survey in surveys]
    except Exception as e:
        print(f"Error loading surveys: {e}")
        return []

selected_survey1 = 'F3_Demo_2020'

selected_seismic1= '4 Dip steered median filter'
# Global variable to store selections
selected_values = {
    'survey': None,
    'well': None,
    'seismic': None,
    'horizon': None,
    'log': None,
    'plot': None
}

# Function to create and run Dash apps
def create_dash_app1():
#def create_dash_app(seismic_canvas):
    app = Dash(__name__)
    #plotter = SeismicPlotter(selected_survey1, selected_seismic1)

    # Load surveys once
    survey_options = load_surveys()

    # CSS styles for the dropdowns to be displayed vertically on the left
    dropdown_style = {
        'width': '300px',
        'marginBottom': '20px'  # Add spacing between dropdowns
    }

    
            
    

    # Main layout for the app with dropdowns in a vertical column
    app.layout = html.Div([
      html.Div([
        html.Label('Select Survey:', style={'marginBottom': '1px'}),
        dcc.Dropdown(id='survey-dropdown', options=survey_options, value=None, style=dropdown_style),

        html.Label('Select Well:', style={'marginBottom': '1px'}),
        dcc.Dropdown(id='well-dropdown', options=[], value=None, style=dropdown_style),

        html.Label('Select Seismic Volume:', style={'marginBottom': '1px'}),
        dcc.Dropdown(id='seismic-dropdown', options=[], value=None, style=dropdown_style),

        html.Label('Select Horizon:', style={'marginBottom': '1px'}),
        dcc.Dropdown(id='horizon-dropdown', options=[], value=None, style=dropdown_style),

        html.Label('Select Log:', style={'marginBottom': '1px'}),
        dcc.Dropdown(id='log-dropdown', options=[], value=None, style=dropdown_style),

        html.Label('Select Plot (1 to 10):', style={'marginBottom': '1px'}),
        dcc.Dropdown(
            id='plot-dropdown',
            options=[{'label': f'Plot {i}', 'value': i} for i in range(1, 11)],  # Dropdown options for Plot 1 to 10
            value=None,
            style=dropdown_style
        ),



        html.Label('Select Wells:', style={'marginBottom': '10px'}),
        dcc.Checklist(
            id='well-checklist',
            options=[],  # Initialize with an empty list
            value=[] , # Initially, no wells are selected
        ),
        html.Div(id='selected-wells')
      ], style={'display': 'flex', 'flexDirection': 'column', 'width': '300px', 'padding': '2px', 'marginRight': '20px'}),

      html.Button('Save Selections', id='save-button'),
      html.Div(id='dummy-output'),  # Dummy output to satisfy callback
    

      html.Div([
        dcc.Dropdown(  
          id='inline-dropdown',
          
          options=[],  # Initialize with an empty list
          value=[] , # Initially, no wells are selected

          #options=[{'label': f'Inline {inline}', 'value': inline} for inline in plotter.inline_range],
          
          #value=plotter.inline_range[0],  # Default value
          clearable=False,
          style={'width': '48%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
          id='crossline-dropdown',
          
          options=[],  # Initialize with an empty list
          value=[],  # Initially, no wells are selected

          #options=[{'label': f'Crossline {crossline}', 'value': crossline} for crossline in plotter.crossline_range],
          
          #value=plotter.crossline_range[0],  # Default value
          clearable=False,
          style={'width': '48%', 'display': 'inline-block'}
        ),
      html.Div([
        dcc.Graph(id='seismic-inline-heatmap', style={'display': 'inline-block'}),
        dcc.Graph(id='seismic-crossline-heatmap', style={'display': 'inline-block'}),
        
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    #dcc.Graph(id='xy-map', style={'margin-top': '20px','height': '1200px','width': '100%' })  # XY map for inline and crossline
]),



      html.Div([
            dcc.Graph(id='xy-map'),
            dcc.Store(id='line_data', data=None)   # Store to hold line data  # Placeholder for the well map
        ], style={'width': '70%', 'padding': '2px', 'marginLeft': '20px'}),


    ], style={'display': 'flex', 'justifyContent': 'flex-start', 'alignItems': 'flex-start'})
    
        




    #html.Div(id='dummy-output', style={'display': 'none'})  # Dummy output for callbacks
    #], style={'display': 'flex', 'justifyContent': 'flex-start', 'alignItems': 'flex-start'})



      
    @app.callback(
        Output('well-dropdown', 'options'),
        [Input('survey-dropdown', 'value')]
    )
    def update_wells(selected_survey):
        if selected_survey:
            wells = Well.names(Survey(selected_survey))
            return [{'label': well, 'value': well} for well in wells]
        return []

    @app.callback(
        Output('seismic-dropdown', 'options'),
        [Input('survey-dropdown', 'value')]
    )
    def update_seismic(selected_survey):
        if selected_survey:
            seismic_volumes = Seismic3D.names(Survey(selected_survey))
            return [{'label': volume, 'value': volume} for volume in seismic_volumes]
        return []

    @app.callback(
        Output('horizon-dropdown', 'options'),
        [Input('survey-dropdown', 'value')]
    )
    def update_horizon(selected_survey):
        if selected_survey:
            horizons = Horizon3D.names(Survey(selected_survey))
            return [{'label': hrz, 'value': hrz} for hrz in horizons]
        return []

    @app.callback(
      Output('log-dropdown', 'options'),
      [Input('survey-dropdown', 'value'),  # Add survey-dropdown as an input
        Input('well-dropdown', 'value')]    # well-dropdown already present
    )
    def update_logs(selected_survey, selected_well):
      if selected_survey and selected_well:
        specific_well = Well(Survey(selected_survey), selected_well)
        logs = specific_well.log_names
        return [{'label': log, 'value': log} for log in logs]
      return []

    @app.callback(
        Output('log-dropdown', 'value'),
        [Input('log-dropdown', 'options')]
    )
    def reset_log_dropdown(log_options):
        return log_options[0]['value'] if log_options else None

    @app.callback(
        Output('dummy-output', 'options'),
        [Input('plot-dropdown', 'value')]
    )
    def update_plot_selection(selected_plot):
      if selected_plot:
        print(f"Selected Plot: {selected_plot}")
        #seismic_canvas.inline_number = 400 + (selected_plot * 25)  # Example of updating inline
        #seismic_canvas.plot_image()  # Re-plot with the new parameters
        #print(f"Updated Plot: {selected_plot}")
      return []

    


    @app.callback(
      Output('well-checklist', 'options'),
      [Input('survey-dropdown', 'value')]
    )
    def update_well_checklist(selected_survey):
      global selected_survey1  # Declare the variable as global
      selected_survey1 = selected_survey  # Update the global variable

      if selected_survey:
        wells = Well.names(Survey(selected_survey))
        return [{'label': well, 'value': well} for well in wells]
      return []

    
   


    @app.callback(
      [Output('inline-dropdown', 'options'),
       Output('crossline-dropdown', 'options')],
      [Input('inline-dropdown', 'value'),
       Input('crossline-dropdown', 'value'),
       Input('survey-dropdown', 'value'),
       Input('seismic-dropdown', 'value')]


    )
    def update_heatmap(selected_inline, selected_crossline,selected_survey, selected_seismic):
      """Update the heatmap and XY map based on the selected inline and crossline."""
      print(selected_survey, selected_seismic)
      if selected_seismic:
            plotter1 = SeismicPlotter(selected_survey, selected_seismic)  # Update plotter1
            inline_options = [{'label': f'Inline {inline}', 'value': inline} for inline in plotter1.inline_range]
            crossline_options = [{'label': f'Crossline {crossline}', 'value': crossline} for crossline in plotter1.crossline_range]
            return inline_options, crossline_options
            

      return [], []  # Return empty lists if no seismic is selected
      

    @app.callback(
        [Output('seismic-inline-heatmap', 'figure'),
         Output('seismic-crossline-heatmap', 'figure'),
         Output('xy-map', 'figure'),
         Output('line_data', 'data')], 
        [Input('inline-dropdown', 'value'),
         Input('crossline-dropdown', 'value'),       
         Input('survey-dropdown', 'value'),
         Input('seismic-dropdown', 'value'),
         Input('horizon-dropdown', 'value'),
         Input('xy-map', 'clickData'),],
        State('line_data', 'data',),

    )
    def update_heatmap1(selected_inline, selected_crossline, selected_survey, selected_seismic, selected_horizon, clickData,line_data):
     

      if selected_seismic:
        plotter1 = SeismicPlotter(selected_survey, selected_seismic)  # Update plotter1
        print(selected_survey, selected_seismic,selected_inline, selected_crossline)
        xy_map=update_well_map(selected_survey, selected_horizon, clickData,line_data)
        
        line_data=arb_map(clickData,xy_map)
        print('newline:',line_data)
        fig_inline, fig_crossline, xy_map = plotter1.plot_seismic_data(selected_inline, selected_crossline,xy_map)
        
        # If figures are valid, return them
        if fig_inline and fig_crossline and xy_map:
            return fig_inline, fig_crossline, xy_map,line_data


      # Return empty figures if no seismic is selected
      return go.Figure(), go.Figure(), go.Figure(),line_data


    



    # Initialize a global list to store clicked points
    clicked_points = []

    #@app.callback(
      #Output('xy-map', 'figure', allow_duplicate=True),
      
      #[Input('survey-dropdown', 'value'),
       #Input('horizon-dropdown', 'value'),      
       #Input('xy-map', 'clickData'),  # Add clickData to capture click events
       #Input('line_data', 'data',)],
      #prevent_initial_call=True
    #)
    def update_well_map(selected_survey, selected_horizon, clickData,line_data):
      
      global clicked_points  # Reference the global clicked_points list
      
      if selected_survey:
        wells = Well.names(Survey(selected_survey))

        well_coordinates = {}
        selected_wells = []  # Define this as an empty list or populate from a valid input
        clicked_points = []  # Initialize clicked_points here
        xy_map = go.Figure()  # Initialize the figure


        for well_no in wells:
            try:
                specific_well = Well(Survey(selected_survey), well_no)
                trk = specific_well.track()  # Try to get the track data

                x_coords = trk['x']
                y_coords = trk['y']

                last_x = x_coords[-1]
                last_y = y_coords[-1]

                # Add the well data to the figure
                xy_map.add_trace(go.Scatter(
                    x=[last_x], y=[last_y],
                    mode='markers+text',
                    text=[well_no],
                    marker=dict(color='red', size=10),
                    textposition='top center',
                    customdata=[well_no],
                    hoverinfo='text',
                    name=well_no
                ))

            except ValueError as e:
                print(f"Error retrieving track for well {well_no}: {e}")

        # Plot the horizon if selected
        if selected_horizon:
            horizon_names = Horizon3D.names(Survey(selected_survey))
            horizons = []

            for horizon_name in horizon_names:
                try:
                    horizon = Horizon3D(Survey(selected_survey), horizon_name)
                    horizons.append((horizon_name, horizon))
                except Exception as e:
                    print(f"Error loading horizon {horizon_name}: {e}")

            for name, horizon in horizons:
                if name == selected_horizon:
                    try:
                        result = horizon.getdata()

                        if isinstance(result, tuple):
                            data, info = result
                            x_data = np.array(info['x'])
                            y_data = np.array(info['y'])
                            z = data[0]  # z-values

                        elif isinstance(result, xr.Dataset):
                            x_data = result['x'].values
                            y_data = result['y'].values
                            z = result['z'].values

                        # Check for dimension match
                        if x_data.shape[0] != y_data.shape[0] or x_data.shape[1] != z.shape[1]:
                            print(f"Skipping plot due to dimension mismatch for horizon {selected_horizon}")
                            return xy_map  # Return empty figure

                        # Add the contour for the horizon
                        xy_map.add_trace(go.Contour(
                            z=z,
                            x=x_data[0, :],  # X coordinates
                            y=y_data[:, 0],  # Y coordinates
                            colorscale='Viridis',
                            showscale=True,
                            name=selected_horizon,
                            hoverinfo='z+name'
                        ))

                    except Exception as e:
                        print(f"Error retrieving data for horizon {selected_horizon}: {e}")

        # Check if clickData has been received
        # Initialize line data if none exists
        
        #line_data =  None
        



        xy_map.update_layout(
            title='Well and Horizon Map',
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(domain=[0, 1]),
            yaxis=dict(scaleanchor="x", scaleratio=1, domain=[0, 1]),
            showlegend=True,
            clickmode='event+select',
            height=1500  # Set the height of the plot (adjust as needed)
        )

        return xy_map # Return the figure (even if empty)
 


 # Initialize a global list to store clicked points
    clicked_points = []

    @app.callback(
      #Output('xy-map', 'figure', allow_duplicate=True),
      Output('line_data', 'data', allow_duplicate=True),
      Input('xy-map', 'clickData'),
      #Input('xy-map', 'figure',),
      prevent_initial_call=True
    )
    def arb_map(clickData,xy_map):
      
      #clicked_points = []
      line_data= None
      print('cl:',clickData)
      #print('ld:',line_data)
      

      if clickData:
          # Extract the x and y coordinates from the clickData
          x = clickData['points'][0]['x']
          y = clickData['points'][0]['y']
          print(x,y)
          #if not clicked_points or (x, y) != clicked_points[-1]:
          clicked_points.append((x, y))
          print (clicked_points)
          # Draw a line if two points have been clicked
          if len(clicked_points) == 2:
            # Extract the coordinates of the two clicked points
            x_values = [clicked_points[0][0], clicked_points[1][0]]
            y_values = [clicked_points[0][1], clicked_points[1][1]]

            # Store the line coordinates in the last_line variable
            line_data = (x_values, y_values)
            print(line_data)
            # Clear the clicked points list for future clicks
            #clicked_points.clear()

        # Plot the last line if it exists
      if line_data:
          x_values, y_values = line_data
          xy_map.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            line=dict(color='blue', width=5, dash='solid'),
            marker=dict(size=10, color='blue'),
            name='Line between clicks'
      ))
        

      xy_map.update_layout(
            title='Well and Horizon Map',
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(domain=[0, 1]),
            yaxis=dict(scaleanchor="x", scaleratio=1, domain=[0, 1]),
            showlegend=True,
            clickmode='event+select',
            height=1500  # Set the height of the plot (adjust as needed)
      )

      return line_data # Return the figure (even if empty)




    # Return an empty figure if no survey is selected
    #return go.Figure(data=[], layout={'xaxis': {'title': 'X'}, 'yaxis': {'title': 'Y'}})



    # Callback to update selected values and save them to a file on button press
    @app.callback(
      Output('dummy-output', 'children'),
      [Input('save-button', 'n_clicks')],
      [Input('survey-dropdown', 'value'),
       Input('well-dropdown', 'value'),
       Input('seismic-dropdown', 'value'),
       Input('horizon-dropdown', 'value'),
       Input('log-dropdown', 'value'),
       Input('plot-dropdown', 'value'),
       Input('inline-dropdown', 'value'),
       Input('crossline-dropdown', 'value'),
       Input('line_data', 'data'),],
      prevent_initial_call=True
    )
    def save_selected_values(n_clicks, selected_survey, selected_well, selected_seismic, selected_horizon, selected_log, selected_plot,selected_inline,selected_crossline,line_data ):
      if n_clicks is None:
        return ""
      print("Line data before saving:", line_data)

      # Store selected values in a dictionary
      selected_values = {
        'survey': selected_survey,
        'well': selected_well,
        'seismic': selected_seismic,
        'horizon': selected_horizon,
        'log': selected_log,
        'plot': selected_plot,
        'inline': selected_inline, 
        'crossline': selected_crossline,
        'arbline': line_data if line_data is not None else [],  # Use an empty list if None

      }

      # Save the dictionary to a JSON file
      with open('selected_values.json', 'w') as f:
        json.dump(selected_values, f, indent=4)

      return f"Selections saved to selected_values.json"


    @app.callback(
        Output('dummy-output', 'children', allow_duplicate=True),  # Dummy output to satisfy Dash callback requirements
        [Input('survey-dropdown', 'value'),
         Input('well-dropdown', 'value'),
         Input('seismic-dropdown', 'value'),
         Input('horizon-dropdown', 'value'),
         Input('log-dropdown', 'value'),
         Input('plot-dropdown', 'value')],
        prevent_initial_call=True
    )
    def print_selected_values(selected_survey, selected_well, selected_seismic, selected_horizon,selected_log, selected_plot):
        print(f"Selected Survey: {selected_survey}")
        print(f"Selected Well: {selected_well}")
        print(f"Selected Seismic Volume: {selected_seismic}")
        print(f"Selected Horizon: {selected_horizon}")
        print(f"Selected Log: {selected_log}")
        print(f"Selected Plot: {selected_plot}")
        selected_survey1=selected_survey

        global selected_values
        selected_values['survey'] = selected_survey
        selected_values['well'] = selected_well
        selected_values['seismic'] = selected_seismic
        selected_values['horizon'] = selected_horizon
        selected_values['log'] = selected_log
        selected_values['plot'] = selected_plot

        #refresh_plot()


        return ""

    

    return app, selected_values


def run_dash_server(app, port):
    app.run_server(debug=True, port=port, use_reloader=False)




def refresh_plot():
    #global selected_values  # Access the global variable
    # Print the current selections
    print(f"Refreshing plot with selections: {selected_values}")


# Call to refresh the plot after the app is created
refresh_plot()




# Define the colormap and normalization
cmap = cm.viridis  # Choose the colormap (you can change this to any other colormap)
norm = Normalize(vmin=0, vmax=150)  # Set the color range from 0 to 150


with open('selected_values.json', 'r') as f:
   selected_values = json.load(f)
selected_survey=selected_values.get('survey')
selected_seismic=selected_values.get('seismic')
selected_horizon=selected_values.get('horizon')
selected_inline=selected_values.get('inline')
selected_crossline=selected_values.get('crossline')
selected_well=selected_values.get('well')
#coordinates=selected_values.get('arbline')
#start_coords1 = (coordinates[0][0], coordinates[1][0])  # (x, y)
#end_coords1 = (coordinates[0][1], coordinates[1][1])  # (x, y)
#print(start_coords1,end_coords1)



# Define your Survey and Well classes (assuming these are correct)
survey = Survey(selected_survey)
wells = Well.names(survey)
print(wells)
well_name = selected_well
base_path = r'C:\open_dtect'
survey_name = selected_survey
well_folder = 'WellInfo'

horizon_names = Horizon3D.names(survey)
inline_number = selected_inline
time_slice = 1250  # Example time slice
file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name}.wlt"
depth_to_time_file_path = file_path



# Define the functions for depth-to-time conversion
def load_depth_to_time_model(file_path):
    depths = []
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('!') and not line.startswith('Name') and not line.startswith('Description'):
                try:
                    depth, time = map(float, line.split())
                    depths.append(depth)
                    times.append(time)
                except ValueError:
                    # Handle lines that can't be converted to float (possibly headers or comments)
                    pass
    return np.array(depths), np.array(times)

def create_interpolation_function(depths, times):
    return interp1d(depths, times, kind='linear', fill_value='extrapolate')

def depth_to_time(depth, interp_func):
    return interp_func(depth)




def plot_track_data(well: Well):
    """Plot well track data."""
    try:
        # Retrieve well track data as a DataFrame
        track_data = well.track_dataframe()
        print("Track Data:", track_data)  # Debugging print
        x_coord=track_data['x']
        y_coord=track_data['y']
        print(x_coord[1],y_coord[1])
        if track_data.empty:
            print("No track data available.")
            return

        #plt.figure(figsize=(12, 8))
        #for column in track_data.columns:
            #plt.plot(track_data[column],track_data.index,  label=column)
        
        #plt.xlabel('Depth')
        #plt.ylabel('Value')
        #plt.title('Well Track Data')
        #plt.legend()
        #plt.grid(True)
        #plt.show()
        return x_coord[1],y_coord[1]
    except Exception as e:
        print(f"Error retrieving track data: {e}")



def generate_well_map_figure(selected_wells=None):
    

    if selected_wells is None:
        selected_wells = []

    fig = go.Figure()

    well_coordinates = {}
    for well_no in wells:
        specific_well = Well(survey, well_no)
        trk = specific_well.track()

        x_coords = trk['x']
        y_coords = trk['y']

        last_x = x_coords[-1]
        last_y = y_coords[-1]

        color = 'blue' if well_no in selected_wells else 'red'

        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y],
            mode='markers+text',
            text=[well_no],
            marker=dict(color=color, size=10),
            textposition='top center',
            customdata=[well_no],  # Add customdata to store well numbers
            hoverinfo='text',
            name=well_no
        ))

        well_coordinates[well_no] = (last_x, last_y)

    fig.update_layout(
        #title='Well Map - All Wells',
        xaxis_title='X',xaxis=dict(domain=[0, 1],),
        yaxis_title='Y',
        showlegend=False,yaxis=dict(scaleanchor="x", scaleratio=1,domain=[0, 1],),
        clickmode='event+select'  # Enable click events
    )

    return fig

def generate_connection_map(selected_wells, well_coordinates,fig):
    #fig = go.Figure()

    well_coordinates = {}
    for well_no in wells:
        specific_well = Well(survey, well_no)
        trk = specific_well.track()

        x_coords = trk['x']
        y_coords = trk['y']

        last_x = x_coords[-1]
        last_y = y_coords[-1]

        color = 'blue' if well_no in selected_wells else 'red'

        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y],
            mode='markers+text',
            text=[well_no],
            marker=dict(color=color, size=5),
            textposition='top center',textfont=dict(size=10), 
            customdata=[well_no],  # Add customdata to store well numbers
            hoverinfo='text',
            name=well_no,xaxis='x3',yaxis='y3',
        ),row=1,col=20,secondary_y=True,)

        well_coordinates[well_no] = (last_x, last_y)



    # Plot selected wells
    for well_no in selected_wells:
        if well_no in well_coordinates:
            x, y = well_coordinates[well_no]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[well_no],
                marker=dict(color='blue', size=5),
                textposition='top center',textfont=dict(size=10),
                name=well_no,xaxis='x3',yaxis='y3',

            ),row=1,col=20,secondary_y=True,)

    # Connect selected wells with blue lines
    for i in range(len(selected_wells) - 1):
        well_no1 = selected_wells[i]
        well_no2 = selected_wells[i + 1]
        if well_no1 in well_coordinates and well_no2 in well_coordinates:
            x1, y1 = well_coordinates[well_no1]
            x2, y2 = well_coordinates[well_no2]
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f'{well_no1} to {well_no2}',xaxis='x3',yaxis='y3',

            ),row=1,col=20,secondary_y=True,)
    y_values = [coords[1] for coords in well_coordinates.values()]
    x_values = [coords[0] for coords in well_coordinates.values()]

    y_range=max(y_values)-min(y_values)
    fig.update_yaxes(  row=1, col=20,domain=[0.9, 1])
    fig.update_xaxes(  row=1, col=20,domain=[0.9, 1])

    fig.update_layout(
        #title='Well Map with Connections',
        #xaxis3=dict(domain=[0, 1],),
        #yaxis3=dict(domain=[0, 1],),

        showlegend=False,
        #yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig

class WebChannel(QObject):
    @pyqtSlot(str)
    def select_well(self, well_no):
        print(f"Selected Well: {well_no}")
        self.parent().update_selected_well(well_no)

class MainWindow1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Geology Dashboard - Well Map and Data Visualization')
        self.setGeometry(100, 100, 1200, 800)  # Set window dimensions

        # Initialize MDI area
        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

        self.seismic_html_file = os.path.abspath("seismic_plot.html")

        self.html_file = os.path.abspath("corl_plot.html")
        self.corl_html_file = os.path.abspath("corl_plot.html")
        self.connection_html_file = os.path.abspath("connection_plot.html")

        self.selected_wells = []

        # Create the sub-windows (Well map, correlation plot, etc.)
        self.create_corl_plot_subwindow()  # Correlation plot sub-window
        self.create_well_map_subwindow()   # Well map and controls
        self.create_seismic_subwindow()   # Well map and controls



        # Start the Dash servers
        self.start_dash_servers()

        # Create Matplotlib and Plotly windows
        #self.create_visualizations()
        self.create_pyplot_subwindow()

        self.create_webview_subwindow()  # Right sub-window

        # Initial run of the logic
        self.rerun_logic()
        




        #self.create_dash_subwindow()
    def start_dash_servers(self):
            dash_app1, selected_values = create_dash_app1()
            dash_app2 = create_dash_app2()

            # Set the selected values as a class attribute for later use
            self.selected_values = selected_values or {}  # Ensure it's an empty dict if not yet populated
            print("Initial selected values:", self.selected_values)

            # Run Dash applications in separate threads
            threading.Thread(target=run_dash_server, args=(dash_app1, 8050), daemon=True).start()
            threading.Thread(target=run_dash_server, args=(dash_app2, 8051), daemon=True).start()

            # Create QWebEngineView for each Dash app
            self.create_dash_subwindow()    

            # Call rerun_logic only if selected_values is not empty
            if self.selected_values:
                self.rerun_logic()
            else:
                 print("No values selected yet, waiting for user input...")


    def create_seismic_subwindow(self):
        # Create a sub-window for the correlation plot
        seismic_subwindow = QMdiSubWindow()
        seismic_subwindow.setWindowTitle("SEISMIC ARBITARY LINE")

        # Create a QWebEngineView and load the correlation Plotly HTML file
        seismic_web_view = QWebEngineView()
        seismic_web_view.setUrl(QUrl.fromLocalFile(self.seismic_html_file))

        seismic_subwindow.setWidget(seismic_web_view)
        self.mdi_area.addSubWindow(seismic_subwindow)
        seismic_subwindow.show()



    def create_corl_plot_subwindow(self):
        # Create a sub-window for the correlation plot
        corl_subwindow = QMdiSubWindow()
        corl_subwindow.setWindowTitle("WELL MAP")

        # Create a QWebEngineView and load the correlation Plotly HTML file
        corl_web_view = QWebEngineView()
        corl_web_view.setUrl(QUrl.fromLocalFile(self.corl_html_file))

        corl_subwindow.setWidget(corl_web_view)
        self.mdi_area.addSubWindow(corl_subwindow)
        corl_subwindow.show()

    def create_well_map_subwindow(self):
        # Create a sub-window for the well map and controls
        well_map_subwindow = QMdiSubWindow()
        well_map_subwindow.setWindowTitle("LOG CORELATION")
        
        # Create central widget and layout for this sub-window
        well_map_widget = QWidget()
        well_map_layout = QVBoxLayout()
        well_map_widget.setLayout(well_map_layout)

        # Create a QTextEdit to display logs (log area at the bottom)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        well_map_layout.addWidget(self.log_text_edit)

        # Initialize the GUI for well map selection and plotting
        self.init_gui(well_map_layout)
        #self.init_gui()
        well_map_subwindow.setWidget(well_map_widget)
        self.mdi_area.addSubWindow(well_map_subwindow)
        well_map_subwindow.show()



    def create_dash_subwindow(self):
      # Create the first sub-window for the Dash app
      dash_subwindow = QMdiSubWindow()
      dash_subwindow.setWindowTitle("Dash App 1")

      # Create a QWebEngineView to display the Dash app
      dash_web_view = QWebEngineView()
      dash_web_view.setUrl(QUrl("http://127.0.0.1:8050"))  # URL of the first running Dash app

      dash_subwindow.setWidget(dash_web_view)
      self.mdi_area.addSubWindow(dash_subwindow)

      # Create the second sub-window for another Dash app
      dash_subwindow1 = QMdiSubWindow()  # Create the second sub-window
      dash_subwindow1.setWindowTitle("Dash App 2")

      dash_web_view1 = QWebEngineView()
      dash_web_view1.setUrl(QUrl("http://127.0.0.1:8051"))  # URL of the second running Dash app

      dash_subwindow1.setWidget(dash_web_view1)
      self.mdi_area.addSubWindow(dash_subwindow1)

      # Show both sub-windows
      dash_subwindow.show()
      dash_subwindow1.show()





    def create_pyplot_subwindow(self):
        # Create a sub-window for a PyPlot (matplotlib) plot
        pyplot_subwindow = QMdiSubWindow()
        pyplot_subwindow.setWindowTitle("PyPlot")
        #refresh_plot()
        survey = self.selected_values.get('survey')
        # Create a matplotlib figure and canvas
        fig, ax = plt.subplots()
        ax.text(0, 0, survey, fontsize=12, color='red')  # Add text in the plot

        canvas = FigureCanvas(fig)

        pyplot_subwindow.setWidget(canvas)
        self.mdi_area.addSubWindow(pyplot_subwindow)
        pyplot_subwindow.show()

   
    def create_webview_subwindow(self):
        """Creates the right sub-window with the dropdown and webview."""
        right_subwindow = QMdiSubWindow()
        right_subwindow.setWindowTitle("Web Content Panel")

        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Define paths to HTML files
        self.corl_html_file = os.path.abspath("triple_combo_plot.html")
        self.corl_html_file1 = os.path.abspath("Chart_Book_Solutions.html")
        #self.corl_html_file2 = os.path.abspath("Shale Parameter.html")
        self.corl_html_file2 = os.path.abspath("http://localhost:63677/")

        self.corl_html_file3 = os.path.abspath("Lithology Plot.html")
        self.corl_html_file4 = os.path.abspath("phit_buck_plot.html")
        self.corl_html_file5 = os.path.abspath("Wax_Smits_Plot.html")
        self.corl_html_file6 = os.path.abspath("Produced_Oil.html")
        self.corl_html_file7 = os.path.abspath("Interpretation Depth Plot.html")
        self.corl_html_file8 = os.path.abspath("fig3.html")



        # Create a QComboBox (dropdown) to select which HTML file to display
        self.dropdown = QComboBox()
        self.dropdown.addItem("triple_combo_plot", self.corl_html_file)
        self.dropdown.addItem("Chart_Book_Solutions", self.corl_html_file1)
        self.dropdown.addItem("Shale Parameter", self.corl_html_file2)
        self.dropdown.addItem("Lithology Plot", self.corl_html_file3)
        self.dropdown.addItem("phit_buck_plot", self.corl_html_file4)
        self.dropdown.addItem("Wax_Smits_Plot", self.corl_html_file5)
        self.dropdown.addItem("Produced_Oil", self.corl_html_file6)
        self.dropdown.addItem("Interpretation Depth Plot", self.corl_html_file7)
        self.dropdown.addItem("fig3", self.corl_html_file8)

        # Create the QWebEngineView to display the HTML files
        self.webview = QWebEngineView()
        self.webview.setUrl(QUrl.fromLocalFile(self.corl_html_file))  # Default to first HTML file

        # Connect the dropdown to a method that changes the web view content
        self.dropdown.currentIndexChanged.connect(self.change_html_file)

        # Add the dropdown and web view to the right layout
        right_layout.addWidget(self.dropdown)
        right_layout.addWidget(self.webview)
        right_panel.setLayout(right_layout)

        right_subwindow.setWidget(right_panel)

        # Add the sub-window to the MDI area
        self.mdi_area.addSubWindow(right_subwindow)
        right_subwindow.show()

    def change_html_file(self):
        """Update the QWebEngineView to display the selected HTML file."""
        selected_html = self.dropdown.currentData()
        if selected_html:
            self.webview.setUrl(QUrl.fromLocalFile(selected_html))

    def rerun_logic(self):
        print("Rerunning the logic...")

############################################

    def tile_windows(self):
        # Tile all sub-windows
        self.mdi_area.tileSubWindows()
##########################################


    def init_gui(self,layout):
      grid_layout = QGridLayout()  # Create a grid layout
      # Create a QWebEngineView and load the Plotly HTML file
      self.web_view = QWebEngineView()
      self.web_view.setUrl(QUrl.fromLocalFile(self.html_file))
      self.web_view.setMinimumHeight(900)  # Adjust the height as needed
      grid_layout.addWidget(self.web_view, 0,40)
      self.plot_well_map()  # Initial plot


      well_nos = wells

      #well_label = QLabel('Select Well:')
      #self.layout.addWidget(well_label)  # Add label to the main layout

      #self.well_list = QListWidget()
      #self.well_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)  # Enable multi-selection
      #self.well_list.addItems(wells)
      #grid_layout.addWidget(self.well_list, 0, 1, 1, 2)  # Add list to grid layout

      

      # Add checkboxes for well selection directly on the map
      self.well_checkboxes = {}  # Dictionary to store checkboxes
      checkbox_layout = QVBoxLayout()
      for i, well_no in enumerate(well_nos):
        checkbox = QCheckBox(well_no)
        checkbox.setChecked(False)  # Default to selected
        checkbox.stateChanged.connect(self.on_checkbox_state_changed)  # Connect to state change handler
        self.well_checkboxes[well_no] = checkbox
        checkbox_layout.addWidget(checkbox)   # 
      
      # Create a QWidget to hold the checkboxes and set the layout
      checkbox_widget = QWidget()
      checkbox_widget.setLayout(checkbox_layout)

      # You would now need to add the checkbox_widget to your overall layout
      # For example, if using QGridLayout for the map and controls:
      grid_layout.addWidget(checkbox_widget, 0, 0)  # Add to the layout next to the map    



      # Create a QListWidget to display selected wells
      self.selected_well_list = QListWidget()
      grid_layout.addWidget(self.selected_well_list, 80, 40, len(well_nos), 1)  # Add to the grid
      
      plot_button = QPushButton('Plot Well Map')
      plot_button.clicked.connect(self.plot_well_map)
      grid_layout.addWidget(plot_button, len(well_nos) + 0, 0)  # Add button to grid

      clear_plot_button = QPushButton('Clear Plot')
      clear_plot_button.clicked.connect(self.clear_plot)
      grid_layout.addWidget(clear_plot_button, len(well_nos) + 1, 0)  # Add button to grid

      print_button = QPushButton('Print Selected Wells')
      print_button.clicked.connect(self.print_selected_wells)
      grid_layout.addWidget(print_button, len(well_nos) + 2, 0)  # Add button to grid

      #self.layout.addLayout(grid_layout)  # Add grid layout to the main layout
      layout.addLayout(grid_layout)  # Add grid layout to the main layout

      



    def on_checkbox_state_changed(self, state):
        # Determine which checkboxes are checked
        selected_wells = [well_no for well_no, checkbox in self.well_checkboxes.items() if checkbox.isChecked()]
        print("Selected wells updated:", selected_wells)
        self.update_selected_well(selected_wells)

    def update_selected_well(self, selected_wells):
        # Clear and update the selected wells list
        self.selected_wells = selected_wells
        self.selected_well_list.clear()
        self.selected_well_list.addItems(self.selected_wells)
        #self.update_connection_map()
        self.print_selected_wells()  # Print the selected wells whenever updated

    #def print_selected_wells(self):
        #print("Selected Wells:")
        #for well in self.selected_wells:
            #print(well)

        # Optionally, also display in the log
        #self.log("Selected Wells:\n" + "\n".join(self.selected_wells))

    
    



    def plot_well_map(self):
      # Plot all wells, with selected ones in blue and unselected ones in red
      fig = generate_well_map_figure()

      # Save the plot as an HTML file
      pio.write_html(fig, file=self.html_file, auto_open=False, include_plotlyjs='cdn')

      # Load the updated plot into the web view
      self.web_view.setUrl(QUrl.fromLocalFile(self.html_file))

      self.log("Well map plot updated.")

    def update_connection_map(self,fig):
      if self.selected_wells:
        well_coordinates = {}
        
        for well_no in self.selected_wells:
            specific_well = Well(survey, well_no)
            track_data = specific_well.track()
            
            # Print track_data to debug
            print(f"Track data for well {well_no}: {track_data}")

            # Adjust this line depending on the structure of track_data
            if isinstance(track_data, dict):
                # Assuming track_data contains 'x' and 'y' keys
                last_x = track_data.get('x', [])[-1] if 'x' in track_data and len(track_data['x']) > 0 else None
                last_y = track_data.get('y', [])[-1] if 'y' in track_data and len(track_data['y']) > 0 else None
            else:
                # If track_data is a DataFrame, use iloc as before
                last_x = track_data.iloc[-1]['x']
                last_y = track_data.iloc[-1]['y']
            
            well_coordinates[well_no] = (last_x, last_y)

        # Generate the connection map
        

        fig = generate_connection_map(self.selected_wells, well_coordinates,fig)
        
        # Overlay the connection plot on top of the existing well map
        # Make sure to use a full map base if necessary, or adjust the update logic

        # Save the connection plot as an HTML file
        #pio.write_html(fig, file=self.corl_html_file, auto_open=False, include_plotlyjs='cdn')

        # Load the connection plot into the web view
        #self.web_view.setUrl(QUrl.fromLocalFile(self.connection_html_file))

        self.log("Connection plot updated.")
        return fig

    def update_map_with_connections(self):
      # Plot all wells
      self.plot_well_map()
    
      # Update the connection map on top of the existing map
      #self.update_connection_map()
#################################################################

# Define a function well corl
    

    def print_selected_wells(self):

     plotter = SeismicHorizonPlotter(survey, selected_seismic, horizon_names, inline_number, time_slice,self.corl_html_file)
     

     #def plot_well_corl(self):

     alias = {
            'sonic': ['none', 'DTC', 'DT24', 'DTCO', 'DT', 'AC', 'AAC', 'DTHM'],
            'ssonic': ['none', 'DTSM', 'DTS'],
            'GR': ['none', 'GR', 'GRD', 'CGR', 'GRR', 'GRCFM'],
            'RT': ['none', 'HDRS', 'LLD', 'M2RX', 'MLR4C', 'RD', 'RT90', 'RLA1', 'RDEP', 'RLLD', 'RILD', 'ILD', 'RT_HRLT', 'RACELM', 'RT'],
            'resshal': ['none', 'LLS', 'HMRS', 'M2R1', 'RS', 'RFOC', 'ILM', 'RSFL', 'RMED', 'RACEHM'],
            'RHOZ': ['none', 'ZDEN', 'RHOB', 'RHOZ', 'RHO', 'DEN', 'RHO8', 'BDCFM'],
            'NPHI': ['none', 'CNCF', 'NPHI', 'NEU'],
            'pe': ['none','PE', 'PEF', 'PEFZ'],  # Photoelectric factor aliases
            'caliper': ['none','CALI', 'CAL', 'CALI2', 'CALX'],  # Caliper log aliases
            'bs': ['none','BS', 'BIT', 'BITSIZE', 'BDT'],  # Bit size aliases
            'vpvs': ['none','VPVS', 'VP_VS', 'VPVS_RATIO', 'VPVS_R'],  # Vp/Vs ratio aliases
            'rxo': ['none','RXO', 'RMLL', 'Rxo', 'RTXO', 'RSXO', 'RMSFL', 'MSFL', 'RXXO', 'RX', 'M2R6'],
            'sp': ['none','SP', 'SSP', 'SPONT', 'SPONPOT', 'SPOT', 'SPT', 'SP_CURVE']
        }

     

     


     abcd = ['3']
     colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'magenta',
          'teal', 'lime', 'indigo', 'yellow', 'black', 'lime', 'indigo', 'yellow', 'black']

     #well_names = ['WADU-79','kj3','NDSN-37','WADU-79','WADU-67','WADU-67','WADU-67','WADU-67','WADU-67','WADU-67']  # specify well names
     well_names = self.selected_wells
     log_names = ['GR', 'RT', 'NPHI', 'RHOZ']
     #log_names = ['gr', 'resdeep', 'neutron', 'density']

     min1 = [0, 1, 0.9, 1.65,-1000]
     max1 = [150, 20, 0.1, 2.65,1000]
     horizon_picks = [  'KIIIA-SILT', 'KIII-COAL2', 'KIIIB-SILT', 'KIV-COAL1', 'KIV-SILT', 'KIV_sand_base', 'KV-COAL', 'KV-COAL-BASE', 'KVII_SAND', 'KVIII-COAL', 'KIX-COAL', 'KIX_COAL_BASE', 'KX-COAL', 'KXSAND', 'KXI', 'YCS']
     logs_data = []
     depth_min = float('100')
     depth_max = float('1800')
     log_data = []

   
  


     global_depth_min = float('inf')
     global_depth_max = float('-inf')

    # Step 1: Determine the global depth range
     for well_name in well_names:
          file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name}.wlt"
          
          depths, times = load_depth_to_time_model(file_path)
          interp_func = create_interpolation_function(depths, times)
          well = Well(survey, well_name)
          log_data1, uom = well.logs_dataframe(log_names, zstep=0.1, upscale=False)
        
          #log_depth = log_data1['dah'].values


          depths_from_well = log_data1['dah']
          times = [depth_to_time(depth, interp_func) for depth in depths_from_well]
          log_data1['twt']=times
          md_values = log_data1['dah']
          tvdss_values = [well.tvdss(md) for md in md_values]  # Call the method for each MD value
          log_data1['tvdss'] = tvdss_values
          log_depth = log_data1['twt']*1000



          global_depth_min = min(global_depth_min, np.min(log_depth))
          global_depth_max = max(global_depth_max, np.max(log_depth))

    # Define a common depth track
     common_depth = np.linspace(global_depth_min, global_depth_max, num=8000)

    # Step 2: Interpolate log data to the common depth track
     for well_name in well_names:
          file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name}.wlt"
          
          depths, times = load_depth_to_time_model(file_path)
          interp_func = create_interpolation_function(depths, times)


          well = Well(survey, well_name)
          log_data1, uom = well.logs_dataframe(log_names, zstep=0.1, upscale=False)

          #log_depth = log_data1['dah'].values

          depths_from_well = log_data1['dah']
          times = [depth_to_time(depth, interp_func) for depth in depths_from_well]
          log_data1['twt']=times
          md_values = log_data1['dah']
          tvdss_values = [well.tvdss(md) for md in md_values]  # Call the method for each MD value
          log_data1['tvdss'] = tvdss_values
          log_depth = log_data1['twt']*1000



          logs = []

          for log_name in log_names:
            # Check if there is a valid alias for the log name
            alias_matches = [elem for elem in log_data1 if elem in set(alias[log_name])]
            selected_alias = alias_matches[0] if alias_matches else None

            if selected_alias:
                log_data = log_data1[selected_alias].values
                log_data_interp = np.interp(common_depth, log_depth, log_data)
                logs.append(log_data_interp)
            else:
                print(f"No valid data found for '{log_name}'.")
                logs.append(np.full_like(common_depth, np.nan))  # Fill with NaN if no data is available
        
          logs_data.append(logs)

     #return common_depth, logs_data

     log_depth = common_depth
     print("Log Data test:", logs_data[0][0])

     #data['GR']= data[alias['gr'][0]].values
     #if not alias['sonic']:
       # data['DT'] = 0
    
     #else:
        #data['DT'] = data[alias['sonic'][0]].values


     #fig, axs = plt.subplots(len(abcd), (len(well_names)+4)*len(log_names), figsize=(28, 8), sharey='row', sharex='col' )
     #self.update_connection_map()

    ##################################
     # Define the number of subplots (rows = len(well_names), cols = len(log_names))
     rows = 1
     #cols = (len(well_names)+4)*len(log_names)
     cols = 20
     # Create subplots
     fig = make_subplots(
        rows=rows, cols=cols,
        shared_yaxes=True, shared_xaxes=False,
        subplot_titles=log_names,specs=[[{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},
                                         {"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},
                                         {"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},
                                         {"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True}]],


        horizontal_spacing=0.001
     )

     for i, well_name1 in enumerate(well_names):
        well_name = Well(survey, well_name1)
        wellno = well_name
        well_horizon_picks = well_name.marker_info_dataframe()
        well1 = Well(survey, well_name1)

        

        file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name1}.wlt"
          
        depths, times = load_depth_to_time_model(file_path)
        interp_func = create_interpolation_function(depths, times)


          

          



        pick = well_horizon_picks['dah']
        col = well_horizon_picks['color']
        sand = well_horizon_picks['name']

        depths_from_well = well_horizon_picks['dah']

        times = [depth_to_time(depth, interp_func) for depth in depths_from_well]
        well_horizon_picks['twt']=times
        pick = well_horizon_picks['twt']*1000



        x_coord,y_coord=plot_track_data(well1)
        if i == 0:
            start_coords1 = (x_coord+100,y_coord+100)
        end_coords1 = (x_coord,y_coord)
        if i > 0:
            start_coords1 = (x_coord1,y_coord1)
        
        x_coord1=x_coord
        y_coord1=y_coord
        




        for j, log_name in enumerate(log_names):
            
            log_data = logs_data[i][j]
            if j == 2:
               log_datan = logs_data[i][j + 1]
               x_data_filtered1 = log_datan
               y_data_filtered1 = common_depth
               non_nan_mask1 = ~np.isnan(x_data_filtered1)
 
            if j == 0:
                log_data_1 = np.array(log_data[1])
            
                x_data_filtered2 = log_data


            # Ensure log_data has the right shape and dimensions
            if log_data.ndim == 1:
              x_data = log_data
              y_data = common_depth
            elif log_data.ndim == 2:
              x_data = log_data[0]
              y_data = log_data[1]
            else:
              print(f"Unexpected log data shape for '{log_name}'.")
              continue

            # Plot with x_data as the log values and y_data as depth
            non_nan_mask = ~np.isnan(x_data)
            


            # Add main log data plot
            if j < 4:
            
             fig.add_trace(
                go.Scatter(
                    x=x_data[non_nan_mask],  # Align x values with non-NaN y values
 
                    y=y_data[non_nan_mask], 
                    mode='lines',
                    line=dict(color=colors[j], width=0.5),
                    name=log_name
                ),
                row=1, col=j+(i*5) + 2
             )
             #ax = axs[(j)+(i*5)]


            # Add secondary log (twin axis)
            if j == 0:
                 plotter.plot_arbitrary_seismic_linexy_horizon(start_coords1, end_coords1,j+(i*5) + 1,fig)
  
            if j == 6:   

                fig.add_trace(
                    go.Scatter(
                        x=x_data_filtered1[non_nan_mask1], 
                        y=y_data_filtered1[non_nan_mask1],
                        mode='lines',
                        line=dict(color='red', width=0.5),
                        xaxis=f'x{cols+1}',  # Overlay on a secondary x-axis
                        name=f'{log_names[j + 1]} (Twin)'
                    ),
                    row=1, col=j+(i*5) + 2
                )
                
                # Set x-axis for twin axis
                fig.update_xaxes(range=[min1[j+1], max1[j+1]], row=1, col=j + 2)

            
            
            # Set axes properties
            if j != 1:
                 fig.update_xaxes(range=[min1[j], max1[j]], row= 1, col=j+(i*5) + 2)
            fig.update_yaxes(range=[depth_max, depth_min], autorange="reversed", row=1, col=j+(i*5) + 2)
            fig.update_xaxes(title_text=log_name, row=1, col=j+(i*5) + 2)

            # Customize x-axis tick labels
            if j == 1:
                fig.update_xaxes(type='log', row=1, col=j+(i*5) + 2)  # Log scale

            # Plot horizon picks as horizontal lines
            for k, horizon_pick in enumerate(pick):
                if horizon_pick != -9999:
                    fig.add_trace(
                        go.Scatter(
                            x=[min1[j], max1[j]], y=[horizon_pick, horizon_pick],
                            mode='lines',
                            line=dict(color='blue', width=0.5),
                            name=sand[k]
                        ),
                        row=1, col=j+(i*5) + 2
                    )
                    if j == 3:
                        fig.add_annotation(
                            x=max1[j] - 0.5,
                            y=horizon_pick,
                            text=sand[k],
                            showarrow=False,
                            font=dict(size=6, color='red'),
                            row=1, col=j+(i*5) + 2
                        )
            #fig.update_xaxes(range=[min1[j], max1[j]], row= 1, col=j+(i*5) + 1)



     # Customize layout
     fig.update_layout(
        height=2000, width=1800,
        title_text="Well Log Correlation Plot",
        legend_title_text="Logs",
        showlegend=True,
        
     )

     # Add overall depth labels
     for depth in range(int(depth_min), int(depth_max) + 1, 100):
        fig.add_annotation(
            x=-10, y=depth,
            text=str(depth),
            showarrow=False,
            font=dict(size=6),
            

        )
     
     # Update the connection map on top of the existing map
     fig=self.update_connection_map(fig)

     #fig.show()
     #plotter.plot_arbitrary_seismic_linexy_horizon(start_coords1, end_coords1,j+(i*5) + 2)
     #Save the connection plot as an HTML file
     pio.write_html(fig, file=self.corl_html_file, auto_open=False, include_plotlyjs='cdn')

     # Load the connection plot into the web view
     self.web_view.setUrl(QUrl.fromLocalFile(self.corl_html_file))

     self.log("corl plot updated.")







##############################################################
    def clear_plot(self):
        self.selected_wells.clear()
        self.selected_well_list.clear()
        self.plot_well_map()  # Refresh main map

    def log(self, message):
        # Append messages to the QTextEdit
        self.log_text_edit.append(message)


# Run the application
#app = QApplication(sys.argv)
#window = MainWindow()
#window.show()
#sys.exit(app.exec_())
