import os
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import geopandas as gpd
from dash import Dash
import threading
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
#import dash_core_components as dcc
#import dash_html_components as html
from dash import Dash, dcc, html

# Function to run the Dash server in a separate thread
def run_dash_server(app, port):
    app.run_server(debug=True, port=port)

def create_dash_app2():
  #threading.Thread(target=run_dash_server, args=(dash_app2, 8051), daemon=True).start()
  
  
  # Initialize the Dash app
  app = dash.Dash(__name__)
  
  #ogr2ogr -f "ESRI Shapefile" output_fixed.shp combined_geodata.shp -lco ENCODING=UTF-8 -nlt POLYGON
  
  
  # Paths to the shapefiles and HTML files
  shapefile_paths = {
      'plate': "./shp/plates.shp",  # Adjust this path as necessary
      'craton': "./shp/cratons.shp",  # Adjust this path as necessary    
      'oil_gas': "./shp/AU_GEOG.SHP",  # Adjust this path as necessary
      'oil_gas_shale': "./shp/au_sumg.shp",  # Adjust this path as necessary
      'geology_US': "./shp/fixed_geology.shp",  # Adjust this path as necessary
      'geology': "./shp/output_fixed.shp",  # Adjust this path as necessary
      'gprv': "./shp/global_gprv.shp",  # Adjust this path as necessary
      'lip': "./shp/Johansson_etal_2018_EarthByte_LIPs_v2.shp",  # Adjust this path as necessary
  }
  
  
  
  # Load all shapefiles into a dictionary of GeoDataFrames
  gdfs = {}
  color_maps = {
      'plate': {
          'prov_type': 'plate',
          'color_map': {
              'Eurasian Plate': 'blue',
              'Arabian Plate': 'green',
              'Pacific Plate': 'red',
              'North American Plate': 'orange',
              'South American Plate': 'yellow',
              'African Plate': 'purple',
              'Antarctic Plate': 'cyan',
              'Australian Plate': 'magenta',
              'Indian Plate': 'brown',
              'Caribbean Plate': 'pink',
              'Philippine Plate': 'lightblue',
              'Nazca Plate': 'lightgreen',
              'Juan de Fuca Plate': 'darkblue',
              'Cocos Plate': 'darkgreen',
              'Scotia Plate': 'darkred',
              'Somali Plate': 'darkorange',
              'Others': 'grey'  # For any plates not specified
          },
          'hover_attributes': ['plate', 'subplate', 'poly_name', 'plate_type', 'crust_type', 'domain']
      },
      'craton': {
          'prov_type': 'prov_type',
          'color_map': {
              'shield': '#1f77b4',  # blue
              'craton': '#ff7f0e',  # orange
              'orogenic belt': '#2ca02c',  # green
              'volcanic arc': '#d62728',  # red
              'back-arc basin': '#9467bd',  # purple
              'foredeep basin': '#8c564b',  # brown
              'wide rift': '#e377c2',  # pink
              'accretionary complex': '#7f7f7f',  # grey
              'basin': '#bcbd22',  # yellow
              'Others': '#17becf'  # cyan for any unspecified features
          },
          'hover_attributes': ['prov_type', 'prov_name', 'prov_ref','prov_group','lastorogen' ,'conjugate1']
      },
      'oil_gas': {
          'prov_type': 'O_G',
          'color_map': {
              'Oil': '#2ca02c',
              'Gas': '#d62728',
              'Others': '#17becf'  # cyan for any unspecified features
          },
          'hover_attributes': ['NAME', 'O_G', 'SRAGE','RRAGE','RRLITH' ,'SEAL']
      },
      'oil_gas_shale': {
          'prov_type': 'O_G',
          'color_map': {
              'Oil': '#2ca02c',
              'Gas': '#d62728',
              'Others': '#17becf'  # cyan for any unspecified features
          },
          'hover_attributes': ['NAME','O_G', 'KWN_OIL', 'REM_OIL','KWN_GAS','REM_GAS']
      },
      'geology_US': {
      'prov_type': 'MIN_AGE',
      'color_map': {
          'Age unknown': '#d3d3d3',       # Light gray for unknown age
          'Archean': '#800000',           # Dark red
          'Cambrian': '#66ccff',          # Light blue
          'Cretaceous': '#99ff99',        # Light green
          'Devonian': '#ff6666',          # Light red
          'Early Archean': '#cc0000',     # Red
          'Early Cretaceous': '#99cc99',  # Pale green
          'Early Proterozoic': '#cc6666', # Light red
          'Eocene': '#ff9966',            # Light orange
          'Holocene': '#ffffcc',          # Light yellow
          'Jurassic': '#6699ff',          # Medium blue
          'Late Archean': '#990000',      # Dark red
          'Late Cretaceous': '#99ff99',   # Light green
          'Late Jurassic': '#6699cc',     # Lighter blue
          'Late Proterozoic': '#e0a5a5',  # Pale red
          'Lower Cretaceous': '#99ff99',  # Light green
          'Lower Devonian': '#ff6666',    # Light red
          'Lower Eocene': '#ffcc99',      # Pale orange
          'Lower Jurassic': '#6699ff',    # Medium blue
          'Lower Mississippian': '#cc6666', # Light red
          'Lower Ordovician': '#66ccff',  # Light blue
          'Lower Pennsylvanian': '#cc9999', # Light reddish
          'Lower Permian': '#ff6666',     # Light red
          'Lower Silurian': '#66ccff',    # Light blue
          'Maastrichtian': '#99cc99',     # Pale green
          'Mesozoic': '#cccc66',          # Yellow
          'Mid-Cretaceous': '#99cc99',    # Pale green
          'Middle Archean': '#b22222',    # Firebrick
          'Middle Cambrian': '#66ccff',   # Light blue
          'Middle Cretaceous': '#99ff99', # Light green
          'Middle Devonian': '#ff6666',   # Light red
          'Middle Eocene': '#ff9966',     # Light orange
          'Middle Jurassic': '#6699ff',   # Medium blue
          'Middle Ordovician': '#66ccff', # Light blue
          'Middle Proterozoic': '#f99f99',# Pale red
          'Miocene': '#ff9966',           # Light orange
          'Mississippian': '#ff6666',     # Light red
          'Neogene': '#ff9966',           # Light orange
          'Oligocene': '#ff9966',         # Light orange
          'Ordovician': '#66ccff',        # Light blue
          'Paleocene': '#ffcc99',         # Pale orange
          'Paleogene': '#ff9966',         # Light orange
          'Paleozoic': '#ff6666',         # Light red
          'Pennsylvanian': '#ff6666',     # Light red
          'Permian': '#ff6666',           # Light red
          'Pleistocene': '#ffff99',       # Light yellow
          'Pliocene': '#ff9966',          # Light orange
          'Precambrian': '#e24152',       # Dark pink
          'Pre-Cenozoic': '#ff9966',      # Light orange
          'Pre-Cenozoic undivided': '#ff9966', # Light orange
          'Pre-Cretaceous': '#cccc66',    # Yellow
          'Pre-Mesozoic': '#ff6666',      # Light red
          'Quaternary': '#ffff99',        # Light yellow
          'Silurian': '#66ccff',          # Light blue
          'Tertiary': '#ff9966',          # Light orange
          'Triassic': '#cccc66',          # Yellow
          'Upper Cambrian': '#66ccff',    # Light blue
          'Upper Cretaceous': '#99ff99',  # Light green
          'Upper Devonian': '#ff6666',    # Light red
          'Upper Eocene': '#ffcc99',      # Pale orange
          'Upper Jurassic': '#6699ff',    # Medium blue
          'Upper Mississippian': '#cc6666',# Light red
          'Upper Ordovician': '#66ccff',  # Light blue
          'Upper Pennsylvanian': '#cc9999',# Light reddish
          'Upper Permian': '#ff6666',     # Light red
          'Others': '#17becf'  # cyan for any unspecified features
  
      },
      'hover_attributes': ['MIN_AGE']
  },
  
  
  
      'geology': {
      'prov_type': 'GLG',
      'color_map': {
          # Precambrian
          'pC': '#e24152',     # Undivided Precambrian rocks (red)
          'pCm': '#e0a5a5',    # Undivided Precambrian rocks (light red)
          'pCmi': '#f9a3a3',   # Precambrian intrusive igneous rocks (light pink)
          'pCmv': '#f99f99',   # Precambrian volcanic rocks (pink)
          
          # Paleozoic
          'Pz': '#6699cc',     # Undivided Paleozoic rocks (blue)
          'Pzi': '#6699cc',    # Paleozoic intrusive igneous rocks (blue)
          'Pzm': '#669966',    # Paleozoic metamorphics (greenish blue)
          'Pzv': '#669966',    # Paleozoic volcanic rocks (greenish blue)
          'Pzu': '#6699cc',    # Upper Paleozoic rocks (Permian, Carboniferous, Devonian) (blue)
          'Pzl': '#6699cc',    # Lower Paleozoic rocks (Silurian, Ordovician, Cambrian) (blue)
          'AD': '#cc6666',     # Precambrian - Devonian (reddish brown)
          
          # Mesozoic
          'Mz': '#6699cc',     # Undivided Mesozoic rocks (blue)
          'Mzi': '#6699cc',    # Mesozoic intrusive igneous rocks (blue)
          'Mzm': '#9966ff',    # Mesozoic metamorphic rocks (violet)
          'Mzv': '#6666cc',    # Mesozoic volcanic rocks (purple)
          'MzPz': '#6699cc',   # Mesozoic through Paleozoic undivided rocks (blue)
          'MzPzm': '#669966',  # Mesozoic through Paleozoic metamorphic rocks (greenish blue)
  
          # Cenozoic
          'Czv': '#cc99ff',    # Cenozoic volcanic rocks (light purple)
          'CzMzi': '#6699cc',  # Cenozoic through Mesozoic intrusive igneous rocks (blue)
          'CzMzv': '#6699cc',  # Cenozoic through Mesozoic volcanic rocks (blue)
          
          # Tertiary
          'T': '#ffcc99',      # Undivided Tertiary rocks (orange)
          'Tv': '#ffcc99',     # Tertiary volcanic rocks (orange)
          
          # Quaternary
          'Q': '#ffffcc',      # Undivided Quaternary rocks (yellow)
          'Qv': '#ffff99',     # Quaternary volcanic rocks (light yellow)
  
          # Miscellaneous
          'H2O': '#17becf',    # Surface water (cyan)
          'i': '#cccccc',      # Intrusive igneous rock of undetermined age (gray)
          'm': '#cccccc',      # Metamorphic rocks of undetermined age (gray)
  
          # Additional categories from previous entries
          'AD': '#cc6666',     # Precambrian - Devonian (reddish brown)
          'Cmsm': '#ffcc99',   # Cambrian sedimentary and metamorphic rocks (orange)
          'Cm': '#ffcc99',     # Cambrian (orange)
          'CmO': '#cc99ff',    # Ordovician - Cambrian (light purple)
          'Osm': '#ff99cc',    # Ordovician metamorphic and sedimentary rocks (pink)
          'OS': '#ff6699',     # Silurian - Ordovician (pinkish red)
          'S': '#ff9999',      # Silurian rocks (light red)
          'D': '#ff99cc',      # Devonian rocks (pink)
          'C': '#9966cc',      # Carboniferous rocks (dark purple)
          'Cs': '#cc99ff',     # Carboniferous sedimentary rocks (purple)
          'CP': '#cc99ff',     # Permian - Carboniferous (purple)
          'P': '#3399ff',      # Permian rocks (bright blue)
          'Pv': '#6666ff',     # Permian volcanics (bright blue)
          'TrCs': '#669966',   # Upper Carboniferous - Lower Triassic sedimentary rocks (greenish blue)
          'Tr': '#66cccc',     # Triassic rocks (blue-green)
          'JTr': '#3399cc',    # Triassic and Jurassic rocks (blueish green)
          'J': '#66cc99',      # Jurassic rocks (green)
          'Jms': '#9966ff',    # Jurassic metamorphic and sedimentary rocks (violet)
          'JK': '#66cc99',     # Cretaceous and Jurassic (greenish blue)
          'KJs': '#669999',    # Undifferentiated Jurassic and Cretaceous sedimentary rocks (teal)
          'K': '#99cc66',      # Cretaceous rocks (light green)
          'Ks': '#99cc66',     # Cretaceous sedimentary rocks (light green)
          'KTrs': '#669999',   # Middle Triassic - Lower Cretaceous sedimentary rocks (teal)
          'Cv': '#669999',     # Cretaceous - Tertiary volcanics (teal)
          'T': '#ffcc99',      # Tertiary (orange)
          'Ti': '#ffcc99',     # Tertiary igneous rocks (orange)
          'N': '#6699cc',      # Neogene sedimentary rocks (blue)
          'Pg': '#f9ec00',     # Paleogene sedimentary rocks (yellow)
          'Q': '#ffffcc',      # Quaternary sediments (yellow)
          'Qs': '#ffff99',     # Quaternary sand and dunes (light yellow)
          'Qv': '#ffff99',     # Quaternary volcanics (light yellow)
          
          # Non-geological features
          'CO': '#17becf',     # Undifferentiated Carboniferous to Ordovician rocks (cyan)
          'Gry': '#cccccc',    # Undifferentiated igneous rocks (gray)
          'H2O': '#17becf',    # Other regions (cyan)
          'SEA': '#17becf',    # Other regions (cyan)
          'Sea3': '#17becf',    # Other regions (cyan)
          'I': '#17becf',      # Water (cyan)
          'U': '#cccccc',      # Unmapped area (gray)
          '': '#ffffff',       # "unlabeled" (white)
          'Others': '#17becf'  # Cyan for any unspecified features
      },
      'hover_attributes': ['GLG']
  },
  
  
  
  
      'gprv': {
          'prov_type': 'prov_type',
          'color_map': {
              'shield': '#1f77b4',  # blue
              'craton': '#ff7f0e',  # orange
              'orogenic belt': '#2ca02c',  # green
              'volcanic arc': '#d62728',  # red
              'back-arc basin': '#9467bd',  # purple
              'foredeep basin': '#8c564b',  # brown
              'wide rift': '#e377c2',  # pink
              'accretionary complex': '#7f7f7f',  # grey
              'basin': '#bcbd22',  # yellow
              'Others': '#17becf'  # cyan for any unspecified features
          },
          'hover_attributes': ['prov_type', 'prov_name', 'prov_ref','prov_group','lastorogen' ,'conjugate1']
      },
      'lip': {
          'prov_type': 'TYPE',
          'color_map': {
              'shield': '#1f77b4',  # blue
              'craton': '#ff7f0e',  # orange
              'orogenic belt': '#2ca02c',  # green
              'volcanic arc': '#d62728',  # red
              'back-arc basin': '#9467bd',  # purple
              'foredeep basin': '#8c564b',  # brown
              'wide rift': '#e377c2',  # pink
              'accretionary complex': '#7f7f7f',  # grey
              'basin': '#bcbd22',  # yellow
              'Others': '#17becf'  # cyan for any unspecified features
          },
          'hover_attributes': ['TYPE', 'FROMAGE', 'NAME','DESCR','FEATURE_ID','REF']
      }
  }
  
  # Load GeoDataFrames
  for name, path in shapefile_paths.items():
      try:
          gdf = gpd.read_file(path)
          if gdf.crs is None:
              gdf = gdf.set_crs(epsg=4326, allow_override=True)
          gdfs[name] = gdf.to_crs(epsg=4326)
      except Exception as e:
          print(f"Error loading {name}: {e}")
  
  # Function to create hover text based on the selected GeoDataFrame
  def generate_hover_text(gdf, file_name):
      if file_name in color_maps:
          attrs = color_maps[file_name]['hover_attributes']
          return gdf.apply(lambda row: '<br>'.join([f"{attr}: {row[attr]}" for attr in attrs]), axis=1)
      return gdf.apply(lambda row: "No relevant data available", axis=1)
  
  # Create the Plotly figure based on selected shapefile
  def create_world_map(selected_gdf, file_name):
      prov_type_column = color_maps[file_name]['prov_type']
      selected_color_map = color_maps[file_name]['color_map']
      
      # Map colors based on the specific column for the selected file
      selected_gdf['color'] = selected_gdf[prov_type_column].map(selected_color_map).fillna(selected_color_map['Others'])
  
      # Create the figure
      fig = go.Figure()
  
      # Add a choropleth layer for the selected GeoDataFrame
      fig.add_trace(go.Choropleth(
          geojson=selected_gdf.geometry.__geo_interface__,  # Use GeoPandas interface
          locations=selected_gdf.index,  # Use the index for the locations
          z=selected_gdf[prov_type_column].astype('category').cat.codes,  # Encode prov_type for colors
          colorscale=list(selected_color_map.values()),  # Use your defined colors
          marker_line_width=1,
          marker_line_color='darkgrey',
          showscale=False,  # We won't show a color scale since we're using discrete colors
          hoverinfo='text',
          text=generate_hover_text(selected_gdf, file_name)  # Use the dynamic hover text function
      ))
  
      # Update layout for the world map
      fig.update_geos(
          projection_type="natural earth",
          showland=True,
          landcolor="lightgreen",
          showocean=True,
          oceancolor="lightblue",
          showcountries=True,
          countrycolor="black"
      )
  
      # Update figure layout
      fig.update_layout(
          title=f"World Map with {file_name} Features",
          height=800,
          margin=dict(l=0, r=0, t=50, b=0)
      )
  
      return fig
  
  def create_color_index(file_name):
      color_map = color_maps[file_name]['color_map']
      
      # Create a list of legend items with better structure
      color_index_items = [
          html.Div(
              children=[
                  html.Div(
                      style={
                          'backgroundColor': color,
                          'height': '20px',
                          'width': '20px',
                          'display': 'inline-block',
                          'margin-right': '5px'
                      }
                  ),
                  html.Span(key, style={'margin-right': '10px'})
              ],
              style={'display': 'flex', 'alignItems': 'center'}
          )
          for key, color in color_map.items()
      ]
      
      return html.Div(color_index_items, style={'padding': '10px'})
  
  
  # Define the layout for the Dash app
  app.layout = html.Div([
      html.H1("Geological Features Map"),
      dcc.Dropdown(
          id='shapefile-selector',
          options=[
              {'label': name, 'value': name} for name in gdfs.keys()
          ] + [{'label': 'Bouguer Gravity', 'value': 'Bouguer Gravity'}] + [{'label': 'geomag', 'value': 'geomag'}],  # Add PNG option to the dropdown
          value='craton',  # Default value
          clearable=False
      ),
      # Image element to display either the map or the PNG image
      html.Div(id='display-output')
  ])
  
  # Callback to update the displayed content (either map or PNG image)
  @app.callback(
      Output('display-output', 'children'),
      Input('shapefile-selector', 'value')
  )
  def update_output(selected_file):
      if selected_file == 'Bouguer Gravity':
          # Display the PNG image
          return html.Iframe(
              src='/assets/fig3.html',  # Path to the PNG file
              style={'width': '100%', 'height': '900px'}  # Adjust size as needed
          )
      elif selected_file == 'geomag':
          # If you still want to keep the PNG option, uncomment this part
          return html.Img(
              src='/assets/geomag.png',  # Path to the PNG file in the assets folder
              style={'width': '100%', 'height': '900px'}  # Adjust size as needed
          )
      else:
          # Display the world map for the selected shapefile
          selected_gdf = gdfs[selected_file]
          color_index = create_color_index(selected_file)  # Create the color index
          return dcc.Graph(figure=create_world_map(selected_gdf, selected_file)), color_index
  return app
#if __name__ == '__main__':
    #dash_app = create_dash_app()
    
    #dash_app.run_server(debug=True)