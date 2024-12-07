import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import plotly.graph_objects as go
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from odbind.survey import Survey
from odbind.seismic3d import Seismic3D
from odbind.well import Well
from odbind.horizon3d import Horizon3D
import plotly.io as pio
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


class SeismicHorizonPlotter:
    def __init__(self, survey, seismic_name, horizon_names, inline_number, time_slice, html_file):
        self.survey = survey
        self.seismic_name = seismic_name
        self.horizon_names = horizon_names
        self.inline_number = inline_number
        self.time_slice = time_slice
        self.html_file  = html_file
        self.seismic = self._load_seismic_data()
        self.horizons = self._load_existing_horizons()        


    
    def _load_seismic_data(self):
        """Load seismic data."""
        seismic = Seismic3D(self.survey, self.seismic_name)
        return seismic
    
    def _load_existing_horizons(self):
        """Load existing horizons from the survey."""
        horizons = []
        for horizon_name in self.horizon_names:
            try:
                horizon = Horizon3D(self.survey, horizon_name)
                horizons.append((horizon_name, horizon))
            except Exception as e:
                print(f"Error loading horizon {horizon_name}: {e}")
        return horizons

    def fetch_seismic_data(self):
      """Fetch seismic data from the specified time slice."""
      try:
        # Get inline, crossline, and time (z) ranges
        ranges = self.seismic.ranges
        inl_range = ranges.inlrg
        crl_range = ranges.crlrg
        z_range = ranges.zrg
        print("Ranges:", inl_range, crl_range, z_range)

        # Fetch seismic data
        data = self.seismic.getdata(inl_range, crl_range, z_range)
        print("Fetched Data:", data)

        # Check if the returned data is an xarray.Dataset or a tuple
        if isinstance(data, xr.Dataset):
            # Process xarray.Dataset
            seismic_data = data
            x_coords = seismic_data.coords['x'].values
            y_coords = seismic_data.coords['y'].values
            z_coords = seismic_data.coords['twt'].values
            z_values = seismic_data.to_array().values  # Ensure z_values is a numpy array

            return x_coords, y_coords, z_coords, z_values

        elif isinstance(data, tuple):
            # Process tuple
            seismic_arrays, info = data
            dims = info['dims']

            if 'xline' in dims and 'iline' in dims and 'twt' in dims:
                seismic_array = seismic_arrays[0]  # Assuming single component
                x_coords = np.array(info['x'])
                y_coords = np.array(info['y'])
                z_coords = np.array(info['twt'])
                z_values = np.array(seismic_array)  # Ensure z_values is a numpy array

                return x_coords, y_coords, z_coords, z_values
            else:
                raise ValueError("Unexpected dimensions in seismic data.")
        else:
            raise ValueError("Unexpected return format from getdata.")
      except Exception as e:
        print(f"Error fetching seismic data: {e}")
        return None, None, None, None

    def fetch_seismic_data1(self):
      """Fetch seismic data from the specified time slice."""
      try:
        # Get inline, crossline, and time (z) ranges
        ranges = self.seismic.ranges
        inl_range = ranges.inlrg
        crl_range = ranges.crlrg
        z_range = ranges.zrg
        print("Ranges:", inl_range, crl_range, z_range)

        # Fetch seismic data
        data = self.seismic.getdata(inl_range, crl_range, z_range)
        print("Fetched Data:", data)

        # Check if the returned data is an xarray.Dataset or a tuple
        if isinstance(data, xr.Dataset):
            # Process xarray.Dataset
            seismic_data = data
            x_coords = seismic_data.coords['xline'].values
            y_coords = seismic_data.coords['iline'].values
            z_coords = seismic_data.coords['twt'].values
            z_values = seismic_data.to_array().values  # Ensure z_values is a numpy array

            return x_coords, y_coords, z_coords, z_values

        elif isinstance(data, tuple):
            # Process tuple
            seismic_arrays, info = data
            dims = info['dims']

            if 'xline' in dims and 'iline' in dims and 'twt' in dims:
                seismic_array = seismic_arrays[0]  # Assuming single component
                x_coords = np.array(info['xline'])
                y_coords = np.array(info['iline'])
                z_coords = np.array(info['twt'])
                z_values = np.array(seismic_array)  # Ensure z_values is a numpy array

                return x_coords, y_coords, z_coords, z_values
            else:
                raise ValueError("Unexpected dimensions in seismic data.")
        else:
            raise ValueError("Unexpected return format from getdata.")
      except Exception as e:
        print(f"Error fetching seismic data: {e}")
        return None, None, None, None





    def plot_arbitrary_seismic_linexy(self, start_coords, end_coords):
      """
      Plot a seismic section along an arbitrary line defined by start and end coordinates,
      showing both inline and crossline numbers.
    
      Parameters
      ----------
      start_coords : tuple
        Starting (inline, crossline) coordinates of the line.
      end_coords : tuple
        Ending (inline, crossline) coordinates of the line.
      """
      x_coords, y_coords, z_coords, z_values = self.fetch_seismic_data()
      print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, z_coords shape: {z_coords.shape}")
      print(f"z_values shape: {z_values.shape}")
      print(z_coords.max(), z_coords.min())
      print(x_coords, y_coords)
      x_coords1, y_coords1, z_coords1, z_values1 = self.fetch_seismic_data1()
      print(f"x_coords1 shape: {x_coords1.shape}, y_coords1 shape: {y_coords1.shape}, z_coords shape: {z_coords1.shape}")
      print(f"z_values shape: {z_values1.shape}")
      print(z_coords1.max(), z_coords1.min())
      print(x_coords1, y_coords1)

      start_coords = np.array(start_coords)
      end_coords = np.array(end_coords)

      
      # Store the original shapes of x_coords and y_coords before flattening
      original_shape_x = x_coords.shape
      original_shape_y = y_coords.shape


      # Ensure x_coords and y_coords are 1D arrays for distance calculations
      x_coords_flat = np.ravel(x_coords)
      y_coords_flat = np.ravel(y_coords)
     

      # Initialize variables to store the minimum distances and corresponding indices
      min_dist_start = float('inf')
      min_dist_end = float('inf')
      nearest_start_idx = -1
      nearest_end_idx = -1

      # Find the nearest point to the start coordinates
      for i in range(len(x_coords_flat)):
        # Calculate the Euclidean distance for the current point
        dist_start = np.sqrt((x_coords_flat[i] - start_coords[0])**2 + (y_coords_flat[i] - start_coords[1])**2)
        if dist_start < min_dist_start:
            min_dist_start = dist_start
            nearest_start_idx_flat = i

      # Find the nearest point to the end coordinates
      for i in range(len(x_coords_flat)):
        dist_end = np.sqrt((x_coords_flat[i] - end_coords[0])**2 + (y_coords_flat[i] - end_coords[1])**2)
        if dist_end < min_dist_end:
            min_dist_end = dist_end
            nearest_end_idx_flat = i
      
      


      nearest_start_idx = np.unravel_index(nearest_start_idx_flat, original_shape_x)
      nearest_end_idx = np.unravel_index(nearest_end_idx_flat, original_shape_x)

      print("Nearest start and end indices (first dataset):", nearest_start_idx, nearest_end_idx)

      # Get the nearest coordinates from the first dataset
      nearest_start_coords = (x_coords[nearest_start_idx], y_coords[nearest_start_idx])
      nearest_end_coords = (x_coords[nearest_end_idx], y_coords[nearest_end_idx])

      print(f"Nearest start coordinates (first dataset): {nearest_start_coords}")
      print(f"Nearest end coordinates (first dataset): {nearest_end_coords}")

      # Get the actual nearest values, not just indices, from the second dataset
      nearest_start_value1 = (y_coords1[nearest_start_idx[0]], x_coords1[nearest_start_idx[1]])
      nearest_end_value1 = (y_coords1[nearest_end_idx[0]], x_coords1[nearest_end_idx[1]])


      print(f"Nearest start values (second dataset): {nearest_start_value1}")
      print(f"Nearest end values (second dataset): {nearest_end_value1}")

      self.plot_arbitrary_seismic_line(nearest_start_value1, nearest_end_value1)

      plt.show()

    
    def plot_arbitrary_seismic_linexy_horizon(self, start_coords1, end_coords1,col1,fig):
      """
      Plot a seismic section along an arbitrary line defined by start and end coordinates,
      showing both inline and crossline numbers.
    
      Parameters
      ----------
      start_coords : tuple
        Starting (inline, crossline) coordinates of the line.
      end_coords : tuple
        Ending (inline, crossline) coordinates of the line.
      """
      x_coords, y_coords, z_coords, z_values = self.fetch_seismic_data()
      print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, z_coords shape: {z_coords.shape}")
      print(f"z_values shape: {z_values.shape}")
      print(z_coords.max(), z_coords.min())
      print(x_coords, y_coords)
      x_coords1, y_coords1, z_coords1, z_values1 = self.fetch_seismic_data1()
      print(f"x_coords1 shape: {x_coords1.shape}, y_coords1 shape: {y_coords1.shape}, z_coords shape: {z_coords1.shape}")
      print(f"z_values shape: {z_values1.shape}")
      print(z_coords1.max(), z_coords1.min())
      print(x_coords1, y_coords1)

      start_coords = np.array(start_coords1)
      end_coords = np.array(end_coords1)

      
      # Store the original shapes of x_coords and y_coords before flattening
      original_shape_x = x_coords.shape
      original_shape_y = y_coords.shape


      # Ensure x_coords and y_coords are 1D arrays for distance calculations
      x_coords_flat = np.ravel(x_coords)
      y_coords_flat = np.ravel(y_coords)
     

      # Initialize variables to store the minimum distances and corresponding indices
      min_dist_start = float('inf')
      min_dist_end = float('inf')
      nearest_start_idx = -1
      nearest_end_idx = -1

      # Find the nearest point to the start coordinates
      for i in range(len(x_coords_flat)):
        # Calculate the Euclidean distance for the current point
        dist_start = np.sqrt((x_coords_flat[i] - start_coords[0])**2 + (y_coords_flat[i] - start_coords[1])**2)
        if dist_start < min_dist_start:
            min_dist_start = dist_start
            nearest_start_idx_flat = i

      # Find the nearest point to the end coordinates
      for i in range(len(x_coords_flat)):
        dist_end = np.sqrt((x_coords_flat[i] - end_coords[0])**2 + (y_coords_flat[i] - end_coords[1])**2)
        if dist_end < min_dist_end:
            min_dist_end = dist_end
            nearest_end_idx_flat = i
      
      


      nearest_start_idx = np.unravel_index(nearest_start_idx_flat, original_shape_x)
      nearest_end_idx = np.unravel_index(nearest_end_idx_flat, original_shape_x)

      print("Nearest start and end indices (first dataset):", nearest_start_idx, nearest_end_idx)

      # Get the nearest coordinates from the first dataset
      nearest_start_coords = (x_coords[nearest_start_idx], y_coords[nearest_start_idx])
      nearest_end_coords = (x_coords[nearest_end_idx], y_coords[nearest_end_idx])

      print(f"Nearest start coordinates (first dataset): {nearest_start_coords}")
      print(f"Nearest end coordinates (first dataset): {nearest_end_coords}")

      # Get the actual nearest values, not just indices, from the second dataset
      nearest_start_value1 = (y_coords1[nearest_start_idx[0]], x_coords1[nearest_start_idx[1]])
      nearest_end_value1 = (y_coords1[nearest_end_idx[0]], x_coords1[nearest_end_idx[1]])


      print(f"Nearest start values (second dataset): {nearest_start_value1}")
      print(f"Nearest end values (second dataset): {nearest_end_value1}")

      self.plot_arbitrary_seismic_line_and_horizons(nearest_start_value1, nearest_end_value1,start_coords1, end_coords1,col1,fig)

      #plt.show()







    from scipy.spatial import KDTree
    def find_nearest_points1(self, x_coords, y_coords, line_points):
      # Reshape x_coords and y_coords if they are 2D arrays
      # Create 2D mesh grid from the 1D x and y coordinates
      X, Y = np.meshgrid(x_coords, y_coords)

      # Flatten the mesh grid to 1D arrays
      x_flat = X.flatten()
      y_flat = Y.flatten()

      # Initialize arrays to hold the nearest points
      nearest_x = []
      nearest_y = []


      # Loop over each point on the generated line
      for point in line_points:
        # Calculate Euclidean distances to all grid points
        distances = np.sqrt((x_flat - point[0])**2 + (y_flat - point[1])**2)
        # Find the index of the minimum distance
        nearest_idx = np.argmin(distances)
        # Append the nearest point's x and y coordinates
        nearest_x.append(x_flat[nearest_idx])
        nearest_y.append(y_flat[nearest_idx])

      return np.array(nearest_x), np.array(nearest_y)






    def find_nearest_points(self, x_coords, y_coords, line_points):
      # Reshape x_coords and y_coords if they are 2D arrays
      if len(x_coords.shape) > 1 and len(y_coords.shape) > 1:
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
      else:
        x_flat = x_coords
        y_flat = y_coords

      # Initialize arrays to hold the nearest points
      nearest_x = []
      nearest_y = []

      # Loop over each point on the generated line
      for point in line_points:
        # Calculate Euclidean distances to all grid points
        distances = np.sqrt((x_flat - point[0])**2 + (y_flat - point[1])**2)
        # Find the index of the minimum distance
        nearest_idx = np.argmin(distances)
        # Append the nearest point's x and y coordinates
        nearest_x.append(x_flat[nearest_idx])
        nearest_y.append(y_flat[nearest_idx])

      return np.array(nearest_x), np.array(nearest_y)


    def plot_arbitrary_seismic_linexy_horizon_old(self, start_coords, end_coords):
      """
      Plot a seismic section along an arbitrary line defined by start and end coordinates,
      showing both inline and crossline numbers.
    
      Parameters
      ----------
      start_coords : tuple
        Starting (inline, crossline) coordinates of the line.
      end_coords : tuple
        Ending (inline, crossline) coordinates of the line.
      """
      x_coords, y_coords, z_coords, z_values = self.fetch_seismic_data()
      print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, z_coords shape: {z_coords.shape}")
      print(f"z_values shape: {z_values.shape}")
      print(z_coords.max(), z_coords.min())
      print(x_coords, y_coords)
      x_coords1, y_coords1, z_coords1, z_values1 = self.fetch_seismic_data1()
      print(f"x_coords1 shape: {x_coords1.shape}, y_coords1 shape: {y_coords1.shape}, z_coords shape: {z_coords1.shape}")
      print(f"z_values shape: {z_values1.shape}")
      print(z_coords1.max(), z_coords1.min())
      print(x_coords1, y_coords1)

      start_coords = np.array(start_coords)
      end_coords = np.array(end_coords)

      
      # Store the original shapes of x_coords and y_coords before flattening
      original_shape_x = x_coords.shape
      original_shape_y = y_coords.shape


      # Ensure x_coords and y_coords are 1D arrays for distance calculations
      x_coords_flat = np.ravel(x_coords)
      y_coords_flat = np.ravel(y_coords)
     

      # Initialize variables to store the minimum distances and corresponding indices
      min_dist_start = float('inf')
      min_dist_end = float('inf')
      nearest_start_idx = -1
      nearest_end_idx = -1

      # Find the nearest point to the start coordinates
      for i in range(len(x_coords_flat)):
        # Calculate the Euclidean distance for the current point
        dist_start = np.sqrt((x_coords_flat[i] - start_coords[0])**2 + (y_coords_flat[i] - start_coords[1])**2)
        if dist_start < min_dist_start:
            min_dist_start = dist_start
            nearest_start_idx_flat = i

      # Find the nearest point to the end coordinates
      for i in range(len(x_coords_flat)):
        dist_end = np.sqrt((x_coords_flat[i] - end_coords[0])**2 + (y_coords_flat[i] - end_coords[1])**2)
        if dist_end < min_dist_end:
            min_dist_end = dist_end
            nearest_end_idx_flat = i
      
      


      nearest_start_idx = np.unravel_index(nearest_start_idx_flat, original_shape_x)
      nearest_end_idx = np.unravel_index(nearest_end_idx_flat, original_shape_x)

      print("Nearest start and end indices (first dataset):", nearest_start_idx, nearest_end_idx)

      # Get the nearest coordinates from the first dataset
      nearest_start_coords = (x_coords[nearest_start_idx], y_coords[nearest_start_idx])
      nearest_end_coords = (x_coords[nearest_end_idx], y_coords[nearest_end_idx])

      print(f"Nearest start coordinates (first dataset): {nearest_start_coords}")
      print(f"Nearest end coordinates (first dataset): {nearest_end_coords}")

      # Get the actual nearest values, not just indices, from the second dataset
      nearest_start_value1 = (y_coords1[nearest_start_idx[0]], x_coords1[nearest_start_idx[1]])
      nearest_end_value1 = (y_coords1[nearest_end_idx[0]], x_coords1[nearest_end_idx[1]])


      print(f"Nearest start values (second dataset): {nearest_start_value1}")
      print(f"Nearest end values (second dataset): {nearest_end_value1}")
      plt.figure(figsize=(12, 8))
      ax = plt.gca()

      self.plot_arbitrary_seismic_line(nearest_start_value1, nearest_end_value1)
      ax2 = ax.twiny()
      # Overlay horizons, filtering by x- and y-coordinates
      for horizon_name, horizon in self.horizons:
        print(horizon_name)
        try:
            result = horizon.getdata()
            # Resample seismic data along the arbitrary line
            

            # Define the number of points along the arbitrary line
            num_points = 500
            
            #start_coords = np.array(start_coords)
            #end_coords = np.array(end_coords)


            # Generate points along the arbitrary line
            inline_line = np.linspace(start_coords[0], end_coords[0], num_points)
            crossline_line = np.linspace(start_coords[1], end_coords[1], num_points)

            # Find the nearest integer inline and crossline indices
            nearest_inline_indices = np.searchsorted(y_coords[0], np.rint(inline_line)) - 1
            nearest_crossline_indices = np.searchsorted(x_coords[0], np.rint(crossline_line)) - 1
            #print('done')
            # Clip the indices to ensure they are within bounds
            nearest_inline_indices = np.clip(nearest_inline_indices, 0, len(y_coords) - 1)
            nearest_crossline_indices = np.clip(nearest_crossline_indices, 0, len(x_coords) - 1)
            #print(x_coords[nearest_inline_indices],y_coords[nearest_crossline_indices])

            # Combine inline and crossline points to get (x, y) pairs
            line_points = np.column_stack((inline_line, crossline_line))

            # Find the nearest points using Euclidean distance
            nearest_x, nearest_y = self.find_nearest_points(x_coords, y_coords, line_points)


            #plt.figure(figsize=(10, 8))
    
            # Plot the generated line
            #plt.plot(inline_line, crossline_line, 'r-', label='Generated Line')
    
            # Highlight start and end points (this is the full set, not just the start/end)
            #plt.scatter(nearest_x, nearest_y, color='blue', label='Nearest Points', zorder=5)

            # Ensure that you also plot the actual start and end points for clarity
            #plt.scatter([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], color='green', label='Start/End Points', zorder=6)
    
            #plt.title("Generated Line Between Start and End Coordinates in X-Y Space")
            #plt.xlabel("X Coordinates")
            #plt.ylabel("Y Coordinates")
            #plt.legend()
            #plt.grid(True)
            #plt.show()


            if isinstance(result, tuple):
                data, info = result
                if len(data) == 1:
                    z = data[0]  # z-values (horizon depth)
                    x_data = info['x']  # x-coordinates of the horizon
                    y_data = info['y']  # y-coordinates (inline/crossline)

                    # Convert to arrays for easier manipulation
                    x_data = np.array(x_data)
                    y_data = np.array(y_data)
                    z = np.array(z)

                    # Filter horizon points that match the x- and y-coordinates of the seismic inline
                    mask = np.isin(x_data, nearest_x) & np.isin(y_data, nearest_y)

                    filtered_x = x_data[mask]
                    filtered_y = y_data[mask]
                    filtered_z = z[mask]

                    # Calculate cumulative distance along the line for the horizons
                    if len(filtered_x) > 1:
                       differences_x = np.diff(filtered_x)
                       differences_y = np.diff(filtered_y)
                       distances = np.sqrt(differences_x**2 + differences_y**2)
                       horizon_distance = np.insert(np.cumsum(distances), 0, 0)
                    else:
                       horizon_distance = np.zeros_like(filtered_z)  # Handle the case where there’s only 1 point



                    # Plot only filtered horizon values along the distance line
                    ax.plot(horizon_distance, filtered_z, label=f"{horizon_name} Horizon")

                else:
                    print(f"Unexpected number of data arrays returned")

            elif hasattr(result, 'data_vars'):
                x_data = result['x'].values
                y_data = result['y'].values
                z = result['z'].values

                x_data = np.array(x_data)
                y_data = np.array(y_data)
                z = np.array(z)

                # Filter horizon points to match x- and y-coordinates of the seismic inline
                mask = np.isin(x_data, nearest_x) & np.isin(y_data, nearest_y)


                tolerance = 1.0  # Set tolerance to 1 unit, you can adjust this value

                # Filter horizon points that match the x- and y-coordinates of the seismic inline, allowing for some tolerance
                #mask = (
                   #(np.abs(x_data - x_coords[nearest_inline_indices]) <= tolerance) &
                   #(np.abs(y_data - y_coords[nearest_crossline_indices]) <= tolerance)
                #)





                
                filtered_x = x_data[mask]
                filtered_y = y_data[mask]
                filtered_z = z[mask]

                # Calculate cumulative distance along the line for the horizons
                if len(filtered_x) > 1:
                       differences_x = np.diff(filtered_x)
                       differences_y = np.diff(filtered_y)
                       distances = np.sqrt(differences_x**2 + differences_y**2)
                       horizon_distance = np.insert(np.cumsum(distances), 0, 0)
                else:
                       horizon_distance = np.zeros_like(filtered_z)  # Handle the case where there’s only 1 point





                # Plot only filtered horizon values along the distance line
                ax.plot(horizon_distance, filtered_z, label=f"{horizon_name} Horizon")


        except Exception as e:
            print(f"Error plotting horizon {horizon_name}: {e}")

      ax2.legend()
      #ax2.xlabel('x y')
      #ax2.ylabel('Z Coordinate (Time/Depth)')
      #ax2.title(f"Seismic Inline {self.inline_number} with Horizons")
      plt.show()






    def plot_arbitrary_seismic_line(self, start_coords, end_coords):
      """
      Plot a seismic section along an arbitrary line defined by start and end coordinates,
      showing both inline and crossline numbers.
    
      Parameters
      ----------
      start_coords : tuple
        Starting (inline, crossline) coordinates of the line.
      end_coords : tuple
        Ending (inline, crossline) coordinates of the line.
      """
      x_coords, y_coords, z_coords, z_values = self.fetch_seismic_data1()
      print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, z_coords shape: {z_coords.shape}")
      print(f"z_values shape: {z_values.shape}")
      print(z_coords.max(), z_coords.min())
      if x_coords is not None and y_coords is not None and z_values is not None:
        try:
            # Define the number of points along the arbitrary line
            num_points = 500

            # Generate points along the arbitrary line
            inline_line = np.linspace(start_coords[0], end_coords[0], num_points)
            crossline_line = np.linspace(start_coords[1], end_coords[1], num_points)

            # Find the nearest integer inline and crossline indices
            nearest_inline_indices = np.searchsorted(y_coords, np.rint(inline_line)) - 1
            nearest_crossline_indices = np.searchsorted(x_coords, np.rint(crossline_line)) - 1

            # Clip the indices to ensure they are within bounds
            nearest_inline_indices = np.clip(nearest_inline_indices, 0, len(y_coords) - 1)
            nearest_crossline_indices = np.clip(nearest_crossline_indices, 0, len(x_coords) - 1)

            # Extract seismic values along the nearest inline and crossline points
            seismic_values_along_line = z_values[0, nearest_inline_indices, nearest_crossline_indices, :]

            # Generate a combined axis label showing both inline and crossline numbers
            line_labels = [f"Inl: {int(inl)}, Crl: {int(crl)}"
                           for inl, crl in zip(inline_line, crossline_line)]

            # Plot the seismic data along the arbitrary line
            extent = [0, num_points - 1, z_coords.max(), z_coords.min()]
            #plt.figure(figsize=(12, 8))
            #ax = plt.gca()
            plt.imshow(seismic_values_along_line.T, cmap='RdGy', aspect='auto', extent=extent)
            plt.colorbar(label='Amplitude')
            plt.title(f'Arbitrary Seismic Line from {start_coords} to {end_coords}')
            plt.xlabel('Inline and Crossline Numbers')
            plt.ylabel('Time/Depth (TWT or Depth)')

            # Set x-ticks to show inline and crossline numbers
            num_ticks = 10  # Choose how many tick marks you want to show
            tick_positions = np.linspace(0, num_points - 1, num_ticks).astype(int)
            plt.xticks(tick_positions, [line_labels[i] for i in tick_positions], rotation=45)
            
            #plt.show()

        except Exception as e:
            print(f"Error plotting arbitrary seismic line: {e}")
      else:
        print("Error: No seismic data to plot.")

    def plot_arbitrary_seismic_line_old(self, start_coords, end_coords):
      """
      Plot a seismic section along an arbitrary line defined by start and end coordinates,
      showing both inline and crossline numbers.
    
      Parameters
      ----------
      start_coords : tuple
        Starting (inline, crossline) coordinates of the line.
      end_coords : tuple
        Ending (inline, crossline) coordinates of the line.
      """
      x_coords, y_coords, z_coords, z_values = self.fetch_seismic_data1()
      print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, z_coords shape: {z_coords.shape}")
      print(f"z_values shape: {z_values.shape}")
      print(z_coords.max(), z_coords.min())
      if x_coords is not None and y_coords is not None and z_values is not None:
        try:
            # Define the number of points along the arbitrary line
            num_points = 500

            # Generate points along the arbitrary line
            inline_line = np.linspace(start_coords[0], end_coords[0], num_points)
            crossline_line = np.linspace(start_coords[1], end_coords[1], num_points)

            # Find the nearest integer inline and crossline indices
            nearest_inline_indices = np.searchsorted(y_coords, np.rint(inline_line)) - 1
            nearest_crossline_indices = np.searchsorted(x_coords, np.rint(crossline_line)) - 1

            # Clip the indices to ensure they are within bounds
            nearest_inline_indices = np.clip(nearest_inline_indices, 0, len(y_coords) - 1)
            nearest_crossline_indices = np.clip(nearest_crossline_indices, 0, len(x_coords) - 1)

            # Extract seismic values along the nearest inline and crossline points
            seismic_values_along_line = z_values[0, nearest_inline_indices, nearest_crossline_indices, :]

            # Generate a combined axis label showing both inline and crossline numbers
            line_labels = [f"Inl: {int(inl)}, Crl: {int(crl)}"
                           for inl, crl in zip(inline_line, crossline_line)]

            # Plot the seismic data along the arbitrary line
            extent = [0, num_points - 1, z_coords.max(), z_coords.min()]
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            plt.imshow(seismic_values_along_line.T, cmap='RdGy', aspect='auto', extent=extent)
            plt.colorbar(label='Amplitude')
            plt.title(f'Arbitrary Seismic Line from {start_coords} to {end_coords}')
            plt.xlabel('Inline and Crossline Numbers')
            plt.ylabel('Time/Depth (TWT or Depth)')

            # Set x-ticks to show inline and crossline numbers
            num_ticks = 10  # Choose how many tick marks you want to show
            tick_positions = np.linspace(0, num_points - 1, num_ticks).astype(int)
            plt.xticks(tick_positions, [line_labels[i] for i in tick_positions], rotation=45)
            
            plt.show()

        except Exception as e:
            print(f"Error plotting arbitrary seismic line: {e}")
      else:
        print("Error: No seismic data to plot.")


    
    

    

    def plot_seismic_inline_and_horizons(self):
      """Plot seismic inline with horizons and show crossline numbers on the top axis."""
      fig, ax = plt.subplots(figsize=(12, 8))

      try:
        # Access seismic data using the iline property
        iline_slice = self.seismic.iline[self.inline_number]
        
        # Extract x-coordinates (crossline), y-coordinates (inline), and z-coordinates (TWT or depth)
        x_coords = iline_slice['x'].values  # Crossline coordinates
        y_coords = iline_slice['y'].values  # Inline coordinates
        z_coords = iline_slice.coords['twt'].values  # Time or Depth axis (TWT or depth)

        if self.seismic.comp_names:
            firstcomp = self.seismic.comp_names[0]
            seismic_inline = iline_slice[firstcomp].values  # Extract the seismic data
            
            # Resample seismic data along the arbitrary line
            num_points = len(x_coords)
            x_line = np.linspace(x_coords.min(), x_coords.max(), num_points)
            y_line = np.linspace(y_coords.min(), y_coords.max(), num_points)

            # Combine x and y coordinates into a single axis along the inline
            distance_along_line = np.sqrt((x_line - x_coords.min())**2 + (y_line - y_coords.min())**2)

            # Transpose the seismic data to match plotting conventions (time/depth vs distance)
            seismic_inline = np.transpose(seismic_inline)

            # Define the extent along the resampled line and z-coordinates
            extent = [distance_along_line.min(), distance_along_line.max(), z_coords.max(), z_coords.min()]

            # Plot the seismic data
            cax = ax.imshow(seismic_inline, cmap='RdGy', aspect='auto', extent=extent)
            #ax.set_title(f"Seismic Inline at iline {self.inline_number}")
            ax.set_xlabel('Distance Along Line, x, y ')
            ax.set_ylabel('Z Coordinate (Time/Depth)')
            fig.colorbar(cax, ax=ax, label='Amplitude')

            # Create a secondary x-axis on the top for crossline numbers
            ax_top = ax.twiny()
            #ax_top.set_xlim(ax.get_xlim())  # Sync the limits with the bottom axis

            # Set the x_coords as ticks on the top axis
            x_tick_positions = np.linspace(0, num_points - 1, 10).astype(int)  # Choose 10 ticks
            x_ticks = x_coords[x_tick_positions]
            ax_top.set_xticks(x_tick_positions)
            ax_top.set_xticklabels(x_ticks.astype(int), rotation=0)
            #ax_top.set_xlabel('X Coordinates (Crossline)')

            # Create the second secondary x-axis for y_coords (inline)
            ax_top_y = ax.twiny()
            ax_top_y.spines['top'].set_position(('outward', 20))  # Offset this axis from the first
            #ax_top_y.set_xlim(ax.get_xlim())  # Sync the limits with the bottom axis

            # Set the y_coords as ticks on the second top axis
            y_tick_positions = np.linspace(0, num_points - 1, 10).astype(int)  # Choose 10 ticks
            y_ticks = y_coords[y_tick_positions]
            ax_top_y.set_xticks(y_tick_positions)
            ax_top_y.set_xticklabels(y_ticks.astype(int), rotation=0)
            #ax_top_y.set_xlabel('Y Coordinates (Inline)')


      except Exception as e:
        print(f"Error plotting seismic inline: {e}")
        return

      # Overlay horizons, filtering by x- and y-coordinates
      for horizon_name, horizon in self.horizons:
        print(horizon_name)
        try:
            result = horizon.getdata()

            if isinstance(result, tuple):
                data, info = result
                if len(data) == 1:
                    z = data[0]  # z-values (horizon depth)
                    x_data = info['x']  # x-coordinates of the horizon
                    y_data = info['y']  # y-coordinates (inline/crossline)

                    # Convert to arrays for easier manipulation
                    x_data = np.array(x_data)
                    y_data = np.array(y_data)
                    z = np.array(z)

                    # Filter horizon points that match the x- and y-coordinates of the seismic inline
                    mask = np.isin(x_data, x_coords) & np.isin(y_data, y_coords)

                    filtered_x = x_data[mask]
                    filtered_y = y_data[mask]
                    filtered_z = z[mask]

                    # Calculate distance along the line for the horizons
                    horizon_distance = np.sqrt((filtered_x - x_coords.min())**2 + (filtered_y - y_coords.min())**2)

                    # Plot only filtered horizon values along the distance line
                    ax.plot(horizon_distance, filtered_z, label=f"{horizon_name} Horizon")

                else:
                    print(f"Unexpected number of data arrays returned")

            elif hasattr(result, 'data_vars'):
                x_data = result['x'].values
                y_data = result['y'].values
                z = result['z'].values

                x_data = np.array(x_data)
                y_data = np.array(y_data)
                z = np.array(z)

                # Filter horizon points to match x- and y-coordinates of the seismic inline
                mask = np.isin(x_data, x_coords) & np.isin(y_data, y_coords)
                
                filtered_x = x_data[mask]
                filtered_y = y_data[mask]
                filtered_z = z[mask]

                # Calculate distance along the line for the horizons
                horizon_distance = np.sqrt((filtered_x - x_coords.min())**2 + (filtered_y - y_coords.min())**2)

                # Plot only filtered horizon values along the distance line
                ax.plot(horizon_distance, filtered_z, label=f"{horizon_name} Horizon")

        except Exception as e:
            print(f"Error plotting horizon {horizon_name}: {e}")

      ax.legend()
      plt.xlabel('x y')
      plt.ylabel('Z Coordinate (Time/Depth)')
      plt.title(f"Seismic Inline {self.inline_number} with Horizons")
      plt.show()




    def plot_arbitrary_seismic_line_and_horizons(self, start_coords, end_coords,start_coords1, end_coords1,col1,fig):
      """
      Plot a seismic section along an arbitrary line defined by start and end coordinates,
      and overlay existing horizons.
    
      Parameters
      ----------
      start_coords : tuple
        Starting (inline, crossline) coordinates of the line.
      end_coords : tuple
        Ending (inline, crossline) coordinates of the line.
      """
      x_coords, y_coords, z_coords, z_values = self.fetch_seismic_data1()
      print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}, z_coords shape: {z_coords.shape}")
      print(f"z_values shape: {z_values.shape}")
      print(z_coords.max(), z_coords.min())

      x_coords1, y_coords1, z_coords1, z_values1 = self.fetch_seismic_data()
      print(f"x_coords shape: {x_coords1.shape}, y_coords shape: {y_coords1.shape}, z_coords shape: {z_coords1.shape}")
      print(f"z_values shape: {z_values1.shape}")
      print(z_coords1.max(), z_coords1.min())
      print(x_coords1, y_coords1)

 
      if x_coords is not None and y_coords is not None and z_values is not None:
        try:
            
            




            # Define the number of points along the arbitrary line
            num_points = 500

            
            
            inline_line1 = np.linspace(start_coords1[0], end_coords1[0], num_points)
            crossline_line1 = np.linspace(start_coords1[1], end_coords1[1], num_points)



            #print(inline_line,crossline_line)
            
            
            # Combine inline and crossline points to get (x, y) pairs
            line_points = np.column_stack((inline_line1, crossline_line1))

            # Find the nearest points using Euclidean distance
            nearest_x, nearest_y = self.find_nearest_points(x_coords1, y_coords1, line_points)

            #print(x_coords1[nearest_inline_indices],y_coords1[nearest_crossline_indices])

            



            # Generate points along the arbitrary line
            inline_line = np.linspace(start_coords[0], end_coords[0], num_points)
            crossline_line = np.linspace(start_coords[1], end_coords[1], num_points)

            # Find the nearest integer inline and crossline indices
            nearest_inline_indices = np.searchsorted(y_coords, np.rint(inline_line)) - 1
            nearest_crossline_indices = np.searchsorted(x_coords, np.rint(crossline_line)) - 1

            # Clip the indices to ensure they are within bounds
            nearest_inline_indices = np.clip(nearest_inline_indices, 0, len(y_coords) - 1)
            nearest_crossline_indices = np.clip(nearest_crossline_indices, 0, len(x_coords) - 1)

            # Extract seismic values along the nearest inline and crossline points
            seismic_values_along_line = z_values[0, nearest_inline_indices, nearest_crossline_indices, :]

            # Generate a combined axis label showing both inline and crossline numbers
            line_labels = [f"Inl: {int(inl)}, Crl: {int(crl)}"
                           for inl, crl in zip(inline_line, crossline_line)]

            
            

            # Create Plotly figure for seismic section
            #fig = go.Figure()
            fig.add_trace(go.Heatmap(
               z=seismic_values_along_line.T,
               x=np.arange(num_points),
               y=z_coords,
               colorscale='RdGy',
               colorbar=dict(
                 title='Amplitude',
                 thickness=10,  # Adjust the width of the color bar
                 len=0.75,  # Adjust the length of the color bar
                 xanchor='left',  # Align the color bar to the left
                 x=0.97,  # Position the color bar at x=0.85 (right side of the heatmap)
                 yanchor='middle',  # Center the color bar vertically
                 y=0.5  # Position the color bar at y=0.5 (center vertically)
               )
            ),row = 1, col=col1 )

            


            #extent = [0, num_points - 1, z_coords.max(), z_coords.min()]
            #plt.figure(figsize=(12, 8))
            #ax = plt.gca()
            #plt.imshow(seismic_values_along_line.T, cmap='RdGy', aspect='auto', extent=extent)
            #plt.colorbar(label='Amplitude')
            #plt.title(f'Arbitrary Seismic Line from {start_coords} to {end_coords}')
            #plt.xlabel('Inline and Crossline Numbers')
            #plt.ylabel('Time/Depth (TWT or Depth)')

            # Set x-ticks to show inline and crossline numbers
            #num_ticks = 10  # Choose how many tick marks you want to show
            #tick_positions = np.linspace(0, num_points - 1, num_ticks).astype(int)
            #plt.xticks(tick_positions, [line_labels[i] for i in tick_positions], rotation=45)
            
            #plt.show()

            


            #ax2 = ax.twiny()
        except Exception as e:
            print(f"Error plotting seismic inline: {e}")
            return


        #plt.show()

        #ax2 = ax.twiny()
        #ax3 = ax.twinx()

        # Overlay horizons, filtering by x- and y-coordinates
        for horizon_name, horizon in self.horizons:
          print(horizon_name)
          try:
            result = horizon.getdata()

            if isinstance(result, tuple):
                data, info = result
                if len(data) == 1:
                    z = data[0]  # z-values (horizon depth)
                    x_data = info['x']  # x-coordinates of the horizon
                    y_data = info['y']  # y-coordinates (inline/crossline)

                    # Convert to arrays for easier manipulation
                    x_data = np.array(x_data)
                    y_data = np.array(y_data)
                    z = np.array(z)

                    # Filter horizon points that match the x- and y-coordinates of the seismic inline
                    mask = np.isin(x_data, nearest_x) & np.isin(y_data, nearest_y)

                    filtered_x = x_data[mask]
                    filtered_y = y_data[mask]
                    filtered_z = z[mask]

                    # Calculate distance along the line for the horizons
                    horizon_distance = np.sqrt((filtered_x - nearest_x.min())**2 + (filtered_y - nearest_y.min())**2)

                    horizon_distance1 = 0 + (horizon_distance - horizon_distance.min()) / (horizon_distance.max() - horizon_distance.min()) * (num_points - 0)


                    # Plot only filtered horizon values along the distance line
                    #ax2.plot(horizon_distance, filtered_z, label=f"{horizon_name} Horizon")
                    # Plot horizon data
                    fig.add_trace(go.Scatter(
                       x=horizon_distance,
                       y=filtered_z,
                       mode='lines',
                       name=f'{horizon_name} Horizon'
                    ),row = 1, col=col1)

                    # Subtract an offset based on the length of the horizon name
                    #offset = 5000   # Adjust multiplier to control spacing
                    #ax2.text(horizon_distance.max() - offset, filtered_z.mean(), horizon_name.lower(), va='center', ha='left',fontsize=8)


                else:
                    print(f"Unexpected number of data arrays returned")

            elif hasattr(result, 'data_vars'):
                x_data = result['x'].values
                y_data = result['y'].values
                z = result['z'].values

                x_data = np.array(x_data)
                y_data = np.array(y_data)
                z = np.array(z)


                



                # Filter horizon points to match x- and y-coordinates of the seismic inline
                mask = np.isin(x_data, nearest_x) & np.isin(y_data, nearest_y)
                #print(x_data)
                #print(mask)
                
                filtered_x = x_data[mask]
                filtered_y = y_data[mask]
                filtered_z = z[mask]
                #print(filtered_x,filtered_y)
                # Calculate distance along the line for the horizons
                horizon_distance = np.sqrt((filtered_x - filtered_x.min())**2 + (filtered_y - filtered_y.min())**2)
                #print(horizon_distance.max())
                # Plot only filtered horizon values along the distance line
                #ax2.plot(horizon_distance, filtered_z, label=f"{horizon_name} Horizon")
                # Plot horizon data
                
                horizon_distance1 = 0 + (horizon_distance - horizon_distance.min()) / (horizon_distance.max() - horizon_distance.min()) * (num_points - 0)



                fig.add_trace(go.Scatter(
                       x=horizon_distance1,
                       y=filtered_z,
                       mode='lines',
                       name=f'{horizon_name} Horizon'
                ),row = 1, col=col1 )


                # Subtract an offset based on the length of the horizon name
                #offset = 5000   # Adjust multiplier to control spacing
                #ax2.text(horizon_distance.max() - offset, filtered_z.mean(), horizon_name.lower(), va='center', ha='left',fontsize=8)


          except Exception as e:
            print(f"Error plotting horizon {horizon_name}: {e}")

        # Set limits for ax2 based on horizon distance

        


        #if 'horizon_distance' in locals() and len(horizon_distance) > 0:
            #ax2.set_xlim([horizon_distance.min(), horizon_distance.max()])
        # Update layout
        #fig.update_layout(
            #title="Arbitrary Seismic Line with Horizons",
            #xaxis_title="Inline & Crossline",
            #yaxis_title="Time/Depth",
            #yaxis_autorange="reversed",
            # x-axis for the scatter plot (chart 2)
            
            


        #)

        # Save figure as HTML and log the success
        #fig.write_html(self.html_file)
        #pio.write_html(fig, file=self.html_file, auto_open=False, include_plotlyjs='cdn')

        self.log(f"Seismic line plot saved to {self.html_file}")


        #ax2.legend()
      #plt.xlabel('x y')
      #plt.ylabel('Z Coordinate (Time/Depth)')
      #plt.title(f"Seismic Inline {self.inline_number} with Horizons")
        #plt.show()



    def log(self, message):
        if hasattr(self, 'log_text_edit'):
            self.log_text_edit.append(message)
        else:
            print(message)  # Fallback if log_text_edit is not initialized

survey = Survey('D2')
wells = Well.names(survey)
print(wells)
horizon_names = Horizon3D.names(survey)
inline_number = 797
time_slice = 1250  # Example time slice
            

#plotter = SeismicHorizonPlotter(survey, 'D2', horizon_names, inline_number, time_slice)

# Run the application
#app = QApplication(sys.argv)
#window = MainWindow()
#window.show()
#sys.exit(app.exec_())