from netCDF4 import Dataset
from collections import namedtuple
import random
import numpy as np
import os
import json
from src.logger_setup import get_logger


"""
Python generator delivering data for tensorflow dataset.
"""

Scenario = namedtuple('Scenario', ['eta', 'flow_depth', 'scenario'])


class DataReader:

    def __init__(self, 
                 rundir,
                 scenarios_file, 
                 datadir, 
                 pois = None, 
                 shuffle_on_load=False,
                 target=True,
                 reload=False):
        
        self.rundir = rundir
        self.scenarios_file = scenarios_file
        self.pois = pois if pois is not None else slice(None)
        self.datadir = datadir
        self.topofile = os.path.join(self.rundir, "topography.grd")
        self.topomask_file = os.path.join(self.rundir, "topomask.npy")
        self.grid_info_file = os.path.join(self.rundir, "grid_info.json")
        self.shuffle_on_load = shuffle_on_load
        self.reload = reload
        self.target = target
        self.lines = None
        
        self.logger = get_logger(__name__)

        with Dataset(self.topofile, 'r') as ds:
            self.topography = ds.variables["z"][:,:]
        
        self.topomask = None
        if os.path.isfile(self.topomask_file):
            self.load_mask()

    def load(self):
        with open(self.scenarios_file, 'r') as file:
            self.lines = file.readlines()
            if self.shuffle_on_load:
                random.shuffle(self.lines)

    def generator(self):
        self.load()
        while self.lines:
            line = self.lines.pop()
            
            # Reload if no more lines and reload is true
            if not self.lines and self.reload:
                self.load()
            
            flow_depth, eta, deformed_topography = self.get_sample(line.strip())
            scenario = Scenario(eta=eta, flow_depth=flow_depth, scenario=line.strip())
            yield scenario

    def get_sample(self, scenario):
        filename_ts, filename_CT = self.get_filenames(scenario)
        # Initialize the eta array with the proper shape and type
        with Dataset(filename_ts) as ds:
            # eta = ds.variables["eta"][:, self.pois]  
            eta = ds.variables["eta"][:481, self.pois] #Hardcoded for first 4 hours even if we have more duration of input
        
        if self.target:
            # Initialize flow_depth and deformed_topography
            flow_depth = np.zeros(self.topography.shape)
            
            # Assign values to flow depth.
            with Dataset(filename_CT) as ds:
                max_height = ds.variables["max_height"]
                max_height.set_auto_maskandscale(True)
                deformation = ds.variables["deformation"]
                deformation.set_auto_maskandscale(True)
                deformed_topography = self.topography - deformation[:,:].data

                # Create a mask and calculate flow_depth
                mask = np.logical_and(self.topography > 0, ~max_height[:,:].mask, max_height[:,:] > deformed_topography)
                flow_depth[mask] = (max_height[:,:].data - deformed_topography)[mask]
                
                if isinstance(self.topomask, np.ndarray):
                    flow_depth = flow_depth[self.topomask]
        else:
            # Return empty arrays if not loaded.
            flow_depth, deformed_topography = np.zeros(0), np.zeros(0)
        
        return flow_depth, eta.T, deformed_topography
    
    def get_filenames(self, scenario):
        filename_ts = os.path.join(self.datadir, f"{scenario}_ts.nc")
        filename_CT = os.path.join(self.datadir, f"{scenario}_CT_10m.nc")
        return filename_ts, filename_CT
    
    def store_grid_info(self):
        """
        Store information on the output grid based on first file in the scenarios_file.
        """
        with open(self.scenarios_file, 'r') as file:
            _, input_file = self.get_filenames(file.readline().strip())
        
        #_, input_file = self.get_filenames(self.lines[0].strip())
        
        with Dataset(input_file, "r") as src:
            grid_info = {
                "dimensions": {dim: len(src.dimensions[dim]) for dim in src.dimensions},
                "variables": {}
            }

            # Store attributes and data of relevant grid variables
            grid_vars = ["grid_lat", "grid_lon", "lat", "lon"]
            for var in grid_vars:
                if var in src.variables:
                    grid_info["variables"][var] = {
                        "dimensions": src.variables[var].dimensions,
                        "datatype": str(src.variables[var].datatype),
                        "attributes": {attr: src.variables[var].getncattr(attr) for attr in src.variables[var].ncattrs()},
                        "data": src.variables[var][:].tolist()
                    }

        # Save grid information as a JSON file# Save grid information as a JSON file
        with open(self.grid_info_file, "w") as f:
            json.dump(grid_info, f)

        self.logger.info(f"Grid information saved to {self.grid_info_file}")

    def load_mask(self):
        # Ensure the file has a .npy extension
        if not self.topomask_file.endswith(".npy"):
            raise ValueError(f"Invalid file extension: {self.topomask_file}. Expected a .npy file.")
        try:
            if os.path.isfile(self.topomask_file):
                # Try to load the file
                self.topomask = np.load(self.topomask_file)
            else:
                raise FileNotFoundError  # Explicitly trigger mask creation
        except (FileNotFoundError, OSError, ValueError):  # Catch file errors
            self.logger.warning(f"{self.topomask_file} not found, not a .npy file, or cannot be loaded. Creating a new mask.")
            self.create_mask()
            
    def save_mask(self):
        try:
            np.save(self.topomask_file, self.topomask)
        except OSError as e:
            self.logger.info(f"Error: Could not save {self.topomask_file}. Reason: {e}")
    
    def create_mask(self):
        """
        Loop through all scenarios and return a mask containing all the pixels which has been inundated.
        Returns None if no scenarios are returned by the generator.
        """
        mask = None
        gen = self.generator()
        scenario = next(gen, None)
        if scenario is not None:
            mask = scenario.flow_depth>0.
            for index, scenario in enumerate(gen):
                mask = np.logical_or(mask, scenario.flow_depth > 0.)
                if index % 50 == 0:
                    self.logger.info(f"Processed {index} scenarios.")
        
        self.topomask = mask
        
        # Save mask to file
        try:
            np.save(self.topomask_file, self.topomask)
        except OSError as e:
            self.logger.info(f"Error: Could not save {self.topomask_file}. Reason: {e}")