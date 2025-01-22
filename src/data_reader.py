from netCDF4 import Dataset
from collections import namedtuple
import random
from collections.abc import Generator
import numpy as np
import tensorflow as tf
import os


"""
Python generator delivering data for tensorflow dataset.
"""

Scenario = namedtuple('Scenario', ['eta', 'flow_depth', 'scenario'])


class DataReader:

    def __init__(self, 
                 scenarios_file, 
                 pois, 
                 datadir, 
                 topofile,
                 topo_mask_file=None, 
                 shuffle_on_load=False,
                 target=True,
                 reload=False):
        
        self.scenarios_file = scenarios_file
        self.pois = pois
        self.datadir = datadir
        self.topofile = topofile
        self.topo_mask_file = topo_mask_file
        self.shuffle_on_load = shuffle_on_load
        self.reload = reload
        self.target = target
        self.lines = None

        with Dataset(self.topofile, 'r') as ds:
            self.topography = ds.variables["z"][:,:]
        
        if self.topo_mask_file:        
            with open(self.topo_mask_file, 'r') as file:
                lines = file.readlines()

            # Convert each line to a boolean (True for "true", False for "false")
            boolean_array = np.array([
                [element.strip().lower() == 'true' for element in line.split()]
                for line in lines
            ], dtype=bool)
            
            self.topo_mask = boolean_array.T
        
        self.load()

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
        filename_ts = os.path.join(self.datadir, f"{scenario}_ts.nc")
        filename_CT = os.path.join(self.datadir, f"{scenario}_CT_10m.nc")
        
        # Initialize the eta array with the proper shape and type
        with Dataset(filename_ts) as ds:
            eta = ds.variables["eta"][:, self.pois]
        
        if self.target:
            # Initialize flow_depth and deformed_topography
            flow_depth = np.zeros(self.topography.shape)
            
            # Assign values to flow depth.
            with Dataset(filename_CT) as ds:
                max_height = ds.variables["max_height"][:,:]
                deformation = ds.variables["deformation"][:,:]
                deformed_topography = self.topography - deformation

                # Create a mask and calculate flow_depth
                mask = (self.topography > 0) & (max_height != np.ma.masked) & (max_height > deformed_topography)
                flow_depth[mask] = (max_height - deformed_topography)[mask]
                
                if self.topo_mask_file:
                    flow_depth = flow_depth[self.topo_mask]
        else:
            # Return empty arrays if not loaded.
            flow_depth, deformed_topography = np.zeros(0), np.zeros(0)
        
        return flow_depth, eta.T, deformed_topography

