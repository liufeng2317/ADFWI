API
=============

1. **ADFWI.propagator**  
   
   Contains methods for simulating wave propagation, such as solving wave equations and performing forward and inverse modeling of seismic wavefields.

2. **ADFWI.model**  
   
   Provides tools for defining and manipulating velocity models and other subsurface parameters used in the full-waveform inversion (FWI) process.

3. **ADFWI.fwi**  
   
   Focuses on the full-waveform inversion techniques, including misfit function definitions, optimization strategies, and regularization methods. This is central to seismic data inversion.

4. **ADFWI.survey**  
   
   Implements utilities for handling survey data, including survey design, data preprocessing, and integration with the FWI process.

5. **ADFWI.dip**  
   
   Includes algorithms for deep image prior (DIP-related) processing.

6. **ADFWI.utils**  
   
   A collection of utility functions for general-purpose tasks, such as data I/O, mathematical operations, and visualization utilities to support the core functionalities of ADFWI.

7. **ADFWI.view**  
   
   Provides visualization tools to plot and analyze the results of simulations, inversions, and other model outputs in a user-friendly way.


.. toctree::
   :maxdepth: 2

   ADFWI.propagator
   ADFWI.model
   ADFWI.fwi
   ADFWI.survey
   ADFWI.dip
   ADFWI.utils
   ADFWI.view