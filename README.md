# UNgroundwater

**UNgroundwater** is a Python module designed to facilitate the analysis of groundwater storage changes using data from the **GRACE (Gravity Recovery and Climate Experiment)** and **GLDAS (Global Land Data Assimilation System)** datasets. This tool provides essential functionalities for researchers and hydrologists to process, visualize, and analyze groundwater data efficiently.

## Features

The repository offers the following core features:

- **NetCDF Data Handling:**  
  - Reading, processing, and extracting relevant data from GRACE and GLDAS NetCDF files.
  - Efficient handling of large datasets with optimized memory usage.

- **Masking Capabilities:**  
  - Application of geographic and administrative masks to filter data for specific regions.
  - Support for custom shapefile inputs to refine analysis.

- **Groundwater Estimation:**  
  - Computation of groundwater storage anomalies based on GRACE-derived data.
  - Integration with GLDAS for enhanced accuracy and correction.

- **Visualization Tools:**  
  - Plotting spatial and temporal variations in groundwater storage.
  - Interactive and static visualizations for effective data interpretation.

## Installation

To install the module, clone the repository and install the dependencies:

```bash
git clone https://github.com/Peter24K2G/UNgroundwater.git
cd UNgroundwater
pip install -r requirements.txt
