# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:44:46 2023

@author: Peter
"""

from setuptools import setup, find_packages

setup(
    name='ungrounwater',
    version='1.0.0',
    author='Pedro Romero',
    description='''
    This package contains modules that ease the visualization of
    multitemporal data, and its statistics focused inside polygons
    ''',
    packages=find_packages(),
    install_requires=['xarray','numpy','shapefile','shapely'])
