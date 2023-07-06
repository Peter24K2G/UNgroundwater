# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:53:56 2023

@author: Peter
"""
import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import calendar


def available_data(ds1,ds2):

    dates1 = ds1.time.values
    dates2 = ds2.time.values
    
    fig, ax = plt.subplots(figsize=(12,1.5),layout='constrained')
    ax.set(title='Data measurements in time for each dataset')
    
    ax.vlines(dates1,0.05,0.5,color='orange',label='GLDAS',linewidth=1.5)
    ax.vlines(dates2,-0.05,-0.5,color='lightblue',label='GRACE',linewidth=2)
    
    ax.legend(loc='right',bbox_to_anchor=(1.1,0.5))
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    ax.spines[['right','top','left']].set_visible(False)
    ax.yaxis.set_visible(False)
    
    plt.show()
    
def mean_plot(ds,var):

    meanv = np.mean(ds.mean(dim=['lon','lat'])[var])
    

    fig, ax = plt.subplots(figsize=(10,4))
    mean = ds.mean(dim=['lon','lat'])[var].plot
    mean.line(marker='o',color='lightpink')
    meanl = mean(ax=ax)

    mina = ds.min(dim=['lon','lat'])[var].plot
    mina.line(color='red') 
    minl = mina(ax=ax)

    maxa = ds.max(dim=['lon','lat'])[var].plot
    maxa.line(color='green') 
    maxl = maxa(ax=ax)

    ax.fill_between(maxl,minl,color='cyan')
    
    ax.set_title(ds[var].attrs['standard_name'],
            fontdict={
                'fontweight': 'bold'
                })
    
    ax.axhline(np.mean(ds.mean(dim=['lon','lat'])[var]),color='blue',linestyle='--')
    
    ax.set_ylabel(f"{ds[var].attrs['standard_name']} [{ds[var].attrs['units']}]")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True)
    
    plt.show()

    



def mix_plot(amz,car,mgdcau,ori,pcf):

    
    dds = ds.diff(dim='time')
    
    fig, ax = plt.subplots(figsize=(15,3))
    
    mean = ds.mean(dim=['lon','lat'])[var].plot
    dmean = dds.mean(dim=['lon','lat'])[var].plot
    # max = ds.max(dim=['lon','lat'])[var].plot
    
    mean.line(marker='o',color='lightblue',label='')
    dmean.line(marker='o',color='lightpink')
    # max.line(marker='o',color='aquamarine')

    mean(ax=ax)
    dmean(ax=ax)
    # max(ax=ax)
    
    ax.set_title(ds[var].attrs['long_name'],
           fontdict={
               'fontweight': 'bold'
               })
    
    ax.set_ylabel(f"{ds[var].attrs['long_name']} [{ds[var].attrs['units']}]")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True)
    
    plt.show()


def years_plot(dataset):

    sns.set_theme(style='darkgrid')

    years = []
    months = []

    for date in dataset['time'].values:
        date = date.astype('datetime64[s]').astype(datetime)
        years.append(date.year)
        months.append(calendar.month_abbr[date.month])

    df = pd.DataFrame(
            {
                'year'  :   years,
                'month' :  months,
                'mean':  dataset.mean(dim=['lon','lat'])['Groundwater_Estimation'].values
            }
        )

    g = sns.relplot(
        data = df,
        x = 'month', y = 'mean', col = 'year', #hue = 'year',palette = 'crest',
        kind = 'line' , linewidth=3, zorder=3,
        col_wrap=4, height = 2.5, aspect=1.5, legend = False
        )

    for year, ax in g.axes_dict.items():
       ax.text(.8,.15, year, transform = ax.transAxes, fontweight='bold')

       sns.lineplot(
           data = df, x = 'month', y = 'mean', units = 'year',
           estimator=None, color='0.7', linewidth=0.5, ax=ax
           )
    ax.set_xticks(ax.get_xticks()[::2])

    g.set_titles('')
    g.set_axis_labels('','Grounwater Anomalies \n [cm]')
    g.tight_layout()

def crop_ds(dataset):
    ds  = dataset
    # ds1  = xr.Dataset()
    # ds2  = xr.Dataset()
    for var in dataset:
        ds[var] = dataset[var][90:121]
        # ds1[var] = dataset[var][107:154]
        # ds2[var] = dataset[var][167:214]
        #ds[var] = dataset[var][7:215]

    # return xr.concat([ds1,ds2],dim='time')
    return ds

def years_bounds_plot(dataset,bounds):

    sns.set_theme(style='darkgrid')

    years = []
    months = []

    for date in dataset['time'].values:
        date = date.astype('datetime64[s]').astype(datetime)
        years.append(date.year)
        months.append(calendar.month_abbr[date.month])

    df = pd.DataFrame(
            {
                'year'  :   years,
                'month' :  months,
                'mean':  dataset.mean(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'min':  dataset.min(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'max':  dataset.max(dim=['lon','lat'])['Groundwater_Estimation'].values
            }
        )
    
    df = df.query(bounds[0]+'<= year <='+bounds[1])

    g = sns.relplot(data = df,
        x = 'month', y = 'mean', col = 'year',kind = 'line',# hue = None,  palette = 'crest',
        linewidth=3, zorder=3, col_wrap=4, height = 2.5, aspect=1.5, legend = True
        )

    for year, ax in g.axes_dict.items():
       ax.text(.8,.15, year, transform = ax.transAxes, fontweight='bold')

       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'max', units = 'year',
           estimator=None, color='green', linewidth=0.5, ax=ax,label='max'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'min', units = 'year',
           estimator=None, color='red', linewidth=0.5, ax=ax,label='min'
           )

    ax.set_xticks(ax.get_xticks()[::2])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    g.set_titles('')
    g.set_axis_labels('','Groundwater Anomalies \n [cm]')
    g.tight_layout()



def years_regions(amz,car,Col,mgdcau,ori,pcf,bounds):

    #sns.set_theme(style='ticks')
    sns.set_style('ticks',{'axes.grid':True,'grid.color':'darkgray'})

    years = []
    months = []
    dates = []

    for date in amz['time'].values:
        date = date.astype('datetime64[s]').astype(datetime)
        years.append(date.year)
        months.append(calendar.month_abbr[date.month])
        dates.append(f"{date.year} {calendar.month_abbr[date.month]}")
        

    df = pd.DataFrame(
            {
                'time' : dates,
                'year'  :   years,
                'month' :  months,
                'amz':  amz.mean(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'car':  car.min(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'Col':  Col.mean(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'mgdcau':  mgdcau.mean(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'pcf':  pcf.mean(dim=['lon','lat'])['Groundwater_Estimation'].values,
                'ori':  ori.mean(dim=['lon','lat'])['Groundwater_Estimation'].values
            }
        )

    df = df.query(bounds[0]+'<= year <='+bounds[1])
    return df
    g = sns.relplot(
        data = df,
        x = 'month', y = 'Col', col = 'year',kind = 'line',# hue = None,  palette = 'crest',
        linewidth=2, zorder=3, col_wrap=3, height = 5, aspect=1.5,legend = True,label='Col',
        linestyle='--',color='gray'
        )
    
    for year, ax in g.axes_dict.items():
       ax.text(.5,.95, year, transform = ax.transAxes, fontweight='bold',fontsize = 'xx-large')
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'Amz', units = 'year',
           estimator=None, color='green', linewidth=1, ax=ax,label='amz'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'car', units = 'year',
           estimator=None, color='red', linewidth=1, ax=ax,label='car'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'ori', units = 'year',
           estimator=None, color='orange', linewidth=1, ax=ax,label='ori'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'mgdcau', units = 'year',
           estimator=None, color='darkblue', linewidth=1, ax=ax,label='mgdcau'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'pcf', units = 'year',
           estimator=None, color='purple', linewidth=1, ax=ax,label='pcf'
           )
       ax.legend()
       bbox_to_anchor=(0.05, 1.05)
       ax.legend(loc=bbox_to_anchor, bbox_transform=ax.transAxes,ncols=6)
     
    ax.set_xticks(ax.get_xticks()[::2])
    
    g.set_titles('')
    g.set_axis_labels('','Grounwater Anomalies \n [cm]')
    #g.set_axis_labels('','Precipitation [mm]')
    g.tight_layout()
    

def years_regions_vs(Preg,GWreg):

    sns.set_theme(style='darkgrid')

    years = []
    months = []

    for date in Preg['time'].values:
        date = date.astype('datetime64[s]').astype(datetime)
        years.append(date.year)
        months.append(calendar.month_abbr[date.month])

    df = pd.DataFrame(
            {
                'year'  :   years,
                'month' :  months,
                'prec':  Preg.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].values,
                'GWe':  GWreg.mean(dim=['lon','lat'])['Groundwater_Estimation'].values
            }
        )

    g = sns.relplot(
        data = df,
        x = 'month', y = 'GWe', col = 'year',kind = 'line',# hue = None,  palette = 'crest',
        linewidth=2, zorder=3, col_wrap=4, height = 3.5, aspect=1.7,legend = True,label='Ground Water',
        linestyle='-',color='blue'
        )
    for year, ax in g.axes_dict.items():
       ax.text(.5,.85, year, transform = ax.transAxes, fontweight='bold')
       ax2 = ax.twinx() 
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'prec', units = 'year',
           estimator=None, color='green', linewidth=1, ax=ax2,label='Precipitation'
           )

    ax.set_xticks(ax.get_xticks()[::2])

    g.set_titles('')
    #g.set_axis_labels('','Grounwater Anomalies \n [cm]')
    g.set_axis_labels('','')
    g.tight_layout()
def Rain_years_regions(amz,car,Col,mgdcau,ori,pcf):

    sns.set_theme(style='darkgrid')

    years = []
    months = []

    for date in amz['time'].values:
        date = date.astype('datetime64[s]').astype(datetime)
        years.append(date.year)
        months.append(calendar.month_abbr[date.month])

    df = pd.DataFrame(
            {
                'year'  :   years,
                'month' :  months,
                'amz':  amz.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].values,
                'car':  car.min(dim=['lon','lat'])['Rainf_f_tavg_mm'].values,
                'Col':  Col.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].values,
                'mgdcau':  mgdcau.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].values,
                'pcf':  pcf.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].values,
                'ori':  ori.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].values
            }
        )

    g = sns.relplot(
        data = df,
        x = 'month', y = 'Col', col = 'year',kind = 'line',# hue = None,  palette = 'crest',
        linewidth=2, zorder=3, col_wrap=4, height = 3.5, aspect=1.7,legend = True,label='Col',
        linestyle='--',color='gray'
        )
    
    for year, ax in g.axes_dict.items():
       ax.text(.5,.85, year, transform = ax.transAxes, fontweight='bold')

       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'Amz', units = 'year',
           estimator=None, color='green', linewidth=1, ax=ax,label='amz'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'car', units = 'year',
           estimator=None, color='red', linewidth=1, ax=ax,label='car'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'ori', units = 'year',
           estimator=None, color='orange', linewidth=1, ax=ax,label='ori'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'mgdcau', units = 'year',
           estimator=None, color='darkblue', linewidth=1, ax=ax,label='mgdcau'
           )
       sns.lineplot(
           data = df.loc[df['year']==year], x = 'month', y = 'pcf', units = 'year',
           estimator=None, color='purple', linewidth=1, ax=ax,label='pcf'
           )

    ax.set_xticks(ax.get_xticks()[::2])

    g.set_titles('')
    #g.set_axis_labels('','Grounwater Anomalies \n [cm]')
    g.set_axis_labels('','Precipitation [mm]')
    g.tight_layout()


def mean_region(DS,name):
    
    dfmean = DS.mean(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
    dfmin = DS.min(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
    dfmax = DS.max(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
    Concat = pd.concat([dfmean,dfmin,dfmax],ignore_index=False)
    Concat['region'] = name

    df = Concat
    # df = pd.concat(DF,ignore_index=False)
    
    fig,ax = plt.subplots(figsize=(12,4))
    ax.grid(True)
    sns.lineplot(df,ax=ax)
    
    meanv = np.mean(DS.mean(dim=['lon','lat'])['Groundwater_Estimation'])
    maxv = np.max(DS.max(dim=['lon','lat'])['Groundwater_Estimation'])
    minv = np.min(DS.min(dim=['lon','lat'])['Groundwater_Estimation'])
    
    ax.axhline(meanv,color='black',linestyle='--',label=f"Mean: {np.round(meanv.values,1)} [cm]")
    ax.axhline(maxv,color='blue',linestyle='--',label=f"Max: {np.round(maxv.values,1)} [cm]")
    ax.axhline(minv,color='red',linestyle='--',label=f"Min: {np.round(minv.values,1)} [cm]")
        
    ax.set_title(name)
    ax.set_xlabel('Years')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel('Groundwater Anomalies \n [cm]')
    
    ax.legend()
    bbox_to_anchor=(1.05, 0.05)
    ax.legend(loc=bbox_to_anchor, bbox_transform=ax.transAxes)
    

def mean_regions(DS):
    
    regions = ['Amazonas', 'Caribe', 'Colombia' , 'Magdalena - Cauca', 'Orinoco', 'Pacífico']
    # regions = ['Colombia']
    # colors = ['green', 'red', 'orange', 'darkblue', 'purple']
    DF = []
    for ds,region in zip(DS,regions):
        dfmean = ds.mean(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
        dfmin = ds.min(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
        dfmax = ds.max(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
        Concat = pd.concat([dfmean,dfmin,dfmax],ignore_index=False)
        Concat['region'] = region
        # Concat['color'] = color
        DF.append(Concat)
        del dfmean,dfmin,dfmax,Concat

    df = pd.concat(DF,ignore_index=False)
    
    g = sns.relplot(df,x=df.index,col='region',col_wrap=2,height=3.5,aspect=3,legend=None)
    
    palette = sns.color_palette()
    sns.set_style('ticks',{'axes.grid':True})
    
    for region,ax in g.axes_dict.items():
        ax.text(.5,.85, region, transform = ax.transAxes, fontweight='bold')
        sns.lineplot(df.loc[df['region']==region],ax=ax,color=palette[0])
        
    g.set_titles('')
    g.set_axis_labels('Years','Groundwater Anomalies \n [cm]')
    g.tight_layout()
    
def mean_regions_prec(DS1,DS2):
    
    regions = ['Amazonas', 'Caribe','Magdalena - Cauca', 'Orinoco', 'Pacífico','Colombia']
    # colors = ['green', 'red', 'orange', 'darkblue', 'purple']
    DF = []
    for ds1,ds2,region in zip(DS1,DS2,regions):
        dfmean = ds1.mean(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
        # dfmin = ds1.min(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
        # dfmax = ds1.max(dim=['lon','lat'])['Groundwater_Estimation'].to_dataframe()
        dpmean = ds2.mean(dim=['lon','lat'])['Rainf_f_tavg_mm'].to_dataframe()
        # dpmin = ds2.min(dim=['lon','lat'])['Rainf_f_tavg_mm'].to_dataframe()
        # dpmax = ds2.max(dim=['lon','lat'])['Rainf_f_tavg_mm'].to_dataframe()
        # Concat = pd.concat([dfmean,dfmin,dfmax,dpmean,dpmin,dpmax],ignore_index=False)
        Concat = pd.concat([dfmean,dpmean],ignore_index=False)
        Concat['region'] = region
        DF.append(Concat)
        del dfmean,dpmean,Concat
        
    
        
    

    df = pd.concat(DF,ignore_index=False)
    # return df
    df = df.loc[df.index>'2009']
    df = df.loc[df.index<'2012']
    # df = df.loc[df.index>'2014']
    # df = df.loc[df.index<'2018']
    
    g = sns.relplot(df,x=df.index,col='region',col_wrap=2,height=3.5,aspect=1.5,legend=None)
    
    sns.color_palette("rocket")
    sns.set_style('ticks',{'axes.grid':True})
    
    for region,ax in g.axes_dict.items():
        ax1 = ax.twinx()
        ax.text(.5,.85, region, transform = ax.transAxes, fontweight='bold')
        foo1 = df.loc[df['region']==region]
        sns.lineplot(foo1['Groundwater_Estimation'],ax=ax,color='blue',label='Grounwater Anomalies')
        ax.legend(loc='upper left')
        sns.lineplot(foo1['Rainf_f_tavg_mm'],ax=ax1,color='green',label='Precipitation')
        ax1.legend(loc='upper right')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        
    g.set_titles('')
    g.set_axis_labels('Years','Groundwater Anomalies \n [cm]')
    g.tight_layout()


        
    
