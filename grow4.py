import re
import sys, os
import argparse
from datetime import datetime
from datetime import timedelta
#https://stackoverflow.com/questions/35066588/is-there-a-simple-way-to-increment-a-datetime-object-one-month-in-python
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib as mpl
mpl.use("Qt5Agg") #TkAgg crashes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter, WeekdayLocator
import matplotlib.ticker as plticker
import matplotlib.dates as mdates

# The data for this script can be found here:
#   git clone https://github.com/CSSEGISandData/COVID-19.git

cat20_colours = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    #https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    #"#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    #"#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"                 
]

parser = argparse.ArgumentParser()
parser.add_argument( '-a', "--absolute", action='store_true', default=False, help='Do not normalise' )
parser.add_argument( '-A', "--all", action='store_true', default=False, help='All countries with populations' )
parser.add_argument( '-c', '--countries', type=str, default="Sweden,Netherlands", help='Countries')
parser.add_argument( '-f', '--function', type=str, default="power", help='Which function to fit (power, exponential)')
parser.add_argument( '-g', '--graph', type=str, default="confirmed", help='Which data set (confirmed, deaths)')
parser.add_argument( '-l', "--last_n", type=int, default=28, help='Last n days' )
parser.add_argument( '-m', "--minimum", type=int, default=0, help='Use only data > minimum' )
parser.add_argument( '-p', "--predict", type=int, default=1, help='Number of days to predict' ) 
args = parser.parse_args()

#fn = "./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
#fn = "./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
if args.graph[0] == "c":
    fn="./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
else:
    fn="./COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

'''
[pberck@margretetorp Python]$ grep Netherlands time_series_covid19_deaths_global.csv
Province/State,Country/Region,Lat,Long,1/22/20, ....
Aruba,Netherlands,12.5186,-70.0358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Curacao,Netherlands,12.1696,-68.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1
Sint Maarten,Netherlands,18.0425,-63.0548,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
,Netherlands,52.1326,5.2913,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,3,3,4,5,5,10,12,20,24,43,58,76,106,136,179,213,276,356
'''

df = pd.read_csv( fn, sep="," )
print( df )

graphs     = {}
last_n     = args.last_n
population = {}
#https://www.worldometers.info/world-population/norway-population/
population["Sweden"]      =  10.080176 # in 1000,000s (seems common)
population["Netherlands"] =  17.123469
population["Denmark"]     =   5.786051
population["Norway"]      =   5.408352
population["Germany"]     =  83.703211
population["Belgium"]     =  11.574389
population["Iran"]        =  83.662908
population["Italy"]       =  60.488377
population["Iceland"]     =   0.340596
population["Spain"]       =  46.749597
population["US"]          = 330.488824

#countries = ["Netherlands", "Sweden", "Denmark", "Norway", "Germany", "Belgium", "Iran", "Italy"]
if args.all:
    countries = population.keys()
else:
    countries = args.countries.split(",")

title_str = "Normalised per 1e6 inhabitants"
if args.absolute:
    title_str = "Absolute numbers"
    
ymax = 0
ymin = 0
first_date = {}
last_date  = {}
for country in countries:
    dfc=df[ (df["Country/Region"]==country) & (df["Province/State"].isnull())  ]
    print( dfc )
    try:
        data = dfc.iloc[0,4:]
        data = data[-args.last_n:]
        data = data[data>args.minimum] # What if 0 in the middle? 
        print( "data\n", data )
    except IndexError:
        continue
    if not args.absolute:
        try:
            pop = population[country]
        except KeyError:
            pop = 1
    else:
        pop = 1
    date_index = [datetime.strptime(x, '%m/%d/%y') for x in data.index]
    data.index = date_index
    #print( date_index )
    graphs[country] = data / pop # normalise
    ymax = max( ymax, max(graphs[country]) )
    first_date[country] = date_index[0]
    last_date[country]  = date_index[-1]
    print( first_date, min(first_date.values()), max(last_date.values()) )

xlabels = pd.date_range(start=min(first_date.values()), end=max(last_date.values()) ) #, freq='W-Mon')
print( xlabels )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

#print( graphs )

for i, g in enumerate(graphs):
    print( g )
    ax.plot( graphs[g], label=g, c=cat20_colours[i] )
    #ax.plot( graphs[g].index, graphs[g], label=g, c=cat20_colours[i] )
    ax.scatter( x=graphs[g].index, y=graphs[g].values, c=cat20_colours[i], alpha=0.5 )

fig.autofmt_xdate()
#ax.set_yscale("log")
ax.set_title( title_str )
ax.set_xticklabels([], minor=True)
ax.set_xlabel( "Data from https://github.com/CSSEGISandData/COVID-19" )
plt.tight_layout()
ax.grid(linestyle='-', axis="y", alpha=0.5)
yrange = ymax - ymin
if yrange > 1000:
    yinc   = round(yrange // 10, -2)
    yinc   = yinc
elif yrange > 100:
    yinc   = round(yrange // 10, -1)
    yinc   = yinc
elif yrange > 10:
    yinc   = yrange // 10
else:
    yinc = 1
print( yrange, yinc )
loc = plticker.MultipleLocator(base=yinc) # this locator puts ticks at regular intervals
ax.yaxis.set_major_locator(loc)
if len( xlabels ) < 22:
    days = mdates.DayLocator()
else:
    days = mdates.WeekdayLocator(byweekday=(0))
ax.xaxis.set_major_locator(days)
ax.xaxis.set_minor_locator(mdates.DayLocator())

#ax.legend(fontsize='x-small', labelspacing=0.2, frameon=True)
ax.legend(labelspacing=0.2, frameon=True)

pngfile="covid.png"
fig.savefig(pngfile, dpi=300)

# Fit and extrapolate

import scipy.optimize as sio

def f(x, A, B):
    if args.function[0] == "e":
        # f(x) = A * B^x
        return A * np.exp(B*x)
    else:
        # f(x) = A * x^B
        return A * np.power(x, B) 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

x1 = range(0, len(xlabels) + args.predict ) # 'p' extra days
x1labels = pd.date_range(start=xlabels[0], end=xlabels[-1] + pd.Timedelta(str(args.predict)+' day') ) 
x1 = np.array(x1)

for i, g in enumerate(graphs):
    print( g )
    x  = np.array( graphs[g].index ) # original x-index
    x1 = pd.date_range(start=graphs[g].index[0], end=graphs[g].index[-1] + pd.Timedelta(str(args.predict)+'day') ) 
    y  = np.array( graphs[g].values ) # original y-values
    print( x, y[0] )
    xr = np.array( range( 0, len(x) ) ) # integer range over days
    coeffs, coeffs_cov = sio.curve_fit(f, xr, y, p0=(0, 1), bounds=(0, +np.inf), maxfev=1000 )
    print( coeffs )
    print( coeffs_cov )
    # original data
    ax.scatter( x=graphs[g].index, y=graphs[g].values, c=cat20_colours[i], alpha=0.5, label=g )
    #
    xr1 = range( 0, len(x)+args.predict ) # integer r ange for days extra
    pred_y = f(xr1, *coeffs)   # predict
    print( y )
    print( pred_y )
    cf = ["{:.2f}".format(cf) for cf in coeffs]
    # plot interpolated data
    ax.plot( x1, pred_y, c=cat20_colours[i], linewidth=2, label=cf) #extrapolated labels/range
    #ax.stem( [x1[-1]], [pred_y[-1]] )
    #ax.hlines( pred_y[-1], x1[0], x1[-1] )
    ax.annotate(str(int(pred_y[-1])), (x1[-1], pred_y[-1]))
fig.autofmt_xdate()
ax.set_title( title_str )
ax.set_xlabel( "Data from https://github.com/CSSEGISandData/COVID-19" )
ax.grid(linestyle='-', alpha=0.5) #, axis="y"
if len( x1labels ) < 22:
    days = mdates.DayLocator()
else:
    days = mdates.WeekdayLocator(byweekday=(0))
ax.xaxis.set_major_locator(days)
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.legend(labelspacing=0.2, frameon=True)
plt.tight_layout()
pngfile="covid_pred.png"
fig.savefig(pngfile, dpi=300)
plt.show()
