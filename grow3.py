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

# Data:
#   git clone https://github.com/CSSEGISandData/COVID-19.git

cat20_colours = [
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
for country in countries:
    dfc=df[ (df["Country/Region"]==country) & (df["Province/State"].isnull())  ]
    print( dfc )
    #print( dfc.columns )
    cols = dfc.columns[-last_n:]
    try:
        data = dfc.iloc[0,4:]
        data = data[data>0] 
        print( "Values", dfc.iloc[0,4:].values )
        print( "data", data )
        print( dfc.iloc[0,4:].first_valid_index() )
    except IndexError:
        continue
    if not args.absolute:
        try:
            pop = population[country]
        except KeyError:
            pop = 1
    else:
        pop = 1
    graphs[country] = dfc.iloc[0,-last_n:].values / pop
    #graphs[country] = data / pop
    ymax = max( ymax, max(graphs[country]) )

key0   = list(graphs.keys())[0] # need access to one of'm
xcount = list(range( 0, len(graphs[key0]) ))
print( xcount )
first_date = cols[0]
first_date_dt = datetime.strptime(first_date,'%m/%d/%y')
print( first_date, first_date_dt )
xlabels = [first_date_dt + pd.Timedelta('1 day')*d for d in xcount ]
print( xlabels )
           
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

#print( graphs )

for i, g in enumerate(graphs):
    print( g )
    ax.plot( xlabels, graphs[g], label=g, c=cat20_colours[i] )
    #ax.plot( graphs[g].index, graphs[g], label=g, c=cat20_colours[i] )
    #ax.scatter( xlabels, graphs[g], c=cat20_colours[i], alpha=0.4 )
fig.autofmt_xdate()
#ax.set_yscale("log")
ax.set_title( title_str )
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
days = mdates.DayLocator()
ax.xaxis.set_major_locator(days)

#ax.legend(fontsize='x-small', labelspacing=0.2, frameon=True)
ax.legend(labelspacing=0.2, frameon=True)

pngfile="covid.png"
fig.savefig(pngfile, dpi=300)

#

import scipy.optimize as sio

def f(x, A, B):
    if args.function[0] == "e":
        return A + np.exp(B*x)
    else:
        return A * np.power(x, B) 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

x1 = range(0, len(xcount) + 1 ) # One extra day
x1labels = [first_date_dt + pd.Timedelta('1 day')*d for d in x1 ]
x1 = np.array(x1)

for i, g in enumerate(graphs):
    print( g )
    x = np.array(xcount)
    y = np.array(graphs[g])
    print( x, y[0] )
    coeffs, coeffs_cov = sio.curve_fit(f, x, y, p0=(0, 1), bounds=(0, +np.inf) )
    print( coeffs )
    print( coeffs_cov )
    # original data
    ax.scatter(xlabels, y, c=cat20_colours[i], label=g)
    #ax.plot(xlabels, y, c=cat20_colours[i], linewidth=1, label=g) # labels/range from above
    #
    pred_y = f(x1, *coeffs)
    print( y )
    print( pred_y )
    cf = ["{:.2f}".format(cf) for cf in coeffs]
    # interpolated data
    ax.plot(x1labels, (f(x1, *coeffs)), c=cat20_colours[i], linewidth=2, label=cf) #extrapolated labels/range
    #ax.set_yscale("log")
fig.autofmt_xdate()
ax.set_title( title_str )
ax.set_xlabel( "Data from https://github.com/CSSEGISandData/COVID-19" )
ax.grid(linestyle='-', alpha=0.5) #, axis="y"
days = mdates.DayLocator()
ax.xaxis.set_major_locator(days)
ax.legend(labelspacing=0.2, frameon=True)
plt.tight_layout()
pngfile="covid_pred.png"
fig.savefig(pngfile, dpi=300)
plt.show()
