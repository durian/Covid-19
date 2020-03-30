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

fn = "COVID-19-geographic-disbtribution-worldwide-2020-03-30.csv"
df = pd.read_csv( fn, sep="," )
#df['DateRep'] = df['DateRep'].apply(lambda x: dt.datetime.strptime(x,'%d/%m/%Y'))
df.loc[:,'dateRep'] = pd.to_datetime(df['dateRep']) #, format='%d/%m/%Y')
print( df )

populations = { }
populations["SE"] = 10.080176 # in 1000,000s 
populations["NL"] = 17.123469
populations["DK"] =  5.786051
populations["NO"] =  5.408352
populations["DE"] = 83.703211
populations["BE"] = 11.574389
populations["IR"] = 83.662908
populations["IT"] = 60.488377
populations["IS"] =  0.340596

dfs = {}

if False:
    c = "EU" # Hardcoded EU
    dfs[c] = df[ df["EU"] == "EU" ]
    #dfs[c] = dfs[c][ dfs[c]["dateRep"] > "2020-02-12" ]
    dfs[c].sort_values( by=['dateRep'], inplace=True )
    #dfs[c].loc[:, 'Sum']     = dfs[c]['NewConfcases'].cumsum()
    dfs[c].loc[:, 'Sum']     = dfs[c]['cases'].cumsum()
    dfs[c].loc[:, 'SumNorm'] = dfs[c]['Sum'] / 1000
    print( dfs[c] )

for c in ["SE", "NL", "NO", "DK"]:
    if not c in populations:
        populations[c] = 1000
    dfs[c] = df[ df["geoId"] == c ]
    #dfs[c] = dfs[c][ dfs[c]["dateRep"] > "2020-02-12" ]
    dfs[c].sort_values( by=['dateRep'], inplace=True )
    #dfs[c].loc[:, 'Sum']     = dfs[c]['NewConfcases'].cumsum()
    dfs[c].loc[:, 'Sum']     = dfs[c]['cases'].cumsum()
    dfs[c].loc[:, 'SumNorm'] = dfs[c]['Sum'] / populations[c]
    print( dfs[c] )
    print( dfs[c]["Sum"].values )

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
for c in dfs:
    print( c )
    #ax.plot( df_SE["dateRep"], df_SE["NewConfcases"], label="SE new" )
    ax.plot( dfs[c]["dateRep"], dfs[c]["SumNorm"], label=c )

ax.xaxis.set_major_formatter( DateFormatter('%Y-%m-%d') )
ax.xaxis.set_minor_locator( WeekdayLocator() )
fig.autofmt_xdate()
plt.tight_layout()
ax.grid()
ax.legend(fontsize='x-small', labelspacing=0.2, frameon=True)
plt.show(block=True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
for i, c in enumerate(["SE", "NL"]): #, "NO", "DK", "DE", "BE"]:
    dfs[c] = dfs[c][ dfs[c]["dateRep"] >= "2020-03-01" ]
    rects = ax.bar( dfs[c]["dateRep"] + i*pd.Timedelta('6 hours'),
                    dfs[c]["Sum"],
                    alpha=0.8,
                    #width=0.5,
                    width=pd.Timedelta('6 hours'), #24, #in "days"
                    align="center",
                    label=c,
                    #color=colours[0],
                    #edgecolor="black",
    )
fig.autofmt_xdate()
plt.tight_layout()
ax.grid()
ax.legend(fontsize='x-small', labelspacing=0.2, frameon=True)
plt.show(block=True)
