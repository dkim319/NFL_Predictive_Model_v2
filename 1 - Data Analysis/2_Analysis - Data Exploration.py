# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:21:23 2017

@author: DKIM
"""

import pandas as pd
import numpy as np

# required libraries loaded 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

seednumber = 319

data = pd.read_csv('Data.csv')

# Initial dataset 
print('Initial dataset dimensions')
print(data.shape)

target_year = 2017

print('Filter to only the training data')
orig_data = data[data['season'] <= target_year]

# Data Preprocessing

# replace any null values with 0
data = data.fillna(0)

# use one-hot coding to replace the favorite and underdog categorical variables
fav_team = pd.get_dummies(data['favorite'])
und_team = pd.get_dummies(data['underdog'])

# use a prefix to distinguish the two categorical variables
fav_team = fav_team.add_prefix('fav_')
und_team = und_team.add_prefix('und_')

# remove the original fields
data = data.drop('favorite', axis = 1)
data = data.drop('underdog', axis = 1)

# add the one-hot coded fields
data = pd.concat([data, fav_team], axis = 1)
data = pd.concat([data, und_team], axis = 1)

#print data.head(5)
#print(data.describe())

# split the dataset into training and testing datasets
data = data[data['season'] <= target_year]
data.reset_index()

print('Final dataset dimensions')
print(data.shape)

#statistics = data.describe()
#statistics.to_csv('stats.csv')
print('Review the distribution of the target variable')
print('Target variable is evenly distributed and is not skewed')

spread_by_year = data.groupby(['season'])['spreadflag'].mean()
print(spread_by_year)

corr_data = data.corr(method = 'pearson')

print('Review the correlation between the variables and the target variable')
print('Top 10 correlated variables')
print(corr_data['spreadflag'].sort_values(ascending=False).head(11))

print('Top 10 negatively correlated variables')
print(corr_data['spreadflag'].sort_values(ascending=True).head(10))

years = [2010,2011,2012,2013,2014,2015,2016,2017]

for x in years:
    year_data = data[data['season'] == x]
    
    year_data_corr = year_data.corr(method = 'pearson')
    
    print('Top 10 correlated variables for the target variable, spreadflag, for the year ' + str(x))
    print(year_data_corr['spreadflag'].sort_values(ascending=False).head(11))
    print('')

    print('Top 10 negatively correlated variables for the target variable, spreadflag, for the year ' + str(x))
    print(year_data_corr['spreadflag'].sort_values(ascending=True).head(10))
    print('')

# Plot favorite win % over spread
spread_agg =  data.groupby(['spread'])['spreadflag'].mean()
spread_count = data.groupby(['spread'])['spreadflag'].count() / data.shape[0]

fig, axes = plt.subplots(2,1)

spread_agg_ax = spread_agg.plot(ax = axes[0])
spread_agg_ax.set_ylabel('favorite win %')
spread_agg_ax.set_title('Figure 1 - Spread')
spread_agg_figure = spread_agg_ax.get_figure()

spread_count_ax = spread_count.plot(kind = 'line',ax = axes[1])
spread_count_ax.set_ylabel('spread %')
spread_count_figure = spread_count_ax.get_figure()
plt.show()
#plt.savefig('2b - fig 1 - spread_vis.png')

# Plot the favorite win % over total 
total_agg =  data.groupby(['total'])['spreadflag'].mean()
total_count = data.groupby(['total'])['spreadflag'].count() / data.shape[0]
 
fig, axes = plt.subplots(2,1)

total_agg_ax = total_agg.plot(ax = axes[0])
total_agg_ax.set_ylabel('favorite win %')
total_agg_ax.set_title('Figure 2 - Total')
total_agg_figure = total_agg_ax.get_figure()

total_count_ax = total_count.plot(kind = 'line',ax = axes[1])
total_count_ax.set_ylabel('total %')
total_count_figure = total_count_ax.get_figure()
plt.show()
#plt.savefig('2b - fig 2 - total_vis.png')
  
# Check the Team over winning %
favorite_win_percent = orig_data.groupby(['favorite'])['spreadflag'].mean()
underdog_win_percent = 1 - orig_data.groupby(['underdog'])['spreadflag'].mean()

print('Top 10 Favorites by ATS percent')
print(favorite_win_percent.sort_values(ascending=False).head(10))
print('')

print('Top 10 Underdogs by ATS percent')
print(underdog_win_percent.sort_values(ascending=False).head(10))
print('')

# Plot the favorite win % over favorite's win record over last 5 and 10 games
fav_last_5_percent_vis_agg =  data.groupby(['fav_last_5_percent'])['spreadflag'].mean()
fav_last_10_percent_vis_agg = data.groupby(['fav_last_10_percent'])['spreadflag'].mean()
 
fig, axes = plt.subplots(2,1)

fav_last_5_percent_vis_agg_ax = fav_last_5_percent_vis_agg.plot(ax = axes[0])
fav_last_5_percent_vis_agg_ax.set_ylabel('favorite win %')
fav_last_5_percent_vis_agg_ax.set_title('Figure 3a - Favorite Win % Last 5 Games')
fav_last_5_percent_vis_agg_figure = fav_last_5_percent_vis_agg_ax.get_figure()
fav_last_5_percent_vis_agg_figure.subplots_adjust(hspace=0.75)

fav_last_10_percent_vis_agg_ax = fav_last_10_percent_vis_agg.plot(kind = 'line',ax = axes[1])
fav_last_10_percent_vis_agg_ax.set_ylabel('favorite win %')
fav_last_10_percent_vis_agg_ax.set_title('Figure 3b - Favorite Win % Last 10 Games')
fav_last_10_percent_vis_count_figure = fav_last_10_percent_vis_agg_ax.get_figure()
plt.show()
#plt.savefig('2b - fig 3 - fav_last_5_percent.png')


# Plot the favorite win % over underdog's win record over last 5 and 10 games

undlast_5_percent_vis_agg =  data.groupby(['und_last_5_percent'])['spreadflag'].mean()#.sum()/ data.groupby(['spread'])['spreadflag'].count()
und_last_10_percent_vis_agg = data.groupby(['und_last_10_percent'])['spreadflag'].mean()
 
fig, axes = plt.subplots(2,1)

und_last_5_percent_vis_agg_ax = undlast_5_percent_vis_agg.plot(ax = axes[0])
und_last_5_percent_vis_agg_ax.set_ylabel('underdog win %')
und_last_5_percent_vis_agg_ax.set_title('Figure 4a - Underdog Win % Last 5 Games')
und_last_5_percent_vis_agg_figure = und_last_5_percent_vis_agg_ax.get_figure()
und_last_5_percent_vis_agg_figure.subplots_adjust(hspace=0.75)

und_last_10_percent_vis_agg_ax = und_last_10_percent_vis_agg.plot(kind = 'line',ax = axes[1])
und_last_10_percent_vis_agg_ax.set_ylabel('underdog win %')
und_last_10_percent_vis_agg_ax.set_title('Figure 4b - Underdog Win % Last 10 Games')
und_last_10_percent_vis_agg_figure = und_last_10_percent_vis_agg_ax.get_figure()
plt.show()    
#plt.savefig('2b - fig 4 - und_last_5_percent.png')

