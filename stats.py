import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/natha/Documents/UNI/ECOS3997/e10sydhourly2021.csv')
df = df.drop(index=1, axis=1)
df = df.drop(labels=['Average', 'Average_1'], axis=1)
df = df.reset_index(drop=True)

dfId = pd.read_csv('C:/Users/natha/Documents/UNI/ECOS3997/e10stationID.csv')
dfpost = dfId.groupby(['Postcode']).count()
dfpost = dfpost[dfpost['Brand'] > 1]

nonmajor_df = dfId[dfId.Brand.str.contains('7-Eleven|BP|Ampol|Caltex|Coles')==False]
nonmajor_df = nonmajor_df.append(dfId.iloc[213:219])
postcodes = nonmajor_df['Postcode'].to_list()

dfI = dfId[~dfId['Postcode'].isin(postcodes)]
stationID = dfI.StationID.to_list()

samedf = df.iloc[0, 2:].isin(stationID)
name_list = samedf[samedf == True].index.to_list()
#finaldf = df.loc[2:,name_list]
#finaldf.plot(legend=False)

spread = pd.DataFrame()
for postcode in all_postcodes:
    postcode_df = dfId[dfId['Postcode'] == postcode]
    station_list = postcode_df.StationID.to_list()
    same_temp = df.iloc[0, 2:].isin(station_list)
    same_temp = df.loc[2:,same_temp[same_temp == True].index]
    same_temp[postcode] = (same_temp.max(axis=1)-same_temp.min(axis=1))
    spread = spread.append(same_temp[postcode])
    
spread = spread.T

spread = spread.replace({0:np.nan})
spread.mean(axis=1).plot(legend = False,
                         title='Mean Range of Same-Suburb Station Petrol Prices (Sydney)',
                         xlabel='Time in hours since the beginning of 2021',
                         ylabel='Spread (cents)')

spread.mean(axis=1).density()

all_postcodes = dfId.Postcode.to_list()



dfI2 = dfId[dfId['Postcode'] == 2066]
stationID2 = dfI2.StationID.to_list()
samedf2 = df.iloc[0, 2:].isin(stationID2)
finaldf2 = df.loc[2:,samedf2[samedf2 == True].index]
finaldf2.plot(title='Petrol Prices in Lane Cove (2066) By Station and Firm',
              xlabel='Time in hours since the beginning of 2021',
              ylabel='Price in Cents').legend(bbox_to_anchor=(1,1))

dfId = dfId['Postcode']]

df1 = pd.DataFrame()

df1['TGP'] = df.tgp

df1['711'] = df.iloc[1:,2:33].mean(axis=1)
df1['Ampol'] = df.iloc[1:,34:56].mean(axis=1)
df1['BP'] = df.iloc[1:,57:92].mean(axis=1)
df1['Budget'] = df.iloc[1:,94:103].mean(axis=1)
df1['Caltex'] = df.iloc[1:,104:140].mean(axis=1)
df1['CaltexW'] = df.iloc[1:,141:158].mean(axis=1)
df1['Coles'] = df.iloc[1:,159:184].mean(axis=1)
df1['Independent'] = df.iloc[1:,188:196].mean(axis=1)
df1['Metro'] = df.iloc[1:,197:210].mean(axis=1)
df1['United'] = df.iloc[1:,215:219].mean(axis=1)

df1.plot()


dfHigher = (df1.iloc[1:,1:8].diff() > 5).astype(int)
dfHigher[1313:1450].plot(xlabel='Time in hours since the beginning of 2021', 
              title='Occurences Where Firm Median Petrol Price rose over 5 cents',
              ylabel='1 if Median Price rose >5c')

# 1:280, 423:1314, 1430:2269, 2414:3134, 3349:4046, 4192:5443, 5558:6580, 6905:7695, 7960:
# leader: BP, Caltex, Caltex, CaltexW, CaltexW, CaltexW, CaltexW, CaltexW, BP
cycle_length = [891, 839, 720, 697, 1251, 1022, 790]
cycle_mean = [887,887,887,887,887,887,887]
cycle_names = ['Caltex', 'Caltex', 'CaltexW', 'CaltexW', 'CaltexW', 'CaltexW', 'CaltexW']
cycles = pd.DataFrame(cycle_length, cycle_names, columns=['Cycle Length'])
cycles['mean'] = cycles['Cycle Length'].mean()
cycles.plot(xlabel='Cycle Starter (In Chronological Order)', 
              title='Cycle Length of Each Price Cycle in Sydney 2021',
              ylabel='Cycle Length (Hours)',
              color=['Black', 'Orange'],
    )

df1['AvgDiff'] = df1.iloc[1:,1:8].pct_change().mean(axis=1)
df1['Change'] = (df1.iloc[1:,1:8].diff().abs() > 0).astype(int).sum(axis=1)


df1 = df1.drop(labels=['AvgDiff', 'Higher', 'Change'], axis=1)
df1.Change.value_counts()
df1.Change.mean()
ax = df1.Change.value_counts().plot.bar(
              logy=True,
              ylim=(0,20000),
              xlabel="Amount of Firm's Who's Median Changed", 
              title="Amount of Periods Where Firm's Median Simultaneously Changed",
              ylabel='Amount of Periods (log)')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()),     
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points')
    
df1.Change.plot()
(df1.Change > 0).astype(int).value_counts()
df1.Change.mean()

df1.Change.plot()

fig = px.line(df1, title = "Mean Hourly Petrol Prices in Sydney (2021)",
              labels={
                  'index':'Time In Hours Since 2021 Began',
                  'value':'Mean Price'
                  })

fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_size': 20,
    'font_color': 'Black'
    })

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

plot(fig)
fig.show()

posChange = (df.iloc[1:,2:].diff() > 0).astype(int).sum(axis=1).sum()
negChange = (df.iloc[1:,2:].diff() < 0).astype(int).sum(axis=1).sum()
zeroChange = (df.iloc[1:,2:].diff() == 0).astype(int).sum(axis=1).sum()

diffdf = df.iloc[1:,2:].pct_change()

avgNegChange = diffdf[diffdf.iloc[1:,2:] < 0].stack().mean()
avgPosChange = diffdf[diffdf.iloc[1:,2:] > 0].stack().mean()

dfChange = diffdf[(diffdf.iloc[1:,2:] !=  0)].stack().dropna()

dfChange.plot.kde(bw_method=0.4, logy=True, ylim=(0.1,15), title='Density plot of Non-Zero Price Changes (skew=3)')

ax = dfChange.plot.hist(bins=12, logy=True, 
                        title='Histogram of Non-Zero Price Changes', 
                        ylim=(0,25000),
                        edgecolor='black', linewidth=.5)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2),
                (p.get_x()+p.get_width()/2., p.get_height()),     
                ha='center',
                va='center',
                xytext=(0, 10),
                textcoords='offset points',
                color='Black')

dfChange.skew()

df.iloc[1:,2:].pct_change().stack().skew()

df2 = pd.DataFrame()
df2 = df.iloc[2:,2:-2].pct_change()

df2.mean(axis=1).mean()
df2.plot.kde(bw_method=0.5, logy=True)
df.Change.plot.hist(bins=10)

dfmean = pd.DataFrame()
dfmean['Price'] = df.iloc[1:,2:].mean(axis=0)
dfmean = dfmean.dropna()
dfmean = dfmean.reset_index()
dfmean['Major'] = dfmean['index'].str.contains('7-Eleven|BP|Ampol|Caltex|Coles').astype(int)
dfmean['Major'] = dfmean['index'].str.contains('7-Eleven|BP|Ampol|Caltex|Coles').astype(int).value_counts()

dfmean.to_csv('sydney_petrol.csv')


