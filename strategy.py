''' Download Stock data from Yahoo finance '''

''' Download numpy and yahoo finance packages '''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pylab
import yahoo_finance
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import *
#from matplotlib.finance import quotes_historical_yahoo_ohlc
#from matplotlib.finance import candlestick_ohlc


''' Pandas packages '''

import datetime
from datetime import datetime
import pandas as pd
from pandas.io.data import DataReader
import pandas.io.data as web
#from pandas_datareader import data # remove the deprecated pandas_datareader package

''' Other packages '''

import StringIO  # Python 2

import csv

''' Start and End dates'''
start_date=str('2015 01 01').split(' ')

start_date=map(int,start_date)

start= datetime(start_date[0],start_date[1],start_date[2])



end_date=str('2016 10 20').split(' ')

end_date=map(int,end_date)

end = datetime(end_date[0],end_date[1],end_date[2])



''' Tickers list '''

symbols_list=['GOOGL']


tickers=[]

for i in range(len(symbols_list)):
    tickers.append(symbols_list[i])


''' Closing stock data '''

ls_key='Close'


''' Use Pandas DataReader to download ticker data from Yahoo and store in stockdata '''

stockdata=[]

for i in range(len(symbols_list)):
    stockdata.append(web.DataReader(tickers[i],'yahoo',start,end))


n=len(stockdata[0]['Close']) # Number of days


''' Moving Averages 20 day, 50 day, 200 day '''


aapl20_day=pd.rolling_mean(stockdata[0][ls_key],window=20)
aapl50_day=pd.rolling_mean(stockdata[0][ls_key],window=50)
aapl200_day=pd.rolling_mean(stockdata[0][ls_key],window=200)



''' Calculate Relative Strength Index (RSI)'''

def rsi(x,window):
    m=np.diff(x)
    dup=m.copy()
    ddown=m.copy()
    dup[dup < 0]=0
    ddown[ddown > 0]=0
    roll_up=pd.rolling_mean(dup,window)
    roll_down=pd.rolling_mean(np.abs(ddown),window)
    rs=roll_up/roll_down
    return 100.0 - (100.0/(1+rs))

current_stock_price=stockdata[0]['Close']

RSI=rsi(current_stock_price,14)





''' Bollinger Bands

N=20d moving average

Upper band = N+K*sigma

lower band= N-K*sigma



'''



K=2

# 20d MA
N=pd.rolling_mean(current_stock_price,window=20)



#Standard deviation of Stock price
sigma=pd.rolling_std(current_stock_price,20)

# Upper & lower Bollinger bands
upper_band=N+K*sigma

lower_band=N-K*sigma


''' Plots '''

fig=plt.figure()
textsize=3
plt.title(symbols_list[0] + ' daily')
#fig.subplots_adjust(bottom=0.2)
ax=fig.add_subplot(2,1,1)
plt.ylabel(symbols_list[0])
alldays = DayLocator()   
quotes = quotes_historical_yahoo_ohlc(symbols_list[0], start, end) # Get ohlc quotes
candlestick_ohlc(ax, quotes,width=0.1) # Plot Candlestick bars
ax.xaxis.set_minor_locator(alldays)
ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.plot(N,'-',label='MA(20)')
plt.plot(upper_band,label='Upper BB')
plt.plot(lower_band,label='lower BB')
plt.legend(loc='upper right',fancybox=True)
plt.grid(True)

ax1=fig.add_subplot(2,1,2)
plt.ylabel('RSI')
fillcolor='darkgoldenrod'
ax1.xaxis_date()
plt.plot(RSI,label='RSI')
ax1.axhline(y=30,color=fillcolor,lw=2,label='RSI <= 30 = oversold')
#ax1.text(0.2,0.3,'RSI <30= oversold')
plt.axhline(y=70,color=fillcolor,lw=2,label='RSI >= 70 = overbought')
plt.grid(True)
ax1.legend(fancybox=True)



plt.show()

'''
Storing the Date column of aapl from the corresponding DataFrame object in a
matrix called Date_stocks

'''


stockdata[0].to_csv('~/Desktop/stockdata-2015-2016.csv')
df=pd.read_csv('~/Desktop/stockdata-2015-2016.csv')




for i in range(len(df)):
    df.loc[i,'Day'] = datetime.strptime(df.loc[i,'Date'], '%Y-%m-%d').day


Date_stocks=np.array([])

for i in range(len(df)):
    Date_stocks=np.append(Date_stocks,df.loc[i,'Date'])



''' Generate the Swing Lows and prior Swing lows'''



daily_open=stockdata[0]['Open']
daily_close=stockdata[0]['Close']
daily_low=stockdata[0]['Low']

swing_low=np.array([])
date_swing_low=np.array([]) # This calculates the absolute index of the dates for the swing lows

'''
for i in range(n-2):
    if daily_close[i]<=daily_open[i] and daily_close[i+1]<=daily_open[i+1] and daily_close[i+2]>=daily_open[i+2]:
        if daily_low[i+1]<=daily_low[i] and daily_low[i+1]<=daily_low[i+2]:
            swing_low=np.append(swing_low,daily_close[i+1])
            date_swing_low=np.append(date_swing_low,i+1)
'''

for i in range(n-2):
    if daily_close[i+1]<=daily_close[i] and daily_close[i+1]<=daily_close[i+2]:
        swing_low=np.append(swing_low,daily_close[i+1])
        date_swing_low=np.append(date_swing_low,i+1)
            

def prior_swing_low(i):
    p=np.where(date_swing_low<=i)[0]
    s=p[len(p)-1]
    return swing_low[s]



''' Identify signals for Long positions after swing lows'''


def check_long_position(a,b,dates):
    counter=0
    long_price=0
    c_long=0
    long_signals=np.array([])
    long_dates=np.array([])
    start=dates[0]
    end=dates[1]
    date_now=0
    prices=b[start:end]
    for i in range(len(prices)):
        date_now=start+i
        if  prices[i]<=1.02*a and prices[i]>=lower_band[date_now] and RSI[date_now]<=30 and N[date_now]>=1.05*prices[i]:
            long_signals=np.append(long_signals,prices[i])
            long_dates=np.append(long_dates,date_now)
            counter+=1
    if counter>0:
        long_price=np.amin(long_signals)
        c_long=int(long_dates[np.where(long_signals==long_price)][0])
            
    return counter,long_price,Date_stocks[c_long]
    

# Long positions and stop orders

first_long=0
initial_investment=0
sell_stop_order=0
strategy_return=np.array([]) # Stores the long and the short positions

for i in range(len(swing_low)-1):

    
    a=swing_low[i]
    b=current_stock_price
    dates=map(int,date_swing_low[i:i+2])
    index=check_long_position(a,b,dates)[0]
    
    
    
    if index > 0 and first_long==0:
        first_long=1
        sell_stop_order=0.99*swing_low[i]
        print 'Enter first long position on',check_long_position(a,b,dates)[2],'at $',check_long_position(a,b,dates)[1]
        initial_investment=check_long_position(a,b,dates)[1]
        print 'Sell-stop order currently set at $',sell_stop_order
        print '---------------------------'
        
        
        strategy_return=np.append(strategy_return,check_long_position(a,b,dates)[1])
        
        continue

    if swing_low[i]> swing_low[i-1] and first_long>0:
        sell_stop_order=0.99*swing_low[i]
        dt=int(date_swing_low[i])
        print 'Sell-stop order increased on',Date_stocks[dt],'to $',sell_stop_order
        print '---------------------------'
        

    if swing_low[i]<swing_low[i-1] and first_long > 0:
        if index >0:
            low_counter=0
            sell_stop_order=0.99*swing_low[i]
            date_prior=map(int,date_swing_low[i-1:i+1])
            #print('Date prior',date_prior)
            for j in range(date_prior[0],date_prior[1]):
                if current_stock_price[j]<= 0.99*swing_low[i-1]:
                    print 'Exit Long position on',Date_stocks[j],'at $',current_stock_price[j]
                    strategy_return=np.append(strategy_return,current_stock_price[j])
                    low_counter=1
                    print '---------------------------'
                    break
            if low_counter==0:
                print 'Exit Long position at $',0.99*swing_low[i-1]
                strategy_return=np.append(strategy_return,0.99*swing_low[i-1])
                print '---------------------------'    
            print 'Enter new long on',check_long_position(a,b,dates)[2],'at $',check_long_position(a,b,dates)[1]
            print 'Sell-stop order decreased on',check_long_position(a,b,dates)[2],'to $',sell_stop_order
            strategy_return=np.append(strategy_return,check_long_position(a,b,dates)[1])
            print '---------------------------'    
     
        
        
        


''' First Close out position '''

strategy_return=np.append(strategy_return,sell_stop_order) # Go Short on the stop order




'''
Calculate Return of trading strategy

'''
actual_strategy_return=np.array([]) # Calculates the periodic returns (M_(t+1)-M_t)/M_t


for i in range(len(strategy_return)-1):
    if i%2==0:
        arr=strategy_return[i:i+2]
        x=(arr[1]-arr[0])/arr[0]
        actual_strategy_return=np.append(actual_strategy_return,x)

print 'Actual Strategy return',actual_strategy_return

print('True Strategy return r:', 1-np.prod(actual_strategy_return))




'''
Calculate Sharpe Ratio S of stock in given period:

S = sqrt(n)*Avg(d)/stdev(d)
d is the daily market return for the stock for a period of n days

daily return = (Current close - Prev close)/Prev close

'''


daily_return=[]
l=current_stock_price
for j in range(n-1):
        daily_return.append((l[j+1]-l[j])/float(l[j])) # Calculate daily return
        average_daily_return=np.average(daily_return)
        standard_deviation=np.std(daily_return)
print 'Sharpe ratio from market return for',symbols_list[0],': $',np.sqrt(n)*average_daily_return/float(standard_deviation)
























