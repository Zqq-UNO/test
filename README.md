import numpy as np
import pandas as pd
import scipy.stats as stats #统计模块
import scipy
# import pymysql #导入数据库模块

from datetime import datetime #时间模块
import statsmodels.formula.api as smf # OLS regression

# import pyreadr # read RDS file

from matplotlib import style
import matplotlib.pyplot as plt #画图模块
import matplotlib.dates as mdates

from matplotlib.font_manager import FontProperties #作图中文
from pylab import mpl

# 输出矢量图 渲染矢量图
%matplotlib inline
%config InlineBackend.figure_format='svg'

from IPython.core.interactiveshell import InteractiveShell #jupyter运行输出的模块
#显示每一个运行结果
InteractiveShell.ast_node_interactivity='all'

#设置列不限制数量
pd.set_option('display.max_columns',None)
data = pd.read_csv("D:/python-homework/python-homework/000001.csv")#导入数据
data['Day']=pd.to_datetime(data['Day'],format='%Y/%m/%d')#变成时间格式
data.set_index('Day',inplace=True)#生成索引
data.sort_values(by=['Day'],axis=0,ascending=True)#升序排序
data_new=data['1995-01':'2024-09'].copy()#用copy来进行深度复制
data_new['Close']=pd.to_numeric(data_new['Close'])#变成数值型
data_new['Preclose']=pd.to_numeric(data_new['Preclose'])#变成数值型
#计算回报率
data_new['Raw_return']=data_new['Close']/data_new['Preclose']-1
data_new
Month_data=data_new.resample('ME')['Raw_return'].apply(lambda x:(1+x).prod()-1).to_frame()
Month_data
inflation=pd.read_csv("D:/python-homework/python-homework/国内生产总值GDP累计同比.csv")
inflation.dropna(axis=0, how='any', inplace=True)#删去缺失值
inflation['date']=pd.to_datetime(inflation['date'],format='%Y-%m-%d')#变成时间格式
inflation.set_index('date',inplace=True)#设置成索引
inflation.sort_values(by=['date'],axis=0,ascending=True)#升序排序
market_variance = data_new.resample('ME').apply(
    {
        'Raw_return':lambda x:sum(x**2)
    }
)
market_variance.reset_index(inplace=True)#把索引给去掉
market_variance.rename(columns={'Day':'month','Raw_return':'RV'},inplace=True)#把名字给换掉
market_variance.set_index('month',inplace=True)#用月当作索引
market_variance
reg_data=pd.merge(Month_data,inflation,left_index=True,right_index=True,how='inner')
reg_data=pd.merge(reg_data,market_variance,left_index=True,right_index=True,how='inner')
reg_data
reg_data['rate'].describe().round(3)#保留3位小数
#看峰度和偏度
reg_data.skew().round(3)#偏度
reg_data.kurt().round(3)#峰度
fig, ax=plt.subplots(figsize=(14,8))

ax.plot(inflation['rate'],label='Inflation',#图片数据
linestyle='-',#图片类型，
color='#D98719',#图片颜色
linewidth=4,#图片线宽
marker='o')
ax.set_ylabel('Inflation')
#添加图例
ax.legend(loc='upper right',fontsize=12)
plt.show();
# 画多条直线图
fig, ax1=plt.subplots(figsize=(14,8))
ax1.plot(reg_data['Raw_return'].shift(1),label='Return',
linestyle='-',#图片类型，
color='red',#图片颜色
linewidth=4,#图片线宽
marker='o')
ax1.set_ylabel('Return',color='red')
data_format=mdates.DateFormatter('%Y')
ax1.xaxis.set_major_formatter(data_format)
ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=90)
ax2=ax1.twinx()#把图纸镜像转一下
ax2.plot(reg_data['rate'],label='inflation',
linestyle='-',#图片类型，
color='blue',#图片颜色
linewidth=4,#图片线宽
marker='o')
ax2.set_ylabel('Inflation',color='blue')

plt.title("China's Return and Inflation")
#提取图例
lines,labels=ax1.get_legend_handles_labels()
lines2,labels2=ax2.get_legend_handles_labels()
ax2.legend(lines+lines2,labels+labels2,loc='upper right')#合并到第二张图

plt.show();
reg_data['lRV'] = reg_data['RV'].shift(1)
reg_data['lrate']=reg_data['rate'].shift(1)#有滞后性
model_forecast_rv = smf.ols('RV ~ lrate', data=reg_data).fit(
    cov_type='HAC' ,
    cov_kwds={'maxlags':6}
)   #左边的是被解释变量，右边的是解释变量
print(model_forecast_rv.summary())
reg_data['lRV'] = reg_data['RV'].shift(1)
reg_data['lrate']=reg_data['rate'].shift(1)
model_forecast_rv = smf.ols('RV ~ lrate', data=reg_data['2000':]).fit(
    cov_type='HAC' ,
    cov_kwds={'maxlags':6}
)  
print(model_forecast_rv.summary())
reg_data['lRV'] = reg_data['RV'].shift(1)
model_forecast_rv = smf.ols('RV ~ lRV', data=reg_data['2000':]).fit(
    cov_type='HAC' ,
    cov_kwds={'maxlags':6}
)
print(model_forecast_rv.summary())
model_forecast_rv = smf.ols('RV ~ lRV+lrate', data=reg_data['2000':]).fit(
    cov_type='HAC' ,
    cov_kwds={'maxlags':6}
)
print(model_forecast_rv.summary())
reg_data_new=reg_data['2000':].copy()
reg_data_new['fitted_RV']=model_forecast_rv.fittedvalues

model_forecast_return=smf.ols('Raw_return~fitted_RV',data=reg_data_new).fit(
    cov_type='HAC' ,
    cov_kwds={'maxlags':6}
)
print(model_forecast_return.summary())
model_rate=smf.ols('Raw_return~lrate',data=reg_data['2000':]).fit(
    cov_type='HAC' ,
    cov_kwds={'maxlags':6}
)
print(model_rate.summary())
print('hello world')
