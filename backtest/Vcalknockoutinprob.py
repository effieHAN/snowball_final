# -*- coding: utf-8 -*-
"""
Created on Tue 16 May 2023

@author: effiehan
"""
from database.code.backtestzz1000 import backtestdatazz1000_
from database.code.backtestzz500 import backtestdatazz500_
from utilfun.utilfun import get_next_months_dates_, replace_with_tradingday_
import pandas as pd
import warnings
from database.code.datetradingupdate import TradingDay_

warnings.filterwarnings('ignore')


class VCalknockprob_:
    # 定义公式所需的变量
    def __init__(self, tick, fromdt, todt, knockinprice, knockoutprice, lockmonth, holding_period, stepdown=0):
        self.testdata = None
        self.date_obv = []
        self.data_ttm = []

        self.knockinprice = knockinprice
        self.knockoutprice = knockoutprice
        self.tick = tick
        self.lockmonth = lockmonth
        self.stoptime = str(int(todt) - (holding_period / 12 * 10000))
        # 如数据截止到2023.05.01 最后可以进行回测的产品的发行时间必须早于2021.05.01 才有数据回测
        self.holding_period = holding_period
        self.stepdown = stepdown
        self.todt = todt
        self.fromdt = fromdt

    # 数据初始化 将testdata导入
    def initdataclass_(self,data):
        if data=='zz1000':
            dbclass=backtestdatazz1000_()
        elif data=='zz500':
            dbclass = backtestdatazz500_()  # testdataforzzz500
        self.testdata = dbclass.getdata_(fromdt=self.fromdt, todt=self.todt)

    def get_data(self):
        data = self.testdata[['date', self.tick, 'pe']]
        return data

    # input是已经处理好的dataframe 第一行当前产品 从当天起找未来24个月内的观察日 并判断 knockin knock out
    # 注意封闭期3个月 每个观察日标记 1 ，累计观察日信号，如果封闭期后 出现敲出则有收益（即累计信号大于3 且当日观察日价格大于期初）
    def knock_func_(self, datause):
        date_obv = get_next_months_dates_(datause['date'].iloc[0], n=self.holding_period)
        start = date_obv[0]
        tradingday = TradingDay_(fromdt=start)['date'].tolist()
        date_obv = replace_with_tradingday_(date_obv, tradingday)
        datause = datause[(datause['date'] >= date_obv[0]) & (datause['date'] <= date_obv[-1])]
        # 将每天的数据认为是一个新发产品 对每天的产品（在未来2年持有期内）判断是否敲入敲出
        self.data_ttm = datause
        self.date_obv = date_obv
        datause.loc[:, "signal"] = 0
        datause.loc[datause.date.isin(date_obv[1:]), "signal"] = 1
        datause.loc[:, 'cumsignal'] = datause['signal'].cumsum()
        assert datause['signal'].sum() == self.holding_period
        knock_in = "未敲入"
        knock_out = "未敲出"
        position = 0
        for num in range(1, len(datause)):
            position = num  # we dont care startdate and enddate [1,len(dateuse))
            if datause[self.tick].iloc[num] < (datause[self.tick].iloc[0] * self.knockinprice):
                knock_in = "敲入"
            if (datause["signal"].iloc[num] == 1) & (datause['cumsignal'].iloc[num] >= self.lockmonth):
                if datause[self.tick].iloc[num] >= datause[self.tick].iloc[0] * (
                        self.knockoutprice - self.stepdown / 100 * (datause["cumsignal"].iloc[num] - 1)):
                    knock_out = "敲出"
                    break
        res_list = [datause['date'].iloc[0], knock_in, knock_out, datause["date"].iloc[position]]
        return res_list

    # 将每天的数据认为是一个新发产品 对每天的产品（在未来2年持有期内）判断是否敲入敲出
    def cal_knock_prob_(self):
        res_set = []
        dataset = self.testdata[['date', self.tick]]
        for i in range(len(dataset)):
            res = dataset.iloc[i:]
            # print(res)
            if res['date'].iloc[0] > self.stoptime:
                break
            fin = self.knock_func_(res)
            res_set.append(fin)
        res_set = pd.DataFrame(res_set)
        res_set.columns = ['date', 'knockin', 'knockout', 'expiremonth']
        res_set['result'] = res_set.knockin + res_set.knockout
        # res_set['result'] = res_set['result'].replace(['敲入敲出', '未敲入敲出'], '敲出')
        pedata = self.get_data()
        res_set_with_pe = pd.merge(res_set, pedata[['date', 'pe']], on='date')
        return res_set_with_pe

# if __name__ == '__main__':
#     dbcalss = VCalknockprob_(tick='price', knockinprice=0.75, knockoutprice=1, stepdown=0.5,lockmonth=3, fromdt='20070115', todt='20230511', holding_period=24)
#     dbcalss.initdataclass_('zz500')
#     res_vsnowall = dbcalss.cal_knock_prob_()
