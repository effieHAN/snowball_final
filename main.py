# -*- coding: utf-8 -*-
"""
Created on Tue 16 May 2023

@author: effiehan
"""
import pandas as pd
from utilfun.utilfun import calculate_percentage_
from backtest.Vcalknockoutinprob import VCalknockprob_
from backtest.Bcalknockoutinprob import BCalknockprob_
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # In[] 普通雪球
    vsnowall = VCalknockprob_(tick='price', knockinprice=0.75, knockoutprice=1, stepdown=0.5,lockmonth=3, fromdt='20070115', todt='20230511', holding_period=24)
    vsnowall.initdataclass_('zz500')
    res_vsnowall = vsnowall.cal_knock_prob_()

    vpebelow20 = res_vsnowall[res_vsnowall.pe < 20]
    vpe20_30 = res_vsnowall[(res_vsnowall.pe >= 20) & (res_vsnowall.pe < 30)]
    vpe30_40 = res_vsnowall[(res_vsnowall.pe >= 30) & (res_vsnowall.pe < 40)]
    vpeabove40 = res_vsnowall[res_vsnowall.pe >= 40]
    stat_vall = []
    for i in [res_vsnowall, vpebelow20, vpe20_30, vpe30_40, vpeabove40]:
        stat = calculate_percentage_(i, 'result')
        stat_vall.append(stat)
    stat_vall = pd.concat(stat_vall, axis=1)
    stat_vall.columns = ['total_count', 'total_prob', 'below20_count', 'below20_prob', '20-30count', '20-30_prob', '30-40_count', '30-40_prob',
                         'above40_count', 'above40_prob', ]
    # In[] 二元雪球
    bsnowball = BCalknockprob_(tick='price', knockoutprice=1, fromdt='20070115', lockmonth=3,todt='20230511', holding_period=12)
    dadata = bsnowball.initdataclass_('zz500')
    res_bsnowall = bsnowball.cal_knock_prob_()

    bpebelow20 = res_bsnowall[res_bsnowall.pe < 20]
    bpe20_30 = res_bsnowall[(res_bsnowall.pe >= 20) & (res_bsnowall.pe < 30)]
    bpe30_40 = res_bsnowall[(res_bsnowall.pe >= 30) & (res_bsnowall.pe < 40)]
    bpeabove40 = res_bsnowall[res_bsnowall.pe >= 40]
    stat_ball = []
    for i in [res_bsnowall, bpebelow20, bpe20_30, bpe30_40, bpeabove40]:
        stat = calculate_percentage_(i, 'knockout')
        stat_ball.append(stat)
    stat_ball = pd.concat(stat_ball, axis=1)
    stat_ball.columns = ['total_count', 'total_prob', 'below20_count', 'below20_prob', '20-30count', '20-30_prob', '30-40_count', '30-40_prob',
                         'above40_count', 'above40_prob', ]

    writer = pd.ExcelWriter(r"snowball_knock_prob.xlsx", engine='xlsxwriter')
    # fmt = writer.book.add_format({"font_name": "Arial"})
    res_vsnowall.to_excel(writer, index=False, sheet_name='vanilla')
    res_bsnowall.to_excel(writer, index=False, sheet_name='binomial')
    writer.save()
    writer.close()
