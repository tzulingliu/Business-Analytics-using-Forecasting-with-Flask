#{{ jsondata["stores"][1].text }}-->
# jsondata=pd.read_json("store1.json")
from statsmodels.tsa.api import ExponentialSmoothing, Holt
from pmdarima.arima import auto_arima


from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm
import plotly.express as px
import ReadInData as rid
from math import ceil
import pandas as pd
import numpy as np
import plotly
import json
import time


# def display_decompition(number):
#     data = rid.data.query("店碼=={}".format(number))
#     services_list=["剪髮", "洗髮", "燙髮", "染髮", "護髮", "養髮", "養護", "美甲", "造型", "美髮其他"]
#     figs_all={}
#     for index, s in enumerate(services_list):
#         data_service = data.query("類別=='{}'".format(s))[["開單日期", "數量"]].set_index(["開單日期"])
#         try:
#             decomp=sm.tsa.seasonal_decompose(data_service["數量"], period=7)
#             obs = decomp.observed.reset_index()
#             sea = decomp.seasonal.reset_index()
#             trend = decomp.trend.reset_index()
#             resid= decomp.resid.reset_index()
#             pvalue=sm.tsa.stattools.adfuller(obs.set_index("開單日期"))[1]
#             decomposition=obs.merge(sea, on="開單日期").merge(trend, on="開單日期").merge(resid, on="開單日期")
#
#
#             figs = make_subplots(rows=4, cols=1)
#             figs.append_trace(go.Scatter(x=decomposition["開單日期"], y=decomposition["數量"], name="Original Series",
#                               line=dict(color=("rgb(246, 140, 166)"), )),row=1, col=1)
#             figs.append_trace(go.Scatter(x=decomposition["開單日期"], y=decomposition["seasonal"], name="Seasonal",
#                               line=dict(color=("rgb(212, 209, 203)"), )), row=2, col=1)
#             figs.append_trace(go.Scatter(x=decomposition["開單日期"],y=decomposition["trend"], name="Trend",
#                               line=dict(color=("rgb(160, 200, 103)"), )), row=3, col=1)
#             figs.append_trace(go.Scatter(x=decomposition["開單日期"], y=decomposition["resid"], name="Resid",
#                                          line=dict(color=("rgb(248, 221, 153)"), )), row=4, col=1)
#
#             figs.update_layout({"plot_bgcolor": "rgb(248, 249, 249)",},width=600, height=600, showlegend=False, title_text=s+" Dickey-Fuller test:"+str(round(pvalue, 4)))
#
#             figs_all[index]=figs
#         except:
#             continue
#
#     DecompJSON = json.dumps(figs_all, cls=plotly.utils.PlotlyJSONEncoder)
#     return DecompJSON
def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


def time_series(dfall, palette):
    # time series plot
    palette = ["#17202A", "#2C3E50", "#566573", "#ABB2B9", "#EAECEE", "#FADBD8", "#EDBB99", "#DC7633", "#A04000",
               "#78281F"]
    timeseries = px.line(dfall, x=dfall.index, y="數量", color="類別", color_discrete_sequence=palette, width=1100,
                         height=600)
    timeseries.update_layout({
        "plot_bgcolor": "rgb(248, 249, 249)",
    })

    return timeseries


def barplot(data, palette):
    # time series plot

    data_bar = data.groupby(["類別"])["數量"].sum().reset_index()
    bar = px.bar(data_bar, x="類別", y="數量", barmode="group", color_discrete_sequence=["#ABB2B9"] * 8, height=600)
    bar.update_layout({
        "plot_bgcolor": "rgb(248, 249, 249)",
    })

    return bar

def boxplot(dfall, palette):
    box = px.box(dfall, x="星期", y="數量", color="類別", color_discrete_sequence=palette, height=600)
    box.update_layout({
        "plot_bgcolor": "rgb(248, 249, 249)",
    })

    return box

def heatmapplot(data_pivot, palette_heatmap):
    data_pivot_corr = data_pivot.corr().replace(np.nan, 0)
    cols = list(data_pivot.columns)
    heapmap = px.imshow(data_pivot_corr, y=cols, x=cols, aspect="auto", color_continuous_scale=palette_heatmap)
    heapmap.update_layout({
        "plot_bgcolor": "rgb(248, 249, 249)",
    })

    return heapmap

def modeling(data_pivot, palette):
    #holt_winter
    validlen=16

    train=data_pivot[:-validlen]
    test=data_pivot[-validlen:]
    testplot=data_pivot[-(validlen+1):]

    fore_ets=testplot.copy()
    fore_arima = testplot.copy()
    fore_sn=testplot.copy()

    figs = make_subplots(rows=len(data_pivot.columns), cols=1, subplot_titles=list(data_pivot.columns))

    for index, sers in enumerate(data_pivot.columns):
        fit_ets = ExponentialSmoothing(train[str(sers)], trend="add", seasonal="add").fit()
        fit_arima = auto_arima(np.array(train[sers]), start_p=1, start_q=1, max_p=3, max_q=3, d=1, m=12,
                               start_P=1, start_Q=1, D=1, max_P=2, max_D=1, max_Q=2, max_order=2,seasonal=True)

        fit_sn=train[str(sers)][-52]

        fore_ets.loc[:, str(sers)] = list(fit_ets.forecast(len(testplot)))
        fore_arima.loc[:, str(sers)] = list(fit_arima.predict(len(testplot)))
        fore_sn.loc[:, str(sers)] = fit_sn

        rmse_ets= np.sqrt(mean_squared_error(testplot[str(sers)], fore_ets[str(sers)]))
        rmse_arima = np.sqrt(mean_squared_error(testplot[str(sers)], fore_arima[str(sers)]))
        rmse_sn = np.sqrt(mean_squared_error(testplot[str(sers)], fore_sn[str(sers)]))


        figs.append_trace(go.Scatter(x=train.index, y=train[str(sers)], name="Train",
                                     line=dict(color=palette[2])), row=index+1, col=1)
        figs.append_trace(go.Scatter(x=testplot.index, y=testplot[str(sers)], name="Test", mode="lines",
                                     line=dict(color=palette[3])), row=index+1, col=1)
        figs.append_trace(go.Scatter(x=fore_ets.index, y=fore_ets[str(sers)], name="Hoit-Winters'", mode="lines",
                                     line=dict(color="#FF69B4")), row=index+1, col=1)
        figs.append_trace(go.Scatter(x=fore_arima.index, y=fore_arima[str(sers)], name="ARIMA", mode="lines",
                                     line=dict(color="#2E8B57")), row=index+1, col=1)
        figs.append_trace(go.Scatter(x=fore_sn.index, y=fore_sn[str(sers)], name="Seasonal Naive", mode="lines",
                                     line=dict(color="#F4C430")), row=index+1, col=1)
        figs.update_xaxes(title_text="ETS MSE: "+str(round(rmse_ets, 2))+"  "+"ARIMA MSE: "+str(round(rmse_arima, 2))
                                     +"  "+"Seasonal Naive MSE: "+str(round(rmse_sn, 2)), row=index+1, col=1)

        figs.update_layout({"plot_bgcolor": "rgb(248, 249, 249)"}, width=1100, height=3600, showlegend=False)


    graph_ets = json.dumps(figs, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_ets

def display_time_series(number, shift):

    if shift=="M":
        data=rid.data[rid.data["時段"]=="M"]
        shift_name = "Morning Shift"
    elif shift=="A":
        data = rid.data[rid.data["時段"]=="A"]
        shift_name = "Afternoon Shift"
    elif shift=="E":
        data = rid.data[rid.data["時段"]=="E"]
        shift_name = "Evening Shift"
    else:
        data=rid.data
        shift_name = ""

    data = data.query("店碼=={}".format(number))
    data["開單日期"] = pd.to_datetime(data["開單日期"])

    dfall=pd.DataFrame()
    for i in set(data["類別"]):
        d=data[data["類別"]==i]
        df=d[["開單日期", "數量"]].resample("W", on="開單日期").sum()
        df["類別"]=i
        dfall=dfall.append(df)

    weekofmonth=[week_of_month(i) for i in dfall.index]
    dfall.loc[:, "星期"]=weekofmonth

    palette = ["#17202A", "#2C3E50", "#566573", "#ABB2B9", "#EAECEE", "#FADBD8", "#EDBB99", "#DC7633", "#A04000",
               "#78281F"]
    palette_heatmap = ["#78281F", "#DC7633", "#EAECEE", "#566573", "#17202A"]

    data_pivot = dfall.pivot_table(values="數量", columns="類別", index=["開單日期"])
    data_pivot = data_pivot.fillna(0)


    timeseries=time_series(dfall, palette)
    bar=barplot(data, palette)
    box=boxplot(dfall, palette)
    heapmap=heatmapplot(data_pivot, palette_heatmap)
    sersfor=modeling(data_pivot, palette)


    graph_timeseries = json.dumps(timeseries, cls=plotly.utils.PlotlyJSONEncoder)
    graph_bar = json.dumps(bar, cls=plotly.utils.PlotlyJSONEncoder)
    graph_box = json.dumps(box, cls=plotly.utils.PlotlyJSONEncoder)
    graph_heatmap = json.dumps(heapmap, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_timeseries, graph_bar, graph_box, graph_heatmap, sersfor, shift_name


# def auto_arima(number):
#     data = rid.data.query("店碼=={}".format(number))
#     services_list = ["剪髮", "洗髮", "燙髮", "染髮", "護髮", "養髮", "養護", "美甲", "造型", "美髮其他"]
#     figs_all = {}
#     for index, s in enumerate(services_list):
#         data_service = data.query("類別=='{}'".format(s))[["開單日期", "數量"]].set_index(["開單日期"])
#         length=len(data_service)/5
#         train=data_service.iloc[:-length]
#         test=data_service.iloc[-length:]




