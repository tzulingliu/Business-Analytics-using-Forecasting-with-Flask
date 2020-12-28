from flask import Flask, render_template, url_for
import ExploratoryDataAnalysis as eda
import pandas as pd


app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/store_<int:store_num>")
def stores(store_num):
    data, bar, box, heatmap, sersfor, shift_name = eda.display_time_series(store_num, "AD")
    return render_template("store_layout.html".format(store_num), title="Store "+str(store_num),
                           graph=data, store_num=int(store_num), shift_name=shift_name, bar=bar, box=box, heatmap=heatmap, sersfor=sersfor)

@app.route("/store_<int:store_num>/analysis/shfit_<string:shift>")
def analysis(store_num, shift):
    data, bar, box, heatmap, sersfor, shift_name = eda.display_time_series(store_num, shift)


    return render_template("store_layout.html", store_num=store_num, graph=data,
                           title="Store "+str(store_num), shift_name=shift_name, bar=bar, box=box, heatmap=heatmap, sersfor=sersfor)

if __name__== "__main__":
    app.run(debug=True)