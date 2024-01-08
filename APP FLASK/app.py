import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static')

models = pickle.load(open('knn.pkl', 'rb'))
knn_Manhattan = models['knn_Manhattan']
knn_Euclidean = models['knn_Euclidean']
knn_Minkowski = models['knn_Minkowski']

# @app.route("/")
# def Home():
#     return render_template('index.html')

# @app.route("/Charts")
# def Charts():
#     return render_template('linechart.html')

@app.route("/")
def form():
    return render_template('website.html')

@app.route('/', methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    result_manhattan = knn_Manhattan.predict(features)
    result_euclidean = knn_Euclidean.predict(features)
    result_minkowski = knn_Minkowski.predict(features)

    return render_template("website.html", prediction_manhattan = "Hasil Prediksi KNN Manhattan : {}".format(result_manhattan[0]), 
    prediction_euclidean = "Hasil Prediksi KNN Euclidean : {}".format(result_euclidean[0]), 
    prediction_minkowski = "Hasil Prediksi KNN Minkowski : {}".format(result_minkowski[0]))

if __name__ =="__main__":
    app.run(debug=True)
