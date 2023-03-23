from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("quality.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("quality.html")

@app.route("/predict",methods=["POST"])
def predict():
    fixed_acidity=float(request.form.get("fixed_acidity"))
    volatile_acidity=float(request.form.get("volatile_acidity"))
    citric_acid=float(request.form.get("citric_acid"))
    residual_sugar=float(request.form.get("residual_sugar"))
    chlorides=float(request.form.get("chlorides"))
    free_sulfur_dioxide=float(request.form.get("free_sulfur_dioxide"))
    total_sulfur_dioxide=float(request.form.get("total_sulfur_dioxide"))
    density=float(request.form.get("density"))
    pH=float(request.form.get("pH"))
    sulphates=float(request.form.get("sulphates"))
    alcohol=float(request.form.get("alcohol"))
    
    result=model.predict(np.array([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]]))
    
    if result[0]>=7:
        return render_template("Good.html")
    else:
        return render_template("Bad.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)