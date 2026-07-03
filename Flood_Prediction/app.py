from flask import Flask, render_template, request
import numpy as np
import joblib

# -------------------------------
# Create Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Model and Scaler
# -------------------------------
try:
    model = joblib.load("floods.save")
    scaler = joblib.load("scaler.save")
except Exception as e:
    print("Error loading model or scaler:", e)
    exit()

# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -------------------------------
# Prediction Page
# -------------------------------
@app.route("/predict")
def predict():
    return render_template("predict.html")


# -------------------------------
# Prediction Result
# -------------------------------
@app.route("/result", methods=["POST"])
def result():
    print("Step 1: Request received")

    Temp = float(request.form["Temp"])
    Humidity = float(request.form["Humidity"])
    Cloud_Cover = float(request.form["Cloud_Cover"])
    ANNUAL = float(request.form["ANNUAL"])
    Jan_Feb = float(request.form["Jan_Feb"])
    Mar_May = float(request.form["Mar_May"])
    Jun_Sep = float(request.form["Jun_Sep"])
    Oct_Dec = float(request.form["Oct_Dec"])
    avgjune = float(request.form["avgjune"])
    sub = float(request.form["sub"])

    print("Step 2: Form data read")

    data = np.array([[
        Temp,
        Humidity,
        Cloud_Cover,
        ANNUAL,
        Jan_Feb,
        Mar_May,
        Jun_Sep,
        Oct_Dec,
        avgjune,
        sub
    ]])

    print("Step 3: Data created")

    data = scaler.transform(data)

    print("Step 4: Data scaled")

    prediction = model.predict(data)

    print("Step 5: Prediction =", prediction)

    if prediction[0] == 1:
        result = "⚠️ Flood Chance Detected"
    else:
        result = "✅ No Flood Chance"

    return render_template("result.html", prediction=result)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
