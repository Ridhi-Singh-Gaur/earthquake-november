from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("earthquake_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏠 Homepage
@app.route('/')
def home():
    return render_template('index.html')

# 🌍 Predict page (GET)
@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

# ⚙️ Predict result (POST)
@app.route('/predict', methods=['POST'])
def predict_result():
    try:
        # Get form data
        month = int(request.form['month'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        depth = float(request.form['depth'])
        magType = int(request.form['magType'])
        nst = float(request.form['nst'])
        gap = float(request.form['gap'])
        dmin = float(request.form['dmin'])
        rms = float(request.form['rms'])
        net = int(request.form['net'])
        place = int(request.form['place'])

        # Prepare data
        features = np.array([[month, latitude, longitude, depth, magType, nst, gap, dmin, rms, net, place]])

        # Predict
        prediction = model.predict(features)[0]
        result = "⚠️ Earthquake likely (1)" if prediction == 1 else "✅ No earthquake (0)"

        return render_template('predict.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error: {str(e)}")


# 🌐 Other pages
@app.route('/recent')
def recent():
    return render_template('recent.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/fun-facts')
def fun_facts():
    return render_template('fun-facts.html')


if __name__ == '__main__':
    app.run(debug=True)
