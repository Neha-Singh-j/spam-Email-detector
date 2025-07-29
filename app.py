from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()
    if not message:
        return render_template("index.html", prediction=None)
    
    try:
        msg_vector = vectorizer.transform([message])
        prediction = model.predict(msg_vector)[0]
        return render_template("index.html", prediction="Spam" if prediction == 1 else "Ham")
    except:
        return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)