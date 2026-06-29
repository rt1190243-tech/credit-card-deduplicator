from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        amount = float(request.form["amount"])
        time = float(request.form["time"])

        if amount > 1000:
            result = "Suspicious Transaction Detected"
        else:
            result = "Safe Transaction"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)