from flask import Flask, render_template, request

from finalModel import EMOTION_LABELS, predict

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    threshold = 0.3
    prediction = None
    error = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        threshold = request.form.get("threshold", "0.3").strip()

        try:
            threshold = float(threshold)
        except ValueError:
            error = "Threshold must be a number between 0 and 1."
            threshold = 0.3

        if error is None and not 0 <= threshold <= 1:
            error = "Threshold must be between 0 and 1."

        if error is None and not text:
            error = "Please enter some text to analyze."

        if error is None:
            prediction = predict(text, threshold)

    return render_template(
        "index.html",
        text=text,
        threshold=threshold,
        prediction=prediction,
        error=error,
        emotions=EMOTION_LABELS,
    )


if __name__ == "__main__":
    app.run(debug=True)
