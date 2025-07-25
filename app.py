from flask import Flask, render_template

app = Flask(__name__)

# Model evaluation metrics (from user)
metrics = {
    "accuracy": 0.8571,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.0
}

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
