from flask import Flask, render_template
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='templates')

# Import and initialize routes
from routes.prediction import init_prediction_routes
from routes.optimization import init_optimization_routes
from routes.explanation import init_explanation_routes

init_prediction_routes(app)
init_optimization_routes(app)
init_explanation_routes(app)
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
