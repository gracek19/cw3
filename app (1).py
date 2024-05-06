import numpy as np
from flask import Flask, request, jsonify
import random

class Perceptron():
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] 
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)

# Create a flask
app = Flask(__name__)

# Initialize Perceptron model
model = Perceptron()

# Create an API end point
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Parse parameters from URL
        sepal_length = float(request.args.get('sl'))
        petal_length = float(request.args.get('pl'))
    elif request.method == 'POST':
        # Parse JSON data
        data = request.get_json(force=True)
        sepal_length = float(data.get('sl'))
        petal_length = float(data.get('pl'))
    else:
        return jsonify(error="Unsupported request method")

    features = np.array([sepal_length, petal_length]).reshape(1, -1)
        
    # Fit the model with some dummy data
    X_dummy = np.array([[5.1, 1.4], [4.9, 3.0], [6.0, 1.4], [6.6, 2.9]])
    y_dummy = np.array([1, -1, 1, -1])
    model.fit(X_dummy, y_dummy)
    
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    
    # Return a json object containing the features and prediction
    return jsonify(features=features.tolist(), predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
