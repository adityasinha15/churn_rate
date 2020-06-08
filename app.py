import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf



app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')


graph = tf.get_default_graph()

@app.route('/predict',methods=['POST'])
def predict():
    dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
    )
    X = transformer.fit_transform(X.tolist())
    X = X.astype('float64')
    X = X[:, 1:]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    gender = request.form['gender']
    country = request.form['country']
    credit_card = request.form['credit_card']
    active = request.form['active']
    credit_score = request.form['credit_score']
    age = request.form['age']
    tenure = request.form['tenure']
    products = request.form['products']
    salary = request.form['salary']
    balance = request.form['balance']
    if country == 'france':
        country1 = 0
        country2 = 0
    if country == 'spain':
        country1 = 0
        country2 = 1
    if country == 'germany':
        country1 = 1
        country2 = 0

    
    
    
    final_features = sc.transform(np.array([[country1,country2, credit_score, gender, age, tenure, balance, products, credit_card, active, salary ]]))
    
    global graph 
    with graph.as_default():
        output = model.predict(final_features)
    if output > 0.5:
        result = 'Yes with probability {}%'.format(output[0][0]*100)
    else:
        result = 'No with probability {}%'.format(output[0][0]*100)
    return render_template('index.html', prediction_text='{}'.format(result))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
