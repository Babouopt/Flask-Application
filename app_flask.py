import pickle
from flask import Flask, jsonify
import numpy as np
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


app = Flask(__name__)

fichier_pickle = r"model.pickle"

with open(fichier_pickle, 'rb') as fichier_grid_search_logit:
        grid_search_logit = pickle.load(fichier_grid_search_logit)

@app.route('/')
def welcome():
    return "Bienvenue dans l'API de prédiction des tags"

@app.route('/predict_tags/<string:question>')
def predict_tags(question):
   
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extraction des représentations vectorielles pour les phrases
    Question_sbert = model.encode([question], convert_to_tensor=True)

    # Prediction
    y_test_predicted_labels_sbert = grid_search_logit.predict(Question_sbert)

    fichier_pickle = r"MultiLabelBinarizer.pickle"
    with open(fichier_pickle, 'rb') as fichier_MultiLabelBinarizer:
        multilabel_binarizer = pickle.load(fichier_MultiLabelBinarizer)

    # transformation inverse
    y_test_pred_inversed = multilabel_binarizer \
        .inverse_transform(y_test_predicted_labels_sbert)

    return jsonify(y_test_pred_inversed)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
