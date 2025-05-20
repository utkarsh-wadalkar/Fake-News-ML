from flask import Flask, render_template, request, jsonify
import re  # for cleaning
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained models and vectorizer
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

lr_model = model_data['lr']
gbc_model = model_data['gbc']
rfc_model = model_data['rfc']
vectorizer = model_data['vectorizer']

# Text cleaning function
def cleantext(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# Route to serve the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle news verification with single model (RFC)
@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    news_text = data.get('news_text', '')
    if not news_text:
        return jsonify({'error': 'No news text provided'}), 400

    cleaned_text = cleantext(news_text)

    # Use loaded vectorizer to transform input
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict using RFC model
    prediction = rfc_model.predict(vectorized_text)[0]

    # For confidence, if model supports predict_proba
    confidence = None
    if hasattr(rfc_model, 'predict_proba'):
        proba = rfc_model.predict_proba(vectorized_text)[0]
        confidence = round(max(proba) * 100, 2)

    # Prepare response
    pred_label = 'Fake' if prediction == 0 else 'Real'
    explanation = ''
    if pred_label == 'Fake':
        explanation = 'This news contains patterns commonly found in misinformation campaigns.'
    else:
        explanation = 'This news appears to be from credible sources and follows factual patterns.'

    return jsonify({
        'prediction': pred_label,
        'confidence': confidence,
        'explanation': explanation
    })

# Renamed route function to avoid endpoint conflict
@app.route('/manual_test', methods=['POST'])
def manual_test_route():
    data = request.get_json()
    news_text = data.get('news_text', '')
    if not news_text:
        return jsonify({'error': 'No news text provided'}), 400

    cleaned_text = cleantext(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    pred_lr = lr_model.predict(vectorized_text)[0]
    pred_gbc = gbc_model.predict(vectorized_text)[0]
    pred_rfc = rfc_model.predict(vectorized_text)[0]

    def output_label(n):
        return "Fake News" if n == 0 else "Real News"

    return jsonify({
        'LR_Prediction': output_label(pred_lr),
        'GBC_Prediction': output_label(pred_gbc),
        'RFC_Prediction': output_label(pred_rfc)
    })

if __name__ == '__main__':
    app.run(debug=True)