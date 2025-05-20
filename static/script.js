document.getElementById('verify-btn').addEventListener('click', manualTest);

const express = require('express');
const app = express();
const port = 5000;

app.get('/manual_test', (req, res) => {
  // Add the Access-Control-Allow-Origin header to allow requests from http://127.0.0.1:5000
  res.setHeader('Access-Control-Allow-Origin', 'http://127.0.0.1:5000');
  res.json({ message: 'Manual test successful' });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

async function manualTest() {
    const newsText = document.getElementById('news-input').value.trim();
    const resultContainer = document.getElementById('result-container');
    
    if (!newsText) {
        alert('Please enter news text to verify');
        return;
    }
    
    // Show loading state
    resultContainer.innerHTML = '<p>Analyzing news... Please wait.</p>';
    resultContainer.style.display = 'block';
    resultContainer.style.backgroundColor = '#f8f9fa';
    
    try {
        const response = await fetch('http://localhost:5000/manual_test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ news_text: newsText }),
        });
        
        const data = await response.json();
        
        resultContainer.innerHTML = `
            <h2>Verification Results</h2>
            <p><strong>News:</strong> ${newsText.substring(0, 100)}${newsText.length > 100 ? '...' : ''}</p>
            <p><strong>Logistic Regression Prediction:</strong> ${data.LR_Prediction}</p>
            <p><strong>Gradient Boosting Prediction:</strong> ${data.GBC_Prediction}</p>
            <p><strong>Random Forest Prediction:</strong> ${data.RFC_Prediction}</p>
        `;
        resultContainer.style.backgroundColor = '#e8f5e9';
    } catch (error) {
        console.error('API Error:', error);
        resultContainer.innerHTML = '<p style="color: red;">Error: Unable to get prediction from the server.</p>';
        resultContainer.style.backgroundColor = '#ffebee';
    }
}
