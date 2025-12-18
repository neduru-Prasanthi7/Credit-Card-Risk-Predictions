# Credit-Card-Risk-Predictions
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Default Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            color: #333;
        }
        code {
            background-color: #eee;
            padding: 2px 6px;
            border-radius: 4px;
        }
        pre {
            background-color: #eee;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        img {
            max-width: 600px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .section {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>

    <h1>Credit Card Default Prediction Project</h1>

    <div class="section">
        <h2>Project Overview</h2>
        <p>
            The <strong>Credit Card Default Prediction System</strong> predicts the likelihood of a credit card user defaulting on payment.
            The project is implemented using <strong>Python</strong> and <strong>Object-Oriented Programming (OOP)</strong> concepts including classes, objects, and functions.
        </p>
        <p>
            It includes <strong>data preprocessing, feature selection, model training, evaluation, and deployment</strong> via a web interface.
        </p>
    </div>

    <div class="section">
        <h2>Features</h2>
        <ul>
            <li><strong>Data Preprocessing</strong>
                <ul>
                    <li>Removed <strong>null values</strong>.</li>
                    <li>Handled <strong>missing values</strong> using random sampling.</li>
                    <li>Detected and treated <strong>outliers</strong>.</li>
                    <li>Removed irrelevant features using <strong>feature selection</strong>.</li>
                    <li>Balanced the dataset to handle <strong>class imbalance</strong>.</li>
                </ul>
            </li>
            <li><strong>Machine Learning Models</strong>
                <ul>
                    <li>K-Nearest Neighbors (KNN)</li>
                    <li>Naive Bayes</li>
                    <li>Logistic Regression</li>
                    <li>Decision Tree</li>
                    <li>Random Forest</li>
                    <li>AdaBoost</li>
                </ul>
            </li>
            <li><strong>Model Evaluation</strong>
                <ul>
                    <li>Test Accuracy</li>
                    <li>Classification Report</li>
                    <li>Confusion Matrix</li>
                    <li>ROC Curve & AUC Score</li>
                </ul>
            </li>
            <li><strong>Best Model:</strong> Logistic Regression (based on ROC-AUC and overall performance)</li>
            <li><strong>Deployment:</strong> Model saved using pickle, web interface via Flask.</li>
        </ul>
    </div>

    <div class="section">
        <h2>Technologies Used</h2>
        <ul>
            <li>Python: pandas, numpy, scikit-learn, matplotlib, seaborn</li>
            <li>Machine Learning: Classification algorithms</li>
            <li>Web Deployment: Flask</li>
            <li>Model Serialization: Pickle</li>
        </ul>
    </div>

    <div class="section">
        <h2>Sample Screenshots</h2>
        <p>Web Interface</p>
        <img src="screenshots/web_interface.png" alt="Web Interface">
        <p>Confusion Matrix</p>
        <img src="screenshots/confusion_matrix.png" alt="Confusion Matrix">
        <p>ROC Curve</p>
        <img src="screenshots/roc_curve.png" alt="ROC Curve">
        <p><em>(Replace with actual images from your project)</em></p>
    </div>

    <div class="section">
        <h2>How to Run</h2>
        <ol>
            <li>Clone the repository:
                <pre>git clone &lt;repository_url&gt;</pre>
            </li>
            <li>Install dependencies:
                <pre>pip install -r requirements.txt</pre>
            </li>
            <li>Run the Flask app:
                <pre>python app.py</pre>
            </li>
            <li>Open your browser and go to:
                <pre>http://127.0.0.1:5000/</pre>
            </li>
            <li>Input user data to get <strong>credit card default prediction</strong>.</li>
        </ol>
    </div>

    <div class="section">
        <h2>Results</h2>
        <ul>
            <li>Logistic Regression provided the best prediction accuracy.</li>
            <li>Confusion matrix and ROC-AUC curve confirm the model's reliability.</li>
            <li>Web interface allows real-time predictions using the trained model.</li>
        </ul>
    </div>

    <div class="section">
        <h2>File Structure</h2>
        <pre>
Credit_Card_Default_Prediction/
│
├── app.py                     # Flask app for deployment
├── credit_model.pkl            # Trained Logistic Regression model
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web page for user input
├── static/
│   └── style.css               # Styling for the web interface
├── data/
│   └── credit_data.csv         # Dataset
├── src/
│   ├── preprocessing.py        # Data cleaning, missing value handling, outlier treatment
│   ├── feature_selection.py
│   ├── model_training.py
│   └── evaluation.py
└── README.html
        </pre>
    </div>

</body>
</html>
