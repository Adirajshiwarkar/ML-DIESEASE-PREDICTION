# ML-DIESEASE-PREDICTION
This project is a machine learning-based system designed to predict potential diseases based on user-provided symptoms. It leverages various classification models, such as Naive Bayes, Random Forest, and Logistic Regression, to provide accurate predictions. The system is designed to act as a smart diagnostic assistant for healthcare applications.

Features
Predicts diseases based on 132 symptoms.
 Supports multiple machine learning models (Naive Bayes, Random Forest, Decision Tree, and Logistic Regression).
High prediction accuracy (100% on test data).
 User-friendly interface to input symptoms and receive disease predictions.


Tech Stack
Programming Language: Python
Libraries: Pandas, NumPy, scikit-learn, Joblib
Visualization: Matplotlib
Model Serialization: Joblib
Deployment-ready: Flask (for web application setup)
How It Works
Users input symptoms via a web interface (Flask).
The system encodes symptoms into a binary vector.
Trained machine learning models predict the most likely disease.
The result is displayed to the user.
Models Used
Naive Bayes: Simple yet effective for disease classification.
Random Forest: Provides robust results through ensemble learning.
Logistic Regression: Effective for binary classification problems.
Decision Tree: Captures complex relationships between features.
Dataset
The project uses a dataset containing 132 symptoms and corresponding diseases for training and testing the models.




License
This project is open-source and available under the MIT License.

