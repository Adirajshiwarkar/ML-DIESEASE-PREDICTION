# Disease Prediction with Machine Learning Models

This project leverages various machine learning models to predict diseases based on input symptoms. The dataset includes numerous health indicators and the corresponding disease labels. The solution implements multiple classifiers and visualizes their performance and feature importance.

## Features
- **Data Preprocessing:** Handles data preparation by splitting the dataset into training and testing sets.
- **Model Training:** Supports Naive Bayes, Random Forest, Logistic Regression, and Decision Tree classifiers.
- **Model Evaluation:** Evaluates models using accuracy scores.
- **Model Persistence:** Saves trained models using joblib for future use.
- **Visualizations:**
  - Comparison of model performance using accuracy scores.
  - Top 10 important features visualization for the Random Forest classifier.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd disease-prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script to train models and generate visualizations:
   ```bash
   python disease_prediction.py
   ```
2. The script will display:
   - Accuracy scores for each classifier.
   - A bar chart comparing model performance.
   - A horizontal bar chart showing the top 10 important features for the Random Forest model.



```

## Project Structure
```
.
├── disease_prediction.py  # Main script
├── model                  # Directory for saved models
└── requirements.txt       # Dependency file
```

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

## License
This project is licensed under the MIT License.

## Contributions
Feel free to contribute by submitting issues or pull requests.

## Contact
For any inquiries, please contact [your_email@example.com].
