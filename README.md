# 📧 Spam Mail Detector

An end-to-end NLP project to classify SMS messages as **Spam** or **Ham** using Machine Learning and deployed as an interactive Streamlit web app.

## 🔍 Overview
Spam messages are a major problem for mobile users. This project builds a machine learning model to automatically detect spam SMS messages. 
The app takes any text input and predicts whether it's spam with a confidence score.

**Live Demo**: [Add your Streamlit Cloud link here after deployment]

## 📊 Dataset
- **Source**: SMS Spam Collection Dataset from UCI
- **Size**: 5,572 messages
- **Classes**: 
    - Ham: 4,827 messages `86.6%`
    - Spam: 747 messages `13.4%`
- **Features**: Raw SMS text + Label

## 🛠️ Tech Stack
| Category | Tools Used |
| --- | --- |
| **Language** | Python 3.10 |
| **ML/NLP** | Scikit-learn, TF-IDF Vectorizer, Logistic Regression |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Model Saving** | Joblib |

## ⚙️ Approach
1. **Data Preprocessing**: Removed duplicates, handled null values, text cleaning
2. **Feature Extraction**: Converted text to numerical features using `TF-IDF Vectorizer`
3. **Model Training**: Trained `Logistic Regression` classifier on 80% of data
4. **Evaluation**: Tested on 20% of data using Accuracy, Precision, Recall, F1-Score
5. **Confusion Matrix**: Analyzed model performance on Spam vs Ham
6. **Deployment**: Saved model with Joblib and built interactive UI with Streamlit

## 📈 Results
The model achieved strong performance on the test set:

- **Accuracy**: 96.77%
- **Precision for Spam**: 1.00
- **Recall for Spam**: 0.78
- **F1-Score**: 0.88

The model has 100% precision for spam, meaning when it predicts "Spam", it's almost always correct.

## 🎯 Key Takeaways
- TF-IDF + Logistic Regression is still very effective for text classification tasks
- Handling class imbalance is important in spam detection
- Model interpretability and high precision are crucial for spam filters to avoid false positives
- Streamlit makes it easy to deploy ML models for non-technical users to test

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/MuhammadAwais-AI/spam_mail_detection_project.git
cd spam_mail_detection_project
```
## Author: Muhammad Awais  
## LinkedIn: https://www.linkedin.com/in/muhammad-awais-khan-95559041a?utm_source=share_via&utm_content=profile&utm_medium=member_android
Feel free to connect with me and try the live demo!
