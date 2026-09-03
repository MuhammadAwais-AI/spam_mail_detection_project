import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")

st.title("📧 Spam Mail Detector")
st.write("Classify SMS messages as **SPAM** or **HAM** using NLP + Logistic Regression")

# Input
user_input = st.text_area("Enter your message here:", height=150)

if st.button("Predict"):
    if user_input:
        # Transform and predict
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        prob = model.predict_proba(input_tfidf)[0][1]
        
        if prediction == 1:
            st.error(f"🚨 SPAM - {prob:.2%} confidence")
        else:
            st.success(f"✅ HAM - {(1-prob):.2%} confidence")
    else:
        st.warning("Please enter a message first")

st.markdown("---")
st.markdown("Built with Scikit-learn | Accuracy: 96.77%")