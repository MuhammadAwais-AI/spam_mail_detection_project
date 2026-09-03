import streamlit as st
import joblib
import streamlit as st
import streamlit as st

st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)
st.image("https://cdn-icons-png.flaticon.com/512/542/542638.png", width=80)
st.title("Spam Mail Detector")
st.write("Classify SMS messages as SPAM or HAM using NLP + Logistic Regression")
# Custom CSS for button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;  /* Green */
    color: white;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 10px 24px;
}
div.stButton > button:hover {
    background-color: #45a049;  /* Darker green on hover */
    color: white;
}
</style>
""", unsafe_allow_html=True)
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
