import streamlit as st
from landing_page import show_landing_page
from app_stock_prediction import show_stock_prediction_page
from app_esg import show_esg_page
from chatbot import show_chatbot_page
from article_suggestion import show_article_suggestions_page
from app_stock_price_prediction import show_stock_price_prediction_page
# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Landing Page", "Stock Prediction", "Stock Price Prediction", "ESG Data", "Chatbot", "Article Suggestions"]
page = st.sidebar.radio("Go to", pages)

# Page routing
if page == "Landing Page":
    show_landing_page()
elif page == "Stock Prediction":
    show_stock_prediction_page()
elif page == "Stock Price Prediction":
    show_stock_price_prediction_page()
elif page == "ESG Data":
    show_esg_page()
elif page == "Chatbot":
    show_chatbot_page()
elif page == "Article Suggestions":
    show_article_suggestions_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("© 2024 Sustainable Investing Management. All rights reserved.")
