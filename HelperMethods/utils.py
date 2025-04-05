import base64
import streamlit as st

@staticmethod
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp{{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            }}
             </style>
            """, unsafe_allow_html=True
        )


@staticmethod
def inject_styles():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Open Sans:wght@400;600&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"]  {
            font-family: 'Open Sans', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: white;
            margin-bottom: 30px;
            padding-left: 50px;
            font-family: Open Sans;
        }
        .centered-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .match-message {
            background-color: #1f4037;
            border-radius: 8px;
            color: white;
            font-size: 0.95em;
            text-align: center;
            display: inline-block;
            margin-top: 15px;
        }
        </style>
    """, unsafe_allow_html=True)
