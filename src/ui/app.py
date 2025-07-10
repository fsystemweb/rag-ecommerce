import streamlit as st
import time
import requests
from src.model_pipeline import generate_app_embeddings
from src.query_executor import process_ui_query

def on_input_change():
    user_input = st.session_state.user_input.strip()
    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        response = process_ui_query(user_input)
        st.session_state.chat_history.append({"role": "assistant", "message": response})
        st.session_state.user_input = ""  # Clear input

st.set_page_config(page_title="Creación de asistente para E-commerce", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = 1
if "ecommerce_data" not in st.session_state:
    st.session_state.ecommerce_data = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Navigation helper
def next_page():
    st.session_state.page += 1
    st.rerun()
# Page 1 - Paste Information
if st.session_state.page == 1:
    st.title("Paso 1: Pegue toda la información relevante de su e-commerce")

    st.markdown("Use la siguiente plantilla para agregar la información en la sección correcta, utilizando formato Markdown:")

    plantilla_url = "https://raw.githubusercontent.com/fsystemweb/rag-ecommerce/4d2cefae322befd8240f1ef1b8b6f12288d07199/data/template.md"
    response = requests.get(plantilla_url)
    response.raise_for_status()  # Check for HTTP errors
    file_content = response.content

    st.download_button(
        label="📄 Descargar plantilla",
        data=file_content,
        file_name="template.md",
        mime="text/markdown"
    )

    ecommerce_info = st.text_area("Pegue la información aquí", height=400)

    if st.button("Siguiente"):
        if ecommerce_info.strip() == "":
            st.warning("Por favor, ingrese la información antes de continuar.")
        else:
            st.session_state.ecommerce_data = ecommerce_info
            next_page()

# Page 2 - Loading and Embedding
elif st.session_state.page == 2:
    st.title("Paso 2: Creando modelo")

    with st.spinner("Creando el modelo y generando embeddings..."):
        generate_app_embeddings(st.session_state.ecommerce_data)
        st.session_state.model_built = True

    st.success("Embeddings completados.")
    if st.button("Siguiente"):
        next_page()

# Page 3 - Chatbot
elif st.session_state.page == 3:
    st.title("Paso 3: Hable con su asistente")
    st.markdown("🧠 Su asistente e-commerce está listo. Pregunte sobre su negocio online basándose en la información compartida.")

    # Inject CSS to style messages
    st.markdown(
        """
        <style>
        .user-message {
            text-align: left;
            background-color: #daf1da;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 90%;
        }
        .assistant-message {
            text-align: right;
            background-color: #e1eaff;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 90%;
            margin-left: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{chat["message"]}</div>', unsafe_allow_html=True)

    # Initialize input state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Input with on_change handler
    st.text_input("Realice una pregunta...", key="user_input", on_change=on_input_change)
