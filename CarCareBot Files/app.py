import streamlit as st
from chatbot import get_answer, log_interaction
import traceback

st.set_page_config(page_title="CarCareBot Chat", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("CarCareBot Chat")
st.markdown("Ask a question about your car (engine, battery, towing, brakes, etc.)")

# User input
user_input = st.text_input("Type your message here:")

if st.button("Send") and user_input.strip():
    try:
        answer = get_answer(user_input)
        if not answer or "Error" in answer:
            answer = "Sorry, I couldn’t find information on that. Please try asking differently."
    except Exception as e:
        answer = f"An error occurred: {e}"
        print(traceback.format_exc())

    # Log and store chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "ai", "content": answer})
    log_interaction(user_input, answer)

# Display messages
for msg in st.session_state.messages:
    content = msg["content"].replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="
                background-color:#DCF8C6;
                padding:10px;
                border-radius:10px;
                margin:5px 0;
                max-width:70%;
                display:inline-block;
                float:left;
                clear:both;">
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background-color:#EAEAEA;
                padding:10px;
                border-radius:10px;
                margin:5px 0;
                max-width:70%;
                display:inline-block;
                float:right;
                clear:both;">
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
