# app.py
import streamlit as st
import requests
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Eureka Assistant", layout="centered")
FASTAPI_BASE_URL = "https://transformers-kdk5.onrender.com"

# -------------------------------------
# SIDEBAR NAVIGATION + INSTRUCTIONS
# -------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🧠 Chat with Assistant", "📜 Chat History"])

with st.sidebar.expander("⚙️ Persistent Instructions", expanded=True):
    st.markdown("Add instructions that apply to all future queries.")
    new_instruction = st.text_input("e.g., 'From now on, show top 5 only'", key="new_instruction_input")

    if st.button("➕ Add Instruction", key="add_instruction_button"):
        if new_instruction.strip():
            try:
                response = requests.post(
                    f"{FASTAPI_BASE_URL}/instructions/add",
                    json={"instruction": new_instruction}
                )
                if response.status_code == 200:
                    st.success("✅ Instruction added.")
                    st.rerun()
                else:
                    st.error("❌ Failed to add instruction.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Instruction cannot be empty.")

    st.markdown("### 📌 Active Instructions:")
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/instructions")
        instructions = response.json().get("instructions", [])
        if instructions:
            for i, instr in enumerate(instructions, 1):
                st.markdown(f"**{i}.** {instr}")
        else:
            st.info("No instructions yet.")
    except Exception as e:
        st.error(f"Failed to fetch instructions: {e}")

    if st.button("🧹 Clear Instructions", key="clear_instruction_button"):
        try:
            requests.post(f"{FASTAPI_BASE_URL}/instructions/clear")
            st.success("✅ Instructions cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear instructions: {e}")

# -------------------------------------
# PAGE 1: Chat Assistant
# -------------------------------------
if page == "🧠 Chat with Assistant":
    st.title("🧠 Eureka Assistant")
    st.markdown("Ask anything related to your data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    user_input = st.chat_input("Type your question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/chat",
                        json={"messages": st.session_state.messages},
                        timeout=120
                    )
                    assistant_msg = response.json().get("message", {}).get("content", "⚠️ No response.")
                    df_json = response.json().get("dataframe", None)

                    # Display text response
                    st.markdown(assistant_msg)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

                    # If there's a dataframe, visualize it
                    if df_json:
                        df_result = pd.read_json(df_json, orient="split")

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# -------------------------------------
# PAGE 2: Chat History Viewer
# -------------------------------------
elif page == "📜 Chat History":
    st.title("📜 Chat History")
    st.markdown("View all recent conversations between you and the assistant.")

    try:
        with st.spinner("Fetching chat history..."):
            response = requests.get(f"{FASTAPI_BASE_URL}/chat/history", timeout=20)
            if response.status_code == 200:
                history = response.json().get("chat_history", "")
                st.markdown(history)
            else:
                st.error("❌ Failed to load chat history from the backend.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

    if st.button("🔄 Refresh"):
        st.rerun()
