# app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def visualization_tool(user_query, df_result):
    chart_type = classify_visualization_type(user_query, df_result)
    print(f"🔍 LLM-chosen chart type: {chart_type}")

    # Clean column names
    df_result.columns = df_result.columns.str.strip()
    print("📊 Columns in DataFrame:")
    print(df_result.dtypes)
    print(df_result.head())

    # Convert numeric-looking strings (with commas) to floats
    for col in df_result.columns:
        if df_result[col].dtype == 'object':
            df_result[col] = df_result[col].str.replace(',', '').str.strip()
            df_result[col] = df_result[col].apply(
                lambda x: float(re.findall(r"\d+\.?\d*", x)[0])
                if isinstance(x, str) and re.findall(r"\d+\.?\d*", x)
                else np.nan
            )
    # Check for numeric columns
    numeric_df = df_result.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("⚠️ Error: No numeric data available to plot.")
        return

    # Auto-select x_col (categorical) and y_col (numeric) for plotting
    x_col = None
    y_col = None

    # Prefer known numeric metrics as y_col
    preferred_y_cols = [
        "Invoice Net value", "Billing Quantity in Sales Units",
        "Sales Invoice Price(USD/MT)", "Quantity MT"
    ]
    for col in preferred_y_cols:
        if col in df_result.columns and pd.api.types.is_numeric_dtype(df_result[col]):
            y_col = col
            break

    # Fallback: first numeric column
    if y_col is None:
        numeric_cols = df_result.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            y_col = numeric_cols[0]

    # Pick first non-numeric column as x_col
    non_numeric_cols = df_result.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        x_col = non_numeric_cols[0]

    # Final check
    if x_col is None or y_col is None:
        st.warning("⚠️ Could not find suitable x and y columns for charting.")
        return

    print(f"x_col: {x_col}, y_col: {y_col}")

    fig, ax = plt.subplots(figsize=(12, 8))

    try:
        if chart_type == "bar" and y_col and pd.api.types.is_numeric_dtype(df_result[y_col]):
            df_result.plot(kind='bar', x=x_col, y=y_col, ax=ax)
        elif chart_type == "line" and y_col and pd.api.types.is_numeric_dtype(df_result[y_col]):
            df_result.plot(kind='line', x=x_col, y=y_col, ax=ax)
        elif chart_type == "scatter" and y_col and pd.api.types.is_numeric_dtype(df_result[y_col]):
            df_result.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
        elif chart_type == "pie" and y_col and pd.api.types.is_numeric_dtype(df_result[y_col]):
            df_result = df_result[df_result[y_col] > 0]
            if len(df_result) > 15:
                df_result = df_result.nlargest(15, y_col)
            df_result.set_index(x_col).plot(kind='pie', y=y_col, ax=ax, labels=None, autopct='%1.1f%%', legend=False)
            ax.legend(labels=df_result[x_col], loc='center left', bbox_to_anchor=(1.0, 0.5), title=x_col)
        elif chart_type == "histogram" and y_col and pd.api.types.is_numeric_dtype(df_result[y_col]):
            df_result[y_col].plot(kind='hist', ax=ax, bins=10)
        else:
            st.warning("⚠️ No suitable numeric columns found for the selected chart type.")
            return

        ax.set_title(f"LLM: {chart_type} chart")
        if chart_type != "pie":
            ax.get_yaxis().set_major_formatter(
                mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Visualization failed: {e}")




# NOTE: Implement classify_visualization_type using a local heuristic or API call
def classify_visualization_type(user_query, df):
    # basic heuristic example, replace this with API or LLM call
    if "trend" in user_query.lower() or "over time" in user_query.lower():
        return "line"
    if "compare" in user_query.lower():
        return "bar"
    if "distribution" in user_query.lower():
        return "histogram"
    if "proportion" in user_query.lower() or "share" in user_query.lower():
        return "pie"
    return "none"

st.set_page_config(page_title="Eureka Assistant", layout="centered")
FASTAPI_BASE_URL = "http://localhost:8000"

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
                        fig = visualization_tool(user_input, df_result)
                        if fig:
                            st.markdown("### 📊 Visual Insight")
                            st.pyplot(fig)

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
