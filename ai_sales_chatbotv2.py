import os
import re
import json
import streamlit as st
import pandas as pd
from openai import OpenAI

# -----------------------------
# 🔐 Setup OpenAI API Key Securely
# -----------------------------
# Recommended: store in .streamlit/secrets.toml like:
# [general]
# OPENAI_API_KEY = "sk-xxxxxxxx"
# Then access via st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# 📂 Load Sales Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Sales.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

df = load_data()

FILTER_COLUMNS = ['region', 'order priority', 'country', 'item type', 'sales channel']

# -----------------------------
# ⚙️ Helper Functions
# -----------------------------
def get_numeric_columns(df):
    numeric_cols = ['units sold', 'unit price', 'unit cost',
                    'total revenue', 'total cost', 'total profit']
    return [c for c in numeric_cols if c in df.columns]

def apply_filters(df, filters):
    filtered_df = df.copy()
    for col, val in filters.items():
        if col in df.columns:
            filtered_df = filtered_df[filtered_df[col].str.lower() == val.lower()]
    return filtered_df

# -----------------------------
# 🧠 LLM-Powered Query Understanding
# -----------------------------
def interpret_query_with_llm(query, df):
    """
    Ask LLM to extract multiple metrics and filters.
    Only schema (column names) is shared — NOT data.
    """
    schema_description = f"""
    The dataset has the following columns:
    {', '.join(df.columns)}.
    Numeric fields: {', '.join(get_numeric_columns(df))}.
    """

    prompt = f"""
    You are a data assistant. Given a user's question and dataset schema,
    identify:
    1. The numeric metrics to aggregate (e.g., total profit, total revenue, units sold) — can be multiple.
    2. The filters (column name and value pairs) from the question.

    Return JSON **only** in this format:
    {{
      "metrics": ["<metric1>", "<metric2>", ...],
      "filters": {{
         "<column>": "<value>"
      }}
    }}

    If the question is not related to the data, return:
    {{
      "metrics": [],
      "filters": {{}}
    }}

    Data schema: {schema_description}
    User question: "{query}"
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0
    )

    try:
        text = response.output_text
        json_part = re.search(r"\{.*\}", text, re.S)
        if json_part:
            return json.loads(json_part.group())
    except Exception:
        pass

    return {"metrics": [], "filters": {}}

# -----------------------------
# 📊 Compute Results for Multiple Metrics
# -----------------------------
def compute_result(df, metrics, filters):
    numeric_cols = get_numeric_columns(df)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        return f"⚠️ No data matched for filters: {filters}"

    results = []
    for metric in metrics:
        if metric.lower() in numeric_cols:
            total = pd.to_numeric(filtered_df[metric.lower()], errors='coerce').sum()
            results.append(f"**{metric.title()}**: {total:,.2f}")
        else:
            results.append(f"❌ Unknown metric: {metric}")

    return "✅ Results:\n" + "\n".join(results)

# -----------------------------
# 💬 Streamlit Chat UI
# -----------------------------
st.set_page_config(page_title="AI Powered Chatbot", page_icon="🤖")
st.title("🤖 AI-Powered Chatbot")
st.markdown("Ask me anything about your data!")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask your question about data...")

if query:
    st.session_state.history.append({"role": "user", "content": query})

    parsed = interpret_query_with_llm(query, df)
    metrics = parsed.get("metrics", [])
    filters = parsed.get("filters", {})

    if not metrics:
        response = (
            "🤔 That question doesn’t seem related to the sales data. "
            "Please ask about profit, revenue, cost, or units sold."
        )
    else:
        response = compute_result(df, metrics, filters)

    st.session_state.history.append({"role": "assistant", "content": response})
    st.rerun()
