import os
from typing import Optional

import requests
import streamlit as st

API_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000/ask")
EXAMPLE_PROMPTS = [
    "What is the expense ratio of HDFC Flexi Cap?",
    "What is the minimum SIP amount for HDFC Mid Cap?",
    "What is the exit load of HDFC Small Cap?",
]


st.set_page_config(page_title="Mutual Fund Facts-Only Assistant")

st.title("Mutual Fund Facts-Only Assistant")
st.subheader("Get citation-backed facts from official AMC / SEBI / AMFI pages.")
st.caption("Facts-only. No investment advice.")


def set_prompt(prompt: str):
    st.session_state["question_input"] = prompt


if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""


st.write("Try one of these example prompts:")
cols = st.columns(len(EXAMPLE_PROMPTS))
for col, prompt in zip(cols, EXAMPLE_PROMPTS):
    col.button(prompt, on_click=set_prompt, args=(prompt,))


question = st.text_input(
    "Your question",
    key="question_input",
    placeholder="Ask about expense ratios, SIP amounts, exit loads, etc.",
)

send_clicked = st.button("Send")

if send_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"question": question.strip()},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException:
            data = None
            st.error("Sorry, something went wrong while contacting the server.")

        if data:
            st.success(data.get("answer", "No answer returned."))
            source_url: Optional[str] = data.get("source")
            if source_url:
                st.markdown(f"**Source:** [{source_url}]({source_url})")
            last_updated = data.get("last_updated")
            if last_updated:
                st.caption(f"Last updated from sources: {last_updated}")


