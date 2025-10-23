# streamlit_app.py
import streamlit as st
from prompts import SYSTEM_PROMPT, GENERATION_TEMPLATE, get_qa_generation_prompt # Added get_qa_generation_prompt
from utils import call_ollama, safe_parse_json_array, testcases_to_dataframe, export_testcases_json, export_testcases_csv
import json
import os
from datetime import datetime

st.set_page_config(page_title="QA Test Case Generation Agent", layout="wide")

st.title("QA Test Case Generation Agent — Streamlit + Ollama Mistral")
st.markdown("Paste requirement(s) or a user story, then click **Generate Test Cases**.")

# Input panel
with st.sidebar:
    st.header("Input")
    requirement_text = st.text_area("Requirement / User story", height=250, placeholder="As a user, I want to ...")
    uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
    model_name = st.text_input("Model name (local Ollama)", value="mistral")
    if uploaded is not None:
        try:
            t = uploaded.read().decode("utf-8")
            # Only append if requirement_text already has content, otherwise replace
            if requirement_text.strip():
                requirement_text = requirement_text + "\n\n" + t
            else:
                requirement_text = t
        except Exception:
            st.error("Couldn't read uploaded file as text.")
    st.markdown("---")
    st.markdown("Advanced / Prompt tweaks")
    additional_instructions = st.text_area("Extra instructions for generator (optional)", height=100,
                                          placeholder="e.g., Focus on security and boundary conditions.")
    st.markdown("---")
    st.write("Ollama endpoint assumed: http://localhost:11434/api/generate")
    st.write("Make sure Ollama is running and model is pulled (e.g., `ollama pull mistral`).")

# Buttons
col1, col2, col3 = st.columns([1,1,1])
with col1:
    generate_btn = st.button("Generate Test Cases")
with col2:
    refine_btn = st.button("Refine / Regenerate")
with col3:
    export_json_btn = st.button("Export JSON")
    export_csv_btn = st.button("Export CSV")

# Output area
if "last_testcases" not in st.session_state:
    st.session_state["last_testcases"] = None
if "last_raw" not in st.session_state:
    st.session_state["last_raw"] = ""

# Removed the 'build_prompt' function from here, as it's now in prompts.py

if generate_btn:
    if not requirement_text.strip():
        st.sidebar.error("Please provide a requirement or upload a text file.")
    else:
        st.info("Generating test cases — calling local Ollama model...")
        # Use the centralized prompt generation function
        prompt = get_qa_generation_prompt(requirement_text.strip(), additional_instructions.strip())
        success, resp = call_ollama(prompt, model=model_name)
        if not success:
            st.error(resp)
        else:
            st.session_state["last_raw"] = resp
            ok, parsed_or_err = safe_parse_json_array(resp)
            if ok:
                st.session_state["last_testcases"] = parsed_or_err
                st.success(f"Parsed {len(parsed_or_err)} test case(s).")
            else:
                st.warning("Couldn't parse model output as JSON array. Showing raw output for manual copy/edit.")
                st.session_state["last_testcases"] = None
                st.text_area("Raw model output", value=resp, height=300, key="raw_output")

# Show last parsed testcases
if st.session_state["last_testcases"]:
    st.subheader("Generated Test Cases (structured)")
    df = testcases_to_dataframe(st.session_state["last_testcases"])
    st.dataframe(df, use_container_width=True)

    # Show JSON in a text area for user edits / iterative refinement
    st.subheader("Edit JSON (for refinement)")
    editable_json = st.text_area("Editable JSON", value=json.dumps(st.session_state["last_testcases"], indent=2, ensure_ascii=False), height=300)

    # If user clicked refine: send the edited JSON back to model with instructions to adjust/expand
    if refine_btn:
        st.info("Refining test cases — asking the model to expand/modify based on edits...")
        # Use the user's edited JSON as instructions
        user_edit_instructions = "Please produce a corrected/expanded set of test cases according to the JSON below. If you can improve coverage, add more test cases. Only return a JSON array of test cases.\n\nUser JSON edits:\n" + editable_json
        prompt = SYSTEM_PROMPT + "\n\n" + user_edit_instructions # SYSTEM_PROMPT is still valuable here
        success, resp = call_ollama(prompt, model=model_name)
        if not success:
            st.error(resp)
        else:
            st.session_state["last_raw"] = resp
            ok, parsed_or_err = safe_parse_json_array(resp)
            if ok:
                st.session_state["last_testcases"] = parsed_or_err
                st.success("Refinement parsed successfully.")
                df = testcases_to_dataframe(parsed_or_err)
                st.dataframe(df, use_container_width=True)
                # update editable area
                st.text_area("Editable JSON (new)", value=json.dumps(parsed_or_err, indent=2, ensure_ascii=False), height=300, key="editable_after_refine")
            else:
                st.warning("Couldn't parse refined output as JSON array. Showing raw output below.")
                st.text_area("Refined raw model output", value=resp, height=300)

    # Exports
    if export_json_btn:
        ts = st.session_state["last_testcases"]
        if ts:
            os.makedirs("exports", exist_ok=True)
            fname = f"exports/testcases_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            export_testcases_json(ts, fname)
            st.success(f"Saved JSON to `{fname}`")
            with open(fname, "r", encoding="utf-8") as f:
                st.download_button("Download JSON", f, file_name=os.path.basename(fname), mime="application/json")
        else:
            st.error("No testcases to export.")

    if export_csv_btn:
        ts = st.session_state["last_testcases"]
        if ts:
            df = testcases_to_dataframe(ts)
            os.makedirs("exports", exist_ok=True)
            fname = f"exports/testcases_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
            export_testcases_csv(df, fname)
            st.success(f"Saved CSV to `{fname}`")
            with open(fname, "r", encoding="utf-8") as f:
                st.download_button("Download CSV", f, file_name=os.path.basename(fname), mime="text/csv")
        else:
            st.error("No testcases to export.")
else:
    # If raw output exists show it for manual correction
    if st.session_state["last_raw"]:
        st.subheader("Last raw model output (not parsed to JSON)")
        st.text_area("Raw output", value=st.session_state["last_raw"], height=300)
