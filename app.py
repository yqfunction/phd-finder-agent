import json
import pandas as pd
import streamlit as st
from duckduckgo_search import DDGS
from openai import OpenAI

st.set_page_config(page_title="PhD Program Search Agent", layout="wide")

st.title("🎓 Multi-Provider PhD Program Search Agent")
st.write("Find relevant PhD programs across target universities using your preferred AI API provider.")

# Provider Configurations
PROVIDERS = {
    "Groq (Free)": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        "help": "Get key from console.groq.com (Key format: gsk_...)"
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "help": "Get key from platform.openai.com (Key format: sk-proj-...)"
    },
    "OpenRouter (Free/Paid Models)": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-lite-001", "deepseek/deepseek-r1"],
        "help": "Get key from openrouter.ai (Key format: sk-or-v1-...)"
    },
    "Google Gemini (OpenAI Endpoint)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "help": "Get key from aistudio.google.com (Key format: AIzaSy...)"
    },
    "Custom OpenAI-Compatible API": {
        "base_url": None,
        "models": [],
        "help": "Specify your custom provider's base URL and model name."
    }
}

# Sidebar Settings
st.sidebar.header("🔑 API Settings")
provider_choice = st.sidebar.selectbox("Select API Provider", list(PROVIDERS.keys()))

provider_info = PROVIDERS[provider_choice]
st.sidebar.caption(provider_info["help"])

api_key = st.sidebar.text_input("API Key", type="password")

# Custom Provider overrides
if provider_choice == "Custom OpenAI-Compatible API":
    custom_base_url = st.sidebar.text_input("Base URL", "https://api.your-provider.com/v1")
    custom_model = st.sidebar.text_input("Model Name", "model-name")
    target_base_url = custom_base_url
    selected_model = custom_model
else:
    target_base_url = provider_info["base_url"]
    selected_model = st.sidebar.selectbox("Select Model", provider_info["models"])

# Main Search Inputs
col1, col2 = st.columns(2)
with col1:
    univ_input = st.text_input("Universities (separated by ';')", "Lund University; University of Cambridge; Harvard University")
with col2:
    keywords_input = st.text_input("Field Keywords", "animal behaviour")

if st.button("Search PhD Programs", type="primary"):
    if not api_key:
        st.error(f"Please enter your API key for {provider_choice} in the sidebar.")
    else:
        # Initialize dynamic client
        client = OpenAI(
            api_key=api_key,
            base_url=target_base_url
        )
        
        universities = [u.strip() for u in univ_input.split(";") if u.strip()]
        all_results = []
        
        for univ in universities:
            st.info(f"Searching web pages for **{univ}** via {provider_choice}...")
            
            # Step A: Perform web search
            search_query = f"{univ} PhD doctorate admissions research {keywords_input}"
            search_context = ""
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(search_query, max_results=5))
                    for r in results:
                        search_context += f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n\n"
            except Exception as e:
                st.warning(f"Web search error for {univ}: {e}")

            # Step B: LLM Processing
            prompt = f"""
            You are an academic research advisor. Analyze the web search results for '{univ}'.
            Extract PhD or Doctoral programs that match the topic '{keywords_input}'.

            Broad Semantic Matching Rules:
            - Include programs where research on '{keywords_input}' is permitted or naturally fits (e.g., 'animal behaviour' matches Biology, Biological Sciences, Ecology, Neuroscience, Zoology).
            - Exclude completely unrelated departments (e.g., Economics, Physics).

            Search Results Context:
            {search_context}

            Return a valid JSON object containing a key "programs" which is a list of objects with these exact keys:
            - "Program name"
            - "Affiliated university" (Set value to '{univ}')
            - "Program website link"
            - "Program application deadline"
            - "Program application requirements"
            """

            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                programs = data.get("programs", [])
                all_results.extend(programs)
            except Exception as e:
                st.error(f"Error processing data for {univ} with model {selected_model}: {e}")

        # Step C: Display & Export Table
        if all_results:
            df = pd.DataFrame(all_results)
            
            cols = ["Program name", "Affiliated university", "Program website link", "Program application deadline", "Program application requirements"]
            for col in cols:
                if col not in df.columns:
                    df[col] = "See website / Rolling"
            df = df[cols]

            st.success("Search complete!")
            st.subheader("Potential PhD Programs Found")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV for Google Sheets",
                data=csv,
                file_name='phd_programs.csv',
                mime='text/csv',
            )
        else:
            st.warning("No matching programs found. Try broadening your field keywords.")
