import json
import pandas as pd
import streamlit as st
from duckduckgo_search import DDGS
from openai import OpenAI

st.set_page_config(page_title="PhD Program Search Agent", layout="wide")

st.title("🎓 PhD Program Search Agent")
st.write("Find relevant PhD programs across target universities based on research keywords.")

# Sidebar for key management
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    univ_input = st.text_input("Universities (separated by ';')", "Lund University; University of Cambridge; Harvard University")
with col2:
    keywords_input = st.text_input("Field Keywords", "animal behaviour")

if st.button("Search PhD Programs", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        client = OpenAI(api_key=api_key)
        universities = [u.strip() for u in univ_input.split(";") if u.strip()]
        
        all_results = []
        
        for univ in universities:
            st.info(f"Searching web pages for **{univ}**...")
            
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

            # Step B: LLM Processing & Broad Field Matching
            prompt = f"""
            You are an academic research advisor. Analyze the web search results for '{univ}'.
            Extract PhD or Doctoral programs that match the topic '{keywords_input}'.

            Broad Semantic Matching Rules:
            - Include programs where research on '{keywords_input}' is permitted or naturally fits (e.g., 'animal behaviour' matches Biology, Biological Sciences, Ecology, Neuroscience, Zoology).
            - Exclude completely unrelated departments (e.g., Economics, Physics).

            Search Results Context:
            {search_context}

            Return a JSON object containing a key "programs" which is a list of objects with these exact keys:
            - "Program name"
            - "Affiliated university" (Set value to '{univ}')
            - "Program website link"
            - "Program application deadline"
            - "Program application requirements"
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                data = json.loads(response.choices[0].message.content)
                programs = data.get("programs", [])
                all_results.extend(programs)
            except Exception as e:
                st.error(f"Error processing data for {univ}: {e}")

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
            
            # CSV Download button for Google Sheets import
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV for Google Sheets",
                data=csv,
                file_name='phd_programs.csv',
                mime='text/csv',
            )
        else:
            st.warning("No matching programs found. Try broadening your field keywords.")
