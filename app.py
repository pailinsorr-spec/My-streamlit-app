import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Reading Comprehension Generator", layout="centered")

#--------------------------------------------------
# Use caching to efficiently load data only once
#--------------------------------------------------
@st.cache_data
def load_data():
   # Load CEFR vocabulary data
   with open("CEFR VOCAB.csv", "r", encoding="utf-8") as file:
       vocab_data = pd.read_csv(file)
   return vocab_data

#-------------------------------------------------
# Gemini Generation
#-------------------------------------------------
def generate_learning_package(level, vocab_list, genre, api_key):
    gemini_model = "gemini-2.5-flash-lite"
    try:
        genai.configure(api_key=api_key)
        
        response_schema = {
            "type": "object",
            "properties": {
                "passage": {"type": "string"},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "choices": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "correct_answer": {"type": "string"}
                        },
                        "required": ["question", "choices", "correct_answer"]
                    }
                },
                "vocab_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "pos": {"type": "string"},
                            "meaning": {"type": "string"},
                            "example": {"type": "string"}
                        },
                        "required": ["word", "pos", "meaning", "example"]
                    }
                }
            },
            "required": ["passage", "questions", "vocab_items"]
        }

        prompt = f"""
        Write a reading passage in the **{genre}** genre that is suitable for CEFR level {level}.
        You MUST use all of the following vocabulary words naturally:
        {', '.join(vocab_list)}.

        The passage should be about 180-220 words.

        Then generate EXACTLY 5 comprehension questions.
        Each question must include:
        - question
        - 4 choices
        - correct_answer

        Next, generate the vocabulary table:
        For each word, provide: part of speech, meaning, and 1 CEFR-{level} example sentence.

        Follow the JSON schema STRICTLY.
        """

        model = genai.GenerativeModel(
            model_name=gemini_model,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )

        response = model.generate_content(prompt)

        return json.loads(response.text)

    except Exception as e:
        st.error(f"API Error: {e}")
        return None
    
#-------------------------------------------------
# Main app
#-------------------------------------------------


# Side bar for API key input
st.sidebar.header("Configuration")
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""

api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password",
        value=st.session_state['api_key'],
        help="Enter your Google Gemini API Key here."
    )
st.session_state['api_key'] = api_key 

# Streamlit App Layout
st.title('Let\'s improve your reading comprehension skills!')

st.caption("An AI-powered tool for generating reading passages, vocabulary tables, and comprehension exercises.")

st.markdown("""
Designed for students and teachers who want:
- 🌟 CEFR-aligned reading passage 
- ✏️ Auto-generated comprehension questions
- 📘 Get a vocabulary list for review  
""")

# CERF level/Genre Selection
col1, col2 = st.columns(2)
with col1:
    st.markdown("## Choose your CEFR Level")
    st.session_state["level"] = st.selectbox("", ["A1", "A2", "B1", "B2", "C1", "C2"])

    descriptions = {
        "A1": "Beginner - basic everyday expressions.",
        "A2": "Elementary - simple communication.",
        "B1": "Intermediate – can handle daily situations.",
        "B2": "Upper-intermediate – can express opinions clearly.",
        "C1": "Advanced – can use complex language effectively.",
        "C2": "Proficient – near-native command of language."
    }
    st.info(descriptions[st.session_state["level"]])

with col2:
    st.markdown("## Select a Genre for the Reading Passage")
    st.session_state["genre"] = st.selectbox("", ["Daily life", "Academic", "Narrative", "Business", "News article"])

# Vocabulary Selection
df = load_data()

if st.session_state["level"] not in df.columns:
    raise ValueError(f"Level '{st.session_state["level"]}' not found in DataFrame columns.")
    
vocab_list = df[st.session_state["level"]].dropna().tolist()

selected_vocab = np.random.choice(vocab_list, size=5, replace=False).tolist()

# Passage and Questions Generation
if st.button("Generate Reading Package"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner("Generating your reading package..."):
            result = generate_learning_package(
                st.session_state["level"],
                selected_vocab,
                st.session_state["genre"],
                api_key
            )
        
        st.session_state["reading_result"] = result


# SHOW PASSAGE + QUESTIONS
if "reading_result" in st.session_state:

    result = st.session_state["reading_result"]
    st.subheader("Reading Passage")
    st.write(result["passage"])
    st.subheader("Comprehension Questions")

    # Prepare user answer state
    if "user_answers" not in st.session_state:
        st.session_state["user_answers"] = {}

    for idx, q in enumerate(result["questions"], 1):
        st.markdown(f"**Q{idx}: {q['question']}**")

        choices = ["-- Select an answer --"] + q["choices"]

        user_choice = st.radio(
            f"Your answer for Q{idx}",
            choices,
            key=f"q{idx}"
        )

        if user_choice == "-- Select an answer --":
            st.session_state["user_answers"][idx] = None
        else:
            st.session_state["user_answers"][idx] = user_choice

    # CHECK ANSWERS
    if st.button("Submit Answers"):
        st.session_state["submitted"] = True

    # If user has submitted answers, show results
    if st.session_state.get("submitted", False):

        correct = 0
        st.write("## 📊 Let's see your results!")

        for idx, q in enumerate(result["questions"], 1):
            user_ans = st.session_state["user_answers"].get(idx)
            correct_ans = q["correct_answer"]

            st.markdown(f"**Q{idx}: {q['question']}**")

            if user_ans == correct_ans:
                correct += 1
                st.success("Correct ✔")
            else:
                st.error("Wrong ✘")
                st.info(f"Correct Answer: {correct_ans}")

        if correct >= 3:
            st.balloons()
            st.success(f"**🎉 Congratulations!, Your Final Score is: {correct}/5**")
        else:
            st.warning(f"**Your Final Score is: {correct}/5. Keep practicing! 💪**")

        #VOCAB TABLE
        vocab_items = result["vocab_items"]
        vocab_df = pd.DataFrame(vocab_items)
        st.write("### 📑 Let's learn some vocabulary from the passage!")

        styled_df = (
            vocab_df.style
                .hide(axis="index")
                .set_table_styles([
                    {
                        'selector': 'th',
                        'props': [
                            ('background-color', '#dae8fc'),
                            ('font-weight', 'bold'),
                            ('color', '#333')
                        ]
                    }
                ])
        )

        st.markdown(styled_df.to_html(), unsafe_allow_html=True)

        #DOWNLOAD FILE
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            vocab_df.to_excel(writer, index=False, sheet_name="Vocabulary")

        excel_data = output.getvalue()

        st.write("**You can download the vocabulary list for your review here!**")
        st.download_button(
            label="⬇️ Download Vocabulary List (Excel)",
            data=excel_data,
            file_name="vocabulary_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )




