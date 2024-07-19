import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # Prediction and probability
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}: {}".format(prediction, emoji_icon))
            st.write("Confidence: {:.2f} %".format(np.max(probability)* 100 ))

        with col2:
            st.success("Prediction Probability")

            # Prepare data for pie chart
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            # Plot pie chart
            fig = alt.Chart(proba_df_clean).mark_arc().encode(
                theta=alt.Theta(field="probability", type="quantitative"),
                color=alt.Color(field="emotions", type="nominal"),
                tooltip=["emotions", "probability"]
            ).properties(
                width=400,
                height=400
            )
            st.altair_chart(fig, use_container_width=True)
            
            

if __name__ == '__main__':
    main()
