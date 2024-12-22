import streamlit as st
from transformers import pipeline


# ------------------------------
# Load Whisper Model
# ------------------------------
@st.cache_resource
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


# ------------------------------
# Load NER Model
# ------------------------------
@st.cache_resource
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, whisper_model):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        whisper_model: Loaded Whisper model pipeline.
    Returns:
        str: Transcribed text from the audio file.
    """
    # We convert the audio file into a byte format that the model can process
    audio_bytes = uploaded_file.read()

    # Adding return_timestamps=True parameter for long sounds
    result = whisper_model(audio_bytes, return_timestamps=True)

    # Transcription and return timestamps if available
    transcription = result["text"]
    return transcription


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = ner_pipeline(text)
    grouped_entities = {"Persons": set(), "Organizations": set(), "Locations": set()}

    for entity in entities:
        entity_group = entity["entity_group"]
        if entity_group == "PER":
            grouped_entities["Persons"].add(entity["word"])
        elif entity_group == "ORG":
            grouped_entities["Organizations"].add(entity["word"])
        elif entity_group == "LOC":
            grouped_entities["Locations"].add(entity["word"])

    return grouped_entities


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    STUDENT_NAME = "Hasan KAN"
    STUDENT_ID = "150220332"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    st.write("Upload a business meeting audio file to:")
    st.write("1.\t Transcribe the meeting audio into text.\n2.\t Extract key entities such as Persons, Organizations, Dates, and Locations.")


    st.write("Upload an Audio File (WAV Format)")
    uploaded_file = st.file_uploader("Choose a file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.info("Transcribing the audio file... This may take a minute.")

        # Load models
        whisper_model = load_whisper_model()
        ner_model = load_ner_model()

        # Transcription
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(uploaded_file, whisper_model)
            st.subheader("Transcription")
            st.write(transcription)

        # Named Entity Recognition (NER)
        st.info("Extracting entities...")
        with st.spinner("Extracting entities..."):
            grouped_entities = extract_entities(transcription, ner_model)
            st.subheader("Extracted Entities")
            # Display Grouped Entities

            st.markdown("**Organizations (ORGs):**")
            st.markdown(
                f"<ul>{''.join(f'<li>{item}</li>' for item in sorted(grouped_entities['Organizations']))}</ul>",
                unsafe_allow_html=True,
            )

            st.markdown("**Locations (LOCs):**")
            st.markdown(
                f"<ul>{''.join(f'<li>{item}</li>' for item in sorted(grouped_entities['Locations']))}</ul>",
                unsafe_allow_html=True,
            )

            st.markdown("**Persons (PERs):**")
            st.markdown(
                f"<ul>{''.join(f'<li>{item}</li>' for item in sorted(grouped_entities['Persons']))}</ul>",
                unsafe_allow_html=True,
            )



if __name__ == "__main__":
    main()