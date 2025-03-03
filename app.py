import streamlit as st
import pandas as pd
from pipeline import run_pipeline
from components.display import (
    display_data_cleaning,
    display_feature_engineering,
    display_model_training,
    display_specialist
)
from io import StringIO

st.set_page_config(page_title="Data Science Pipeline", layout="wide")

def main():
    st.title("Automated Data Science Pipeline")
    
    # Use session state to persist data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)
                target = st.selectbox("Select Target Column", df.columns.tolist())
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
        else:
            target = st.text_input("Target Column")
            
        model = st.selectbox(
            "Select Model",
            ["LogisticRegression", "RandomForestClassifier"]
        )
        
        api_key = st.text_input("Enter Google API Key", type="password")
        
        run_button = st.button("Run Pipeline")

    # Main content area
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin")
        return

    # Display initial dataset info
    if 'df' in locals():
        st.header("Dataset Preview")
        st.dataframe(df.head())
        
        # Display dataset info
        st.subheader("Dataset Information")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    if run_button and uploaded_file and target and api_key:
        with st.spinner("Running pipeline..."):
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Run the pipeline
                result = run_pipeline(
                    data_path=uploaded_file,
                    target=target,
                    model=model,
                    api_key=api_key
                )
                
                # Store the result in session state
                st.session_state.processed_data = result
                
                # Create tabs for different stages
                tabs = st.tabs([
                    "Data Cleaning",
                    "Feature Engineering",
                    "Model Training",
                    "Specialist"
                ])
                
                with tabs[0]:
                    display_data_cleaning(result)
                
                with tabs[1]:
                    display_feature_engineering(result)
                
                with tabs[2]:
                    display_model_training(result)
                
                with tabs[3]:
                    display_specialist(result)
                
                st.success("Pipeline completed successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()