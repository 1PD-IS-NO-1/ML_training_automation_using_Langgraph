import streamlit as st
import pandas as pd
from io import StringIO  # Change this import

def display_data_cleaning(result):
    st.header("Data Cleaning Results")
    
    # Display messages related to data cleaning
    messages = [msg for msg in result["messages"] 
               if "data_cleaning" in str(msg.content)]
    
    if messages:
        st.subheader("Process Messages")
        for msg in messages:
            st.info(msg.content)
    
    # Display cleaned dataframe
    if "df" in result:
        st.subheader("Cleaned Dataset")
        st.dataframe(result["df"].head())
        
        st.subheader("Dataset Info")
        buffer = StringIO()  # Use StringIO from io module
        result["df"].info(buf=buffer)
        st.text(buffer.getvalue())

def display_feature_engineering(result):
    st.header("Feature Engineering Results")
    
    if "df" in result:
        st.subheader("Engineered Features")
        st.dataframe(result["df"].head())
        
        st.subheader("Feature Information")
        buffer = StringIO()
        result["df"].info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write(f"Number of features: {result['df'].shape[1]}")
        st.write("Feature types:")
        st.write(result["df"].dtypes)

def display_model_training(result):
    st.header("Model Training Results")
    
    if result.get("model_metrics"):
        st.subheader("Model Performance")
        metrics = result["model_metrics"]
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{float(metrics['accuracy']):.3f}")
        with col2:
            st.metric("Precision", f"{float(metrics['precision']):.3f}")
        with col3:
            st.metric("Recall", f"{float(metrics['recall']):.3f}")
        
        # Display feature importance if available
        if result.get("feature_importance"):
            st.subheader("Feature Importance")
            # Convert feature importance to proper format
            fi_data = {
                'Feature': list(result["feature_importance"].keys()),
                'Importance': [float(x) for x in result["feature_importance"].values()]
            }
            fi_df = pd.DataFrame(fi_data).sort_values('Importance', ascending=False)
            
            # Create a bar chart using plotly
            import plotly.express as px
            fig = px.bar(fi_df, x='Feature', y='Importance',
                        title='Feature Importance')
            st.plotly_chart(fig)
        
        # Display model messages
        if result.get("messages"):
            st.subheader("Training Process")
            for msg in result["messages"]:
                if "model training" in msg.content.lower():
                    st.info(msg.content)

def safe_display_dataframe(df):
    """Safely display a DataFrame in Streamlit"""
    try:
        # Convert nullable types first
        display_df = convert_int64_columns(df.copy())
        display_df = convert_objects_to_numeric(display_df)
        
        
        # Convert boolean columns to int
        bool_columns = display_df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            display_df[col] = display_df[col].astype(int)
        
        # Convert Int64 to regular int64
        int_columns = display_df.select_dtypes(include=['Int64']).columns
        for col in int_columns:
            display_df[col] = display_df[col].astype('int64')
        
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"Error displaying DataFrame: {str(e)}")
        st.write(df.to_dict())

def display_specialist(result):
    st.header("Specialist Interventions")
    
    if result.get("error"):
        st.error(f"Error: {result['error']}")