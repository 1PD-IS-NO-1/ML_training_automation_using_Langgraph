from typing import TypedDict, Annotated, List, Optional
import pandas as pd
import numpy as np
import re
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import io
import sklearn.metrics as metrics
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
def node_wrapper(func):
    def wrapper(state: DataScienceState):
        print(f"Executing node: {func.__name__}")
        return func(state)
    return wrapper


class DataScienceState(TypedDict):
    messages: Annotated[List[HumanMessage], lambda a, b: a + b]
    df: Annotated[pd.DataFrame, lambda _old, new: new]
    target: Annotated[str, lambda _old, new: new]
    model: Annotated[str, lambda _old, new: new]
    error: Annotated[Optional[str], lambda _old, new: new]
    current_step: Annotated[str, lambda _old, new: new]
    execution_history: Annotated[List[dict], lambda a, b: a + b]
    model_object: Annotated[Optional[object], lambda _old, new: new]
    model_metrics: Annotated[Optional[dict], lambda _old, new: new]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, api_key="AIzaSyCF-jMEoZr2ji5kmJvYg4HQGWG--Bq8n84")
from pandas.api.types import is_integer_dtype

def convert_int64_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Int64 columns and boolean types"""
    for col in df.columns:
        # Convert nullable integers
        if pd.api.types.is_integer_dtype(df[col]) and df[col].dtype.name == 'Int64':
            df[col] = df[col].astype('float64' if df[col].isna().any() else 'int64')
        
        # Convert boolean columns to int
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype('int64')
    return df

def convert_objects_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to numeric where possible"""
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # Preserve columns that can't be converted
            continue
    return df
def generate_code(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response.content, re.DOTALL)
    return code_blocks[0] if code_blocks else response.content

def execute_code_safely(code: str, env_vars: dict) -> tuple:
    local_vars = env_vars.copy()
    try:
        exec(code, local_vars)
        
        # Check for required variables after model training
        if 'accuracy' in local_vars and 'precision' in local_vars and 'recall' in local_vars:
            return True, "Model training completed successfully", local_vars
        
        # For other operations, check if DataFrame exists
        df = local_vars.get('df')
        if not isinstance(df, pd.DataFrame):
            return False, "Result is not a DataFrame", local_vars
        if df.empty:
            return False, "DataFrame is empty after processing", local_vars
        return True, "Code executed successfully", local_vars
    except Exception as e:
        return False, f"Error executing code: {str(e)}", local_vars

def validate_dataframe(state: dict, stage: str) -> bool:
    df = state["df"]
    target = state["target"]
    if target not in df.columns:
        return False
    if df.isnull().sum().sum() > 0:
        return False
    if df.empty:
        return False
    return True

@node_wrapper
def data_cleaner_node(state: DataScienceState) -> DataScienceState:
    state["current_step"] = "data_cleaning"
    df = state["df"].copy()

    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    prompt = f"""As a Data Cleaning expert, generate Python code to clean the entire dataframe.if top 5 row of dataframe is given here as
Sample data:
{df.head()}

DataFrame info:
{info}

Requirements:
1. Handle missing values appropriately
2. Remove irrelevant columns
3. Fix data types
4. Handle duplicates
5. Handle outliers
6. Preserve target column: {state["target"]}
Important: 
- Never use inplace=True
- Always assign modified DataFrame back to 'df'
- Convert boolean columns to integers (0/1)
- Ensure final dataframe has only numeric types.
Return complete, executable Python code.
Important: Use the existing 'df' variable in memory. Do NOT load data from files."""
# In data_cleaner_node and feature_engineer_node, modify prompts to include:

    for _ in range(3):
        code = generate_code(prompt)
        success, message, vars = execute_code_safely(
            code, {"df": df.copy(), "pd": pd, "np": np}
        )
        if success:
            cleaned_df = vars["df"]
            cleaned_df = convert_int64_columns(cleaned_df)
            cleaned_df = convert_objects_to_numeric(cleaned_df)
            state["df"] = cleaned_df

    state["error"] = "Failed data cleaning"
    return state

@node_wrapper
def feature_engineer_node(state: DataScienceState) -> DataScienceState:
    state["current_step"] = "feature_engineering"
    df = state["df"].copy()

    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    prompt = f"""As a Feature Engineering expert, generate Python code for feature engineering.
Sample data:
{df.head()}

DataFrame info:
{info}

Requirements:
1. Create new features
2. Encode categorical variables
3. Scale numerical features
4. Preserve target column: {state["target"]}
- Never use inplace=True
- Always assign modified DataFrame back to 'df'
- Convert boolean columns to integers (0/1)
- Ensure final dataframe has only numeric types.
Return complete, executable Python code.
Important: Use the existing 'df' variable in memory. Do NOT load data from files."""

    for _ in range(3):
        code = generate_code(prompt)
        success, message, vars = execute_code_safely(
            code, {"df": df.copy(), "pd": pd, "np": np}
        )
        if success:
            engineered_df = vars["df"]
            engineered_df = convert_int64_columns(engineered_df)
            engineered_df = convert_objects_to_numeric(engineered_df)
            state["df"] = engineered_df

    state["error"] = "Failed feature engineering"
    return state



@node_wrapper
def model_trainer_node(state: DataScienceState) -> DataScienceState:
    state["current_step"] = "model_training"
    df = state["df"].copy()
    # Add type conversions before training
    prompt = f"""Generate Python code to train a {state["model"]} model.
Target variable: {state["target"]}

Return complete, executable Python code."""

    for _ in range(3):
        code = generate_code(prompt)
        print(code)
        success, message, vars = execute_code_safely(
            code, {
                "df": df.copy(),
                "pd": pd,
                "np": np,
                "train_test_split": train_test_split,
                "RandomForestClassifier": RandomForestClassifier,
                "joblib": joblib,
                "metrics": metrics
            }
        )
        if success:
            state["model_metrics"] = {
                "accuracy": vars.get("accuracy"),
                "precision": vars.get("precision"),
                "recall": vars.get("recall")
            }
            state["messages"].append(HumanMessage(content=message))
            state["error"] = None
            return state

    state["error"] = "Failed model training"
    return state

@node_wrapper
def data_science_specialist_node(state: DataScienceState) -> DataScienceState:
    state["current_step"] = "specialist"
    df = state["df"].copy()
    error = state["error"]

    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    prompt = f"""Fix the following error in the {state['current_step']} step:
Error: {error}

DataFrame info:
{info}

Generate Python code to fix this issue."""

    for _ in range(3):
        code = generate_code(prompt)
        print(code)
        success, message, vars = execute_code_safely(
            code, {
                "df": df.copy(),
                "pd": pd,
                "np": np,
                "train_test_split": train_test_split,
                "RandomForestClassifier": RandomForestClassifier,
                "joblib": joblib,
                "metrics": metrics
            }
        )
        if success:
            state["df"] = vars["df"]
            state["error"] = None
            state["messages"].append(HumanMessage(content="Fix applied successfully"))
            return state

    state["error"] = "Specialist unable to fix the issue"
    return state

def create_workflow():
    workflow = StateGraph(DataScienceState)

    workflow.add_node("data_cleaner", data_cleaner_node)
    workflow.add_node("feature_engineer", feature_engineer_node)
    workflow.add_node("model_trainer", model_trainer_node)
    workflow.add_node("specialist", data_science_specialist_node)

    def route_on_error(state: DataScienceState) -> str:
        return "specialist" if state.get("error") else "next"

    workflow.add_conditional_edges(
        "data_cleaner",
        route_on_error,
        {
            "next": "feature_engineer",
            "specialist": "specialist"
        }
    )

    workflow.add_conditional_edges(
        "feature_engineer",
        route_on_error,
        {
            "next": "model_trainer",
            "specialist": "specialist"
        }
    )

    workflow.add_conditional_edges(
        "model_trainer",
        route_on_error,
        {
            "next": END,
            "specialist": "specialist"
        }
    )

    def specialist_router(state: DataScienceState) -> str:
        if state.get("error"):
            return END

        # Route to next appropriate node based on where we came from
        current_step = state.get("current_step")
        if current_step == "data_cleaning":
            return "feature_engineer"
        elif current_step == "feature_engineering":
            return "model_trainer"
        else:
            return END

    workflow.add_conditional_edges(
        "specialist",
        specialist_router
    )

    workflow.set_entry_point("data_cleaner")
    return workflow.compile()

# pipeline.py
# (Keep all your existing imports)
def run_pipeline(data_path, target: str, model: str, api_key: str):
    try:
        # Handle different types of input
        if hasattr(data_path, 'read'):
            # Create a copy of the data in memory
            data_path.seek(0)
            df = pd.read_csv(data_path)
        else:
            df = pd.read_csv(data_path)
        
        initial_state = DataScienceState(
            messages=[],
            df=df.copy(),  # Make sure to create a copy
            target=target,
            model=model,
            error=None,
            current_step="",
            execution_history=[],
            model_object=None,
            model_metrics=None
        )

        chain = create_workflow()
        return chain.invoke(initial_state)
        
    except Exception as e:
        raise Exception(f"Error processing the file: {str(e)}")