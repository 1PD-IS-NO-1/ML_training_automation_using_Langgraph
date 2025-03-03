def validate_dataset(df, target):
    """Validate the input dataset"""
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    if df[target].isnull().any():
        raise ValueError("Target column contains missing values")
    
    return True

def check_file_size(file):
    """Check if file size is within acceptable limits"""
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    file.seek(0, 2)  # Seek to end of file
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        raise ValueError("File size exceeds maximum limit of 200MB")
    
    return True