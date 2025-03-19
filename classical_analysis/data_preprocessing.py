import pandas as pd
import nltk
from nltk.corpus import stopwords
from dateutil.relativedelta import relativedelta
import string

# Ensure the Portuguese stopwords are downloaded.
nltk.download('stopwords', quiet=True)

TIME_COL = 'DATA DE JULGAMENTO'

TARGET_COL = 'label'

NUMERICAL_COLS = [
    'VALOR DA CAUSA',
    'QTD RTE',
    'QTD RDO'
]

CATEGORICAL_COLS = [
    'VARA DO TRABALHO',
    'RAMO DE ATIVIDADE',
    'CLASSE PROCESSUAL',
    'CIDADE ORIG PET INICIAL',
    'OAB',
    'ASSUNTOS',
    'RECDA PES FÍS OU JUR',
    'PORTADOR DEFICIÊNCIA',
    'SEGREDO DE JUSTIÇA',
    'RECDA ATIVA-INATIVA',
    'ENTE PUB OU PRIV',
    'INDICADOR DO PROC',
    'DOCUMENTOS DAS RECLAMADAS',
    'DOCUMENTOS DOS RECLAMANTES'
]

COL_DELIMITERS = {'OAB':', ',
                'ASSUNTOS': '; ',
                'ENTE PUB OU PRIV': ', ',
                'INDICADOR DO PROC': ', ',
                'DOCUMENTOS DAS RECLAMADAS': '|',
                'DOCUMENTOS DOS RECLAMANTES':'|'}

def load_dataset(path):
    print("Loading dataset from", path)
    df = pd.read_parquet(path)
    
    df.drop_duplicates(subset=['NÚMERO DO PROCESSO'], keep='last', ignore_index=True, inplace=True)
    # df = df.sample(100, random_state=42) # Test 
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['label'] = df['TIPO DE SOLUÇÃO'].apply(lambda x: 1 if x == 'Conciliações' else 0)
    
    for col in CATEGORICAL_COLS:
        print(f"Processing column: {col}")
        df = preprocess_text_column(df, col)
    
    return df

def preprocess_text(text):
    """
    Preprocess a given text by lowercasing, removing punctuation, and filtering out Portuguese stopwords.
    
    Parameters:
    - text: str, the input text.
    
    Returns:
    - A string with the processed text.
    """
    # Convert text to lowercase.
    text = text.lower()
    # Tokenize the text (here using a simple whitespace split).
    tokens = text.split()
    # Remove Portuguese stopwords.
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if word not in stop_words]
    # Rejoin tokens into a single string.
    return ' '.join(tokens)

def preprocess_text_column(df, text_col, new_col=None):
    """
    Apply text preprocessing to a specific column in a DataFrame.
    If new_col is provided, the preprocessed text will be stored in that column;
    otherwise, it will overwrite the original column.
    
    Parameters:
    - df: pandas DataFrame.
    - text_col: str, name of the column containing text.
    - new_col: str, name of the column to store processed text (if None, overwrites text_col).
    
    Returns:
    - DataFrame with the text preprocessed in the specified column.
    """
    if new_col is None:
        new_col = text_col
    df[new_col] = df[text_col].astype(str).apply(preprocess_text)
    return df

def create_rolling_windows(df, date_col, window_size, window_step):
    """
    Split the DataFrame into rolling windows based on a datetime column.
    The window size and step are given as numbers interpreted as years and months, respectively.
    
    Parameters:
    - df: pandas DataFrame that contains a datetime column specified by date_col.
    - date_col: string, name of the datetime column.
    - window_size: integer, window size in years.
    - window_step: integer, step between windows in months.
    
    Returns:
    - List of DataFrame slices corresponding to each rolling window.
    """
    # Ensure the date_col is datetime type.
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    window_size_delta = relativedelta(years=window_size)
    window_step_delta = relativedelta(months=window_step)
    
    windows = []
    start_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Create windows until the start_date goes beyond the maximum date.
    while start_date <= max_date:
        end_date = start_date + window_size_delta
        window_df = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
        if not window_df.empty:
            windows.append(window_df)
        start_date = start_date + window_step_delta
    
    return windows
