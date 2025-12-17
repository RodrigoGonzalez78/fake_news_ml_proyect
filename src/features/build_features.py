import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_process_data(config):
    print("--- PROCESANDO DATOS ---")
    
    # Cargar Dataset
    path = config['paths']['raw_data']
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el archivo en: {path}")
        
    df = pd.read_csv(path)
    df = df.dropna(subset=['combined_text', 'label'])
    
    X = df['combined_text'].astype(str).values
    y = df['label'].values

    # 1. Separar Test (20%) - "Bajo llave"
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config['global_params']['test_size'], random_state=42
    )
    
    # 2. Separar Train y Validation del resto
    # Ajustamos el porcentaje para que val sea aprox 20% del total original
    val_split = config['global_params']['val_size'] / (1 - config['global_params']['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=42
    )

    # 3. Tokenización (Solo aprender de Train)
    tokenizer = Tokenizer(num_words=config['global_params']['vocab_size'], 
                          oov_token=config['global_params']['oov_tok'])
    tokenizer.fit_on_texts(X_train)

    # Guardar Tokenizer
    os.makedirs(os.path.dirname(config['paths']['tokenizer']), exist_ok=True)
    with open(config['paths']['tokenizer'], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. Convertir a secuencias
    def get_sequences(texts):
        seq = tokenizer.texts_to_sequences(texts)
        pad = pad_sequences(seq, 
                            maxlen=config['global_params']['max_length'], 
                            padding=config['global_params']['padding_type'], 
                            truncating=config['global_params']['trunc_type'])
        return np.array(pad)

    return (get_sequences(X_train), y_train), \
           (get_sequences(X_val), y_val), \
           (get_sequences(X_test), y_test)