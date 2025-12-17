import tensorflow as tf

def build_model_architecture(vocab_size, max_length, exp_config):
    model_type = exp_config['type']
    embedding_dim = exp_config['embedding_dim']
    units = exp_config['units']
    
    model = tf.keras.Sequential()
    
    # 1. Entrada explícita
    model.add(tf.keras.layers.Input(shape=(max_length,)))
    
    # 2. Embedding
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
    
    # --- LOGICA DE SELECCIÓN DE ARQUITECTURA ---
    if model_type == 'lstm':
        # LSTM Bidireccional
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=False)))
        model.add(tf.keras.layers.Dropout(0.5))
        
        # Capa densa extra para el experimento complejo
        if units > 100: 
            model.add(tf.keras.layers.Dense(units // 2, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.4))
            
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        
    elif model_type == 'dense':
        # Modelo liviano (Bag of Words approach)
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))

    elif model_type == 'cnn':
        # --- NUEVO: Modelo Convolucional ---
        # Conv1D busca patrones de 5 palabras (kernel_size=5)
        # 'units' aquí se usa como el número de filtros (filters)
        model.add(tf.keras.layers.Conv1D(filters=units, kernel_size=5, activation='relu'))
        # GlobalMaxPooling se queda con la característica más fuerte encontrada en todo el texto
        model.add(tf.keras.layers.GlobalMaxPooling1D())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
    
    # Salida Binaria
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model