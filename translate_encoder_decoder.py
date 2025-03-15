import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Embedding, LSTM, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mlflow
import mlflow.keras

mlflow.start_run()

data = pd.read_csv("data.csv")
data = data.iloc[:1500,:]
english_sentences = data['english'].tolist()
spanish_sentences = data['spanish'].tolist()

tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(english_sentences)

tokenizer_es = Tokenizer()
tokenizer_es.fit_on_texts(spanish_sentences)

input_sequences = tokenizer_en.texts_to_sequences(english_sentences)
output_sequences = tokenizer_es.texts_to_sequences(spanish_sentences)

max_input_len = max([len(seq) for seq in input_sequences])
max_output_len = max([len(seq) for seq in output_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_output_len, padding='post')

X = input_sequences
y = np.expand_dims(output_sequences, -1)

# Encoder
encoder_inputs = Input(shape=(max_input_len,))
encoder_embedding = Embedding(input_dim=len(tokenizer_en.word_index) + 1, output_dim=128)(encoder_inputs)
encoder_lstm = LSTM(128, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_output_len,))
decoder_embedding = Embedding(input_dim=len(tokenizer_es.word_index) + 1, output_dim=128)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(tokenizer_es.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlflow.keras.autolog()
model.fit([X, y], y, epochs=35, batch_size=8)
mlflow.keras.log_model(model, 'translation_model')
mlflow.end_run()

# Inference için ayrı encoder modeli
encoder_model = Model(encoder_inputs, encoder_states)

# Inference için ayrı decoder modeli
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding2 = decoder_embedding(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding2, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def translate(sentence):
    seq = tokenizer_en.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')
    
    states_value = encoder_model.predict(padded)
    target_seq = np.zeros((1, 1))
    translated_sentence = ''
    
    for _ in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        predicted_token_index = np.argmax(output_tokens[0, -1, :])
        if predicted_token_index == 0:
            break
        translated_sentence += tokenizer_es.index_word.get(predicted_token_index, '') + ' '
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = predicted_token_index
        states_value = [h, c]
    
    return translated_sentence.strip()

print(translate("My dog is little"))
