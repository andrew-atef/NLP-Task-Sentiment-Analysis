import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# from tensorflow.keras.layers import SimpleRNN

dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
train_data = dataset['train']
test_data = dataset['test']

train_texts = [item['text'] for item in train_data if item['text'] is not None]
train_labels = [item['label'] for item in train_data if item['text'] is not None]
test_texts = [item['text'] for item in test_data if item['text'] is not None]
test_labels = [item['label'] for item in test_data if item['text'] is not None]


label_to_sentiment = {0: "negative", 1: "neutral", 2: "positive"}
print(label_to_sentiment)
print(test_texts)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
print(test_sequences)

max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

vocab_size = 10000
embedding_dim = 64

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_padded, train_labels,
                    epochs=15,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

test_loss, test_accuracy = model.evaluate(test_padded, test_labels, verbose=0)
print("test Accuracy")
print(test_accuracy)

sample_text = ["This is a great tweet"]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_padded = pad_sequences(sample_seq, maxlen=max_length, padding='post')
prediction = model.predict(sample_padded)
predicted_label = np.argmax(prediction)
print(sample_text[0])
print("predicted:")
print(label_to_sentiment[predicted_label])