from keras.layers import Bidirectional, LSTM
from keras_tuner import HyperModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner.tuners import RandomSearch
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

data = [
    ("I love this!", "Positive"),
    ("Average Performance", "Neutral"),
    ("Great job!", "Positive"),
    ("Excellent work!", "Positive"),
    ("Well done!", "Positive"),
    ("I'm impressed!", "Positive"),
    ("Fantastic effort!", "Positive"),
    ("Outstanding performance!", "Positive"),
    ("Keep it up!", "Positive"),
    ("Remarkable!", "Positive"),
    ("Superb!", "Positive"),
    ("Bravo!", "Positive"),
    ("You nailed it!", "Positive"),
    ("Exceptional!", "Positive"),
    ("Hats off to you!", "Positive"),
    ("First-class!", "Positive"),
    ("You did great!", "Positive"),
    ("Amazing!", "Positive"),
    ("Spectacular!", "Positive"),
    ("Incredible work!", "Positive"),
    ("Terrific!", "Positive"),
    ("You rock!", "Positive"),
    ("Phenomenal!", "Positive"),
    ("Awesome!", "Positive"),
    ("Brilliant!", "Positive"),
    ("You’re a star!", "Positive"),
    ("Top-notch!", "Positive"),
    ("Fantastic!", "Positive"),
    ("Super!", "Positive"),
    ("Impressive!", "Positive"),
    ("Good job!", "Positive"),
    ("Excellent!", "Positive"),
    ("It's okay.", "Neutral"),
    ("Satisfactory.", "Neutral"),
    ("Needs improvement.", "Neutral"),
    ("Not bad.", "Neutral"),
    ("Acceptable.", "Neutral"),
    ("Average.", "Neutral"),
    ("Decent.", "Neutral"),
    ("Passable.", "Neutral"),
    ("Fair.", "Neutral"),
    ("Moderate.", "Neutral"),
    ("Ordinary.", "Neutral"),
    ("Plain.", "Neutral"),
    ("Unremarkable.", "Neutral"),
    ("Adequate.", "Neutral"),
    ("Mediocre.", "Neutral"),
    ("Neither good nor bad.", "Neutral"),
    ("So-so.", "Neutral"),
    ("Common.", "Neutral"),
    ("Regular.", "Neutral"),
    ("Standard.", "Neutral"),
    ("Sufficient.", "Neutral"),
    ("Routine.", "Neutral"),
    ("Indifferent.", "Neutral"),
    ("Unexceptional.", "Neutral"),
    ("Workable.", "Neutral"),
    ("Run-of-the-mill.", "Neutral"),
    ("All right.", "Neutral"),
    ("Nominal.", "Neutral"),
    ("Usual.", "Neutral"),
    ("Tolerable.", "Neutral"),
    ("Disappointing.", "Negative"),
    ("Poor performance.", "Negative"),
    ("Not up to the mark.", "Negative"),
    ("Could be better.", "Negative"),
    ("Unsatisfactory.", "Negative"),
    ("Subpar.", "Negative"),
    ("Not good enough.", "Negative"),
    ("Lacking.", "Negative"),
    ("Below expectations.", "Negative"),
    ("Mediocre.", "Negative"),
    ("Weak.", "Negative"),
    ("Unimpressive.", "Negative"),
    ("Underwhelming.", "Negative"),
    ("Deficient.", "Negative"),
    ("Inferior.", "Negative"),
    ("Inadequate.", "Negative"),
    ("Not great.", "Negative"),
    ("Disheartening.", "Negative"),
    ("Less than ideal.", "Negative"),
    ("Second-rate.", "Negative"),
    ("Unacceptable.", "Negative"),
    ("Wanting.", "Negative"),
    ("Regrettable.", "Negative"),
    ("Unfavorable.", "Negative"),
    ("Displeasing.", "Negative"),
    ("Off the mark.", "Negative"),
    ("Not satisfactory.", "Negative"),
    ("Substandard.", "Negative"),
    ("Lousy.", "Negative"),
    ("Bad.", "Negative")
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

corpus = [preprocess(text) for text, _ in data]
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(corpus)
padded_sequences = pad_sequences(sequences, padding='post')
sentimental_mapping = {"Positive": 0, "Negative": 1, "Neutral": 2}
labels = np.array([sentimental_mapping[sentiment] for _, sentiment in data])
train_texts, test_texts, train_labels, test_labels = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
class_count = np.bincount(train_labels)
total_samples = sum(class_count)
class_weights = {cls: total_samples / count for cls, count in enumerate(class_count)}

class SentimentHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Embedding(len(word_index) + 1, hp.Int('embedding_dim', min_value=64, max_value=256, step=32)))
        model.add(Bidirectional(LSTM(hp.Int('lstm_units', min_value=64, max_value=128, step=32), return_sequences=True)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=32), activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(3, activation='softmax'))
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1, sampling='LOG'))
        loss = SparseCategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

hypermodel = SentimentHyperModel()
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=100,
    directory='my_dir',
    project_name='sentiment_analysis'
)
tuner.search(train_texts, train_labels, epochs=100, validation_data=(test_texts, test_labels), class_weight=class_weights, callbacks=[
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=2)
])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = hypermodel.build(best_hps)

print(best_model.summary())

best_model.fit(train_texts, train_labels, epochs=100, validation_data=(test_texts, test_labels), class_weight=class_weights)

def predict_sentiment_with_best_model(user_input, model):
    preprocessed_input = preprocess(user_input)
    sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=100)
    sentiment_probabilities = model.predict(padded_sequence)[0]
    predicted_sentiment = np.argmax(sentiment_probabilities)
    return list(sentimental_mapping.keys())[list(sentimental_mapping.values()).index(predicted_sentiment)]

while True:
    user_input = input("Enter your message (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment_with_best_model(user_input, best_model)
    print("Predicted sentiment:", sentiment)
