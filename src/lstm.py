import random
from typing import List, Dict
import numpy as np
from keras.src.layers import Dense, Activation, LSTM
from keras.src.models import Sequential
from keras.src.optimizers import RMSprop


def load_and_preprocess_text(filepath: str) -> List[str]:
    with open(filepath, encoding='utf-8') as file:
        text = file.read().lower()
    return text.split()


def create_word_mappings(words: List[str]) -> tuple:
    vocabulary = sorted(list(set(words)))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    index_to_word = {i: word for i, word in enumerate(vocabulary)}
    return vocabulary, word_to_index, index_to_word


def prepare_sequences(
        words: List[str],
        max_length: int,
        step: int,
        word_to_index: Dict[str, int],
        vocabulary_size: int
) -> tuple:
    sentences = []
    next_words = []

    for i in range(0, len(words) - max_length, step):
        sentences.append(words[i:i + max_length])
        next_words.append(words[i + max_length])

    X = np.zeros((len(sentences), max_length, vocabulary_size), dtype=np.bool_)
    y = np.zeros((len(sentences), vocabulary_size), dtype=np.bool_)

    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            X[i, t, word_to_index[word]] = 1
        y[i, word_to_index[next_words[i]]] = 1

    return X, y


def build_model(input_shape: tuple, output_size: int) -> Sequential:
    model = Sequential([
        LSTM(128, input_shape=input_shape),
        Dense(output_size),
        Activation('softmax')
    ])

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def sample_index(predictions: np.ndarray, temperature: float = 1.0) -> int:
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)


def generate_text(
        model: Sequential,
        words: List[str],
        max_length: int,
        word_to_index: Dict[str, int],
        index_to_word: Dict[int, str],
        length: int,
        diversity: float
) -> str:
    start_index = random.randint(0, len(words) - max_length - 1)
    sentence_words = words[start_index:start_index + max_length]
    generated = sentence_words.copy()

    for _ in range(length):
        x_pred = np.zeros((1, max_length, len(word_to_index)))
        for t, word in enumerate(sentence_words):
            x_pred[0, t, word_to_index[word]] = 1.0

        predictions = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(predictions, diversity)
        next_word = index_to_word[next_index]

        generated.append(next_word)
        sentence_words = sentence_words[1:] + [next_word]

    formatted_text = [
        ' '.join(generated[i:i + 15])
        for i in range(0, len(generated), 15)
    ]
    return '\n'.join(formatted_text)


def generate():
    # Parameters
    input_file = 'input.txt'
    output_file = '../result/gen.txt'
    max_length = 10
    step = 1
    text_length = 1500
    diversity = 0.2
    epochs = 50
    batch_size = 128

    words = load_and_preprocess_text(input_file)
    vocabulary, word_to_index, index_to_word = create_word_mappings(words)

    X, y = prepare_sequences(
        words, max_length, step, word_to_index, len(vocabulary)
    )

    model = build_model((max_length, len(vocabulary)), len(vocabulary))
    model.fit(X, y, batch_size=batch_size, epochs=epochs)

    generated_text = generate_text(
        model, words, max_length, word_to_index, index_to_word,
        text_length, diversity
    )

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(generated_text)
    print(generated_text)


if __name__ == "__main__":
    generate()