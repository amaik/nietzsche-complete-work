import os
import random
import argparse
import pickle
import tensorflow as tf
import numpy as np
from pathlib import Path

import draft.constants as constants
import draft.preprocessing.tokenize as tk
from draft.models import recurrent

def main():
    parser = argparse.ArgumentParser(description='Language model training script.')
    parser.add_argument('path', metavar='P', type=str,
                        help='Path to the data folder that keeps the text in individual text files.')
    parser.add_argument('-tp', '--target-path', type=str,
                        help='Target path where the model checkpoints and vocabulary size will be stored')
    parser.add_argument('-m', '--model', type=str, default='lstm',
                        help='The type of model to train. Valid arguements are "gru" an "lstm". Defaults to "lstm"')
    parser.add_argument('-e', '--epochs', type=int, default=constants.EPOCHS,
                        help=f'The number of epochs used for training, defaults to {constants.EPOCHS}')
    parser.add_argument('-t', '--tiny', action='store_true',
                        help='Flag that indicates whether a tiny subset of the training data is taken.')

    args = parser.parse_args()
    path = args.path
    epochs = args.epochs
    model_type = args.model
    if args.target_path is None:
        target_dir = constants.CHECKPOINT_DIR
    else:
        target_dir = args.target_path
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    text = []

    # read all sentences into a list of strings
    num_files = 0
    with os.scandir(path) as it:
        for entry in it:
            with open(entry.path, "r+") as doc:
                text += doc.read().splitlines()
                num_files += 1
    print(f"Finished reading {num_files} files.")

    if args.tiny:
        text = text[:constants.TINY_BATCH_SIZE]

    # shuffle these in order to mix up the books
    random.shuffle(text)

    # tokenize
    tensor, tokenizer = tk.tokenize(text)
    input_tensor = tensor

    # get sentence dimension
    sentence_dim = tensor.shape[1] - 1

    # the target sentences are the input sentences shifted left by one. The last token is replaced by a start sign.
    # in this way the network learns when to start new sentences.
    start_idx = tokenizer.word_index['<start>']
    end_idx = tokenizer.word_index['<end>']

    target_tensor = np.roll(input_tensor, -1, axis=1)  # shift
    target_tensor[:, sentence_dim] = np.full(input_tensor.shape[0], fill_value=0)  # replace start token by 0

    end_token_indices = np.argwhere(target_tensor == end_idx)
    end_token_indices = [idx[1] for idx in end_token_indices]
    target_tensor[range(len(target_tensor)), end_token_indices] = start_idx  # replace end token

    # training section
    buffer_size = len(input_tensor)
    vocab_size = len(tokenizer.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(buffer_size)
    dataset = dataset.batch(constants.BATCH_SIZE, drop_remainder=True)

    if model_type == 'lstm':
        model = recurrent.build_LSTM(vocab_size=vocab_size,
                                  embedding_dim=constants.EMBEDDING_DIMENSION,
                                  rnn_units=constants.RNN_UNITS,
                                  batch_size=constants.BATCH_SIZE)
    elif model_type == 'gru':
        model = recurrent.build_GRU(vocab_size=vocab_size,
                                 embedding_dim=constants.EMBEDDING_DIMENSION,
                                 rnn_units=constants.RNN_UNITS,
                                 batch_size=constants.BATCH_SIZE)
    else:
        print('Incorrect command line argument for --model. Use either "gru" or "lstm".')
        print('Terminating...')
        return 0

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_file = os.path.join(target_dir, "ckpt_best")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        verbose=1,
        monitor='loss',
        mode='min',
        save_best_only=True)

    # save vocab size for reloading the model
    with open(os.path.join(target_dir, constants.VOCAB_SIZE_SUFFIX), 'w+') as file:
        file.write(str(vocab_size))

    # save tokenizer for reloading
    pickle.dump(tokenizer, open(os.path.join(target_dir, constants.TOKENIZER_SUFFIX), "wb"))

    model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


if __name__ == "__main__":
    main()
