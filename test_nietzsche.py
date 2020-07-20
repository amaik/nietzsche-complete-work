"""A test script to evaluate the F. Nietzsche Language Model."""
import pickle
import os
import argparse
from pathlib import Path
import tensorflow as tf

from draft.models import models
import draft.constants as constants


def main():
    parser = argparse.ArgumentParser(description='Language model training script.')
    parser.add_argument('checkpoint', metavar='checkpoint-path', type=str,
                        help='Path to the checkpoint folder of the model')
    parser.add_argument('-m', '--model', type=str, default='lstm',
                        help='The type of model to train. Valid arguements are "gru" an "lstm". Defaults to "lstm"')
    parser.add_argument('-v', '--verbose-output', action='store_true',
                        help='Print a more verbose testing output if set.')
    parser.add_argument('-s', '--start-with', type=str, default="",
                        help='The beginning words of the sentence to create.')

    args = parser.parse_args()
    cp_path = args.checkpoint
    model_type = args.model
    if cp_path is None:
        source_dir = constants.CHECKPOINT_DIR
    else:
        source_dir = cp_path
    start_with = args.start_with

    # load vocab size
    vocab_path = os.path.join(source_dir, constants.VOCAB_SIZE_SUFFIX)
    with open(vocab_path, 'r+') as file:
        vocab_size = int(file.readline())

    tokenizer_path = os.path.join(source_dir, constants.TOKENIZER_SUFFIX)
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    if model_type == "lstm":
        model = models.build_LSTM(vocab_size=vocab_size,
                                  embedding_dim=constants.EMBEDDING_DIMENSION,
                                  rnn_units=constants.RNN_UNITS,
                                  batch_size=1)
    elif model_type == "gru":
        model = models.build_GRU(vocab_size=vocab_size,
                                  embedding_dim=constants.EMBEDDING_DIMENSION,
                                  rnn_units=constants.RNN_UNITS,
                                  batch_size=1)
    else:
        print('Incorrect command line argument for --model. Use either "gru" or "lstm".')
        print('Terminating...')
        return 0

    model.load_weights(tf.train.latest_checkpoint(source_dir)).expect_partial()
    model.build(tf.TensorShape([1, None]))

    # model.summary()

    # number of max tokens to generate if the sentence isn't finished
    max_tokens = 25

    text_generated = []

    start_token_idx = tokenizer.word_index['<start>']
    start_indices = [start_token_idx]
    for word in start_with.split(" "):
        text_generated.append(word)
        start_indices.append(tokenizer.word_index[word])
    model_input = tf.expand_dims(start_indices, 0)

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    predicted_id = 0
    model.reset_states()
    while (max_tokens > 0) and (predicted_id != start_token_idx):
        predictions = model(model_input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_id = model_input[0, 0].numpy()

        if args.verbose_output:
            print(f"Input to model was '{tokenizer.index_word[input_id]}' with id {input_id}.")
            print(f"Generated output was {tokenizer.index_word[predicted_id]} with id {predicted_id}.\n...")

        model_input = tf.expand_dims([predicted_id], 0)

        text_generated.append(tokenizer.index_word[predicted_id])
        max_tokens -= 1

    print(" ".join(text_generated).replace("<start>", "").rstrip())


if __name__ == "__main__":
    main()
