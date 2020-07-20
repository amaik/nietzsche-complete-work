import os
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

from draft.constants import PATH_OUTPUT_SUFFIX


def main():

    parser = argparse.ArgumentParser(description='Language model training script.')
    parser.add_argument('path', metavar='P', type=str,
                        help='Path to the data folder that keeps the text in individual text files.')

    args = parser.parse_args()
    path = args.path
    path = os.path.join(path, PATH_OUTPUT_SUFFIX)

    text = []

    # read all sentences into a list of strings
    with os.scandir(path) as it:
        for entry in it:
            with open(entry.path, "r+") as doc:
                print(f'Starting reading file {entry.name}.')
                text += doc.readlines()

    text = [w.strip() for w in text]

    wordcount = defaultdict(int)

    for line in text:
        for w in line.split():
            wordcount[w] += 1

    print(f"There are {len(wordcount.items())} different words")

    sorted_wc = sorted(wordcount.items(), key=lambda x: x[1])
    one_count = [(w, x) for (w, x) in sorted_wc if x == 1]
    print(len(one_count))
    print(sorted_wc[:50])

    plt.hist(wordcount.values())
    plt.show()


if __name__ == "__main__":
    main()
