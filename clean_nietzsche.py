"""Cleanup typos and noise in Nietzsche data.

Due to the usage of readline, this script can only be run in an linux environment.
"""
import os
import shutil
import argparse
import tempfile
import readline

from typing import List

import enchant

ENCHANT_DICT = enchant.Dict("en_UK")


def has_typo(line: str) -> List[str]:
    line.replace('<unchecked> ', '')
    # first and last element are start/end token
    split = line.split(" ")[1: -1]
    checked = [w for w in split if w.isalpha() and not ENCHANT_DICT.check(w)]
    return checked


def rl_input(editable):
    readline.set_startup_hook(lambda: readline.insert_text(editable))
    try:
        return input()
    finally:
        readline.set_startup_hook()


def main():
    parser = argparse.ArgumentParser(description='Cleanup script for Nietzsche data.')
    parser.add_argument('path', metavar='P', type=str,
                        help='Path to the data folder that keeps the text in individual text files.')

    args = parser.parse_args()
    path = args.path
    all_counter = 0
    has_typos_counter = 0

    with os.scandir(path) as it:
        text = ""
        for entry in it:
            # Create temp file
            tf, abs_path = tempfile.mkstemp()
            with os.fdopen(tf, 'w') as new_file:
                with open(entry.path, "r") as old_file:
                    print(f'Start processing of file {entry.name}.')
                    for line in old_file:
                        if '<unchecked>' in line:
                            typos = has_typo(line)
                            if typos:
                                print(typos)
                                corrected = rl_input(line)
                                has_typos_counter += 1
                                new_file.write(corrected.replace('<unchecked>', ''))
                            all_counter += 1

            # Copy the file permissions from the old file to the new file
            shutil.copymode(entry.path, abs_path)
            # Remove original file
            os.remove(entry.path)
            # Move new file
            shutil.move(abs_path, entry.path)


if __name__ == '__main__':
    main()
