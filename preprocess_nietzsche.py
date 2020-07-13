import re
import os
import argparse
import draft.preprocessing.spellcheck as sp

PATH_INPUT_SUFFIX = "raw/"
PATH_OUTPUT_SUFFIX = "processed/"


def main():
    parser = argparse.ArgumentParser(description='Cleanup script for Nietzsche data.')
    parser.add_argument('path', metavar='P', type=str,
                        help='Path to the data folder that keeps the text in individual text files.')

    args = parser.parse_args()
    path = args.path
    path_input = os.path.join(path, PATH_INPUT_SUFFIX)
    path_output = os.path.join(path, PATH_OUTPUT_SUFFIX)

    with os.scandir(path_input) as it:
        text = ""
        for entry in it:
            with open(entry.path, "r") as doc:
                print(f'Starting processing file {entry.name}.')
                text = doc.read()

                # remove all hyphens from line breaks
                text = re.sub(r'(([^ \n]*)-\n([^ \n]*))', r"\2\3", text)

                # remove short one liners and one liners with numbers (chapter titles, etc.)
                # remove one liners with all caps
                text = re.sub(r'(\n\n.*[0-9]+.*\n\n)', '', text)
                text = re.sub(r'(\n\n[^\n]{0,55}\n\n)', '', text)
                text = re.sub(r'(\n\n[A-Z\s]+\n\n)', '', text)

                # creating a space between a word and the punctuation following it
                # eg: "he is a boy." => "he is a boy ."
                # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
                # do the same for numerals; shrink whitespace to one space
                text = re.sub(r'([,:!.?()\'"^%*])', r" \1 ", text)
                text = re.sub(r'([0-9]+)', r" \1 ", text)
                text = re.sub(r'[" "]+', " ", text)

                # replacing everything with space except [a-zA-Z] and punctuation
                text = re.sub(r"[^a-zA-Z,:.?!]+", " ", text)

                text = text.strip()

                # no free punctuation marks
                text = re.sub(r'(([,:.?!])(\s+[,:.?!])+)', r'\2', text)

                # spellchecking
                text = sp.spell_check(text.split(), True)
                text = " ".join(text)

                # one sentence per line
                text = re.sub(r'([.?!] )', r'\1\n', text)

                # start and end tokens for every sentence <start>, <end>
                text = "\n".join(['<start> ' + w.strip() + ' <end>' for w in text.split("\n")])

                out = open(path_output + entry.name, "w+")
                out.write(text)
                out.close()
        print('\nProcessing finished!')


if __name__ == "__main__":
    main()
