# Complete Work of Friedrich Nietzsche

>He who has a why to live can bear almost any how. 

_F. Nietzsche_

This is a dataset I created for some experimentation with language modelling.  
The source of the dataset are scanned books from the [internet archive](https://archive.org/).
Since the source is a scan from old books the dataset is very noisy and there where challenges with cleaning it.
I ended up writing a script to resolve the last typos and spelling mistakes by hand but deemed it too much work. 

However, this might be helpful as an insight what can be done in cleaning a text dataset

## Structure

The folder structure of the data is as follows:
  - data
    - nietzsche
      - bad
      - pre-cleanup
      - processed
      - raw

_raw_ keeps the raw text files from https://archive.org/.  

_bad_ were books that are so unreadable I deemed them too much hassle for processing.  

_processed_ is the processed data. These are the same text files as in _raw_, but each sentence is in it's own line and has a _<start>_ and _<end>_ tag. The result of the script *preprocess_nietzsche.py*.  
  
_pre-cleanup_ is data in preparation for the second, manual cleaning stage (*clean_nietzsche.py*). This data is not finish and the script wasn't run. 


## Language model
The scripts _train_nietzsche.py_ and _test_nietzsche.py_ contain logic to train and test(empirically, no perplexity) a LM on the data. They need import from an unpublished project of mine which will be updatet in the future.
