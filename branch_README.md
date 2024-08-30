Adds scripts to train models to predict titles, note-seg, whether two sequences should be joined, and where a sequence should be split. 

Scripts added:
1. src/train_title_prediction_model.py
2. src/classify_titles_context.py
3. src/classify_note_seg.py
4. src/train_join_model.py
5. src/classify_joins.py
6. src/train_split_model.py

Data added:
1. datasets/titles/
2. datasets/join_segments/
3. datasets/split_segments/

# Title prediction
src/train_title_prediction_model.py trains a BERT model to predict whether a sequence is a title or not. 
Training, validation and test data are stored in datasets/titles/. 
A model trained on less data is available through huggingface fberi/BertModel-lc.
src/classify_titles_context.py uses this model to add title tags to the corpus. 

A [branch](https://github.com/swerik-project/pyriksdagen/tree/update_context_functions) (update_context_functions) of pyriksdagen adjusts the logic to include more text from the previous sequence in some cases. This may be a small improvement compared to the existing title prediction model.

# Note-seg prediction
src/classify_note_seg.py reclassifies notes and segs in the corpus. 

A pre-trained model to do this can be downloaded from https://github.com/welfare-state-analytics/bert-riksdagen-classifier/releases/tag/v0.4.0

# Join prediction
src/train_join_model.py trains a BERT model to predict whether two adjacent sequences should be joined or not. Training, validation, and test data are stored in datasets/join_segments/.
A model trained using the script is available through huggingface fberi/BertModel-join.
src/classify_joins.py uses this model to join segments in the corpus. 

# split prediction
src/train_split_model.py trains a BERT model to predict which tokens to split a sequence on. This can predict multiple splits per sequence. Training, validation, and test data is stored in datasets/split_segments/. Currently the model tries to perform all types of splits (paragraphs, margins, titles, etc.) but performance is quite poor. Limiting the scope to the easier to classify splits might be a way to make the model useable. 
Since performance is unsatisfactory, there is no script to split segments in the corpus. 
