# CENG442 Assignment 2: Sentiment Analysis with Deep Learning
## Due Date: 24/12/2024 23:59

### Project Overview
In this assignment, you will implement a sentiment analysis model using deep learning techniques. You will work with an English text dataset, perform various preprocessing steps, and implement a neural network model using GloVe embeddings.

### Dataset
- English text dataset with sentiment labels
- Format: CSV/DataFrame with 'reviews' column and sentiment labels
- Students should split the data into training (80%), validation (10%), and test (10%) sets

### Part 1: Text Preprocessing (40 points)
Implement the following preprocessing steps in order:

1. **Contraction Expansion** (`cont_exp`)
   - Expand contractions in the text (e.g., "don't" → "do not")
   - Use appropriate contraction mapping dictionary

2. **Email Removal** (`remove_emails`)
   - Remove all email addresses from the text
   - Use regular expressions to identify and remove email patterns

3. **HTML Tags Removal** (`remove_html_tags`)
   - Remove any HTML tags present in the text
   - Example: "<br>" or "<p>" tags should be removed

4. **URL Removal** (`remove_urls`)
   - Remove all URLs from the text
   - Use regular expressions to identify and remove URL patterns

5. **Special Characters Removal** (`remove_special_chars`)
   - Remove special characters and numbers
   - Keep only alphabetic characters and spaces

6. **Accented Characters Handling** (`remove_accented_chars`)
   - Replace accented characters with their non-accented equivalents
   - Example: 'é' → 'e'

7. **Text Normalization** (`make_base`)
   - Convert text to lowercase
   - Remove extra whitespaces
   - Basic text cleaning and standardization

8. **Spelling Correction**
   - Implement basic spelling correction
   - Extract raw sentences after correction

### Part 2: Model Implementation (60 points)

1. **GloVe Embedding Integration (20 points)**
   - Download and load GloVe embeddings (Stanford's pre-trained embeddings)
   - Use the 100-dimensional GloVe vectors
   - Create an embedding matrix for your vocabulary
   - Handle words not found in GloVe embeddings

2. **Neural Network Architecture (25 points)**
   - Input Layer: Tokenized and padded sequences
   - Embedding Layer: Using pre-trained GloVe embeddings
   - At least one LSTM/BiLSTM layer
   - Appropriate dense layers
   - Output Layer: Suitable for sentiment classification

3. **Training and Evaluation (15 points)**
   - Implement model training with appropriate callbacks
   - Use at least 5 epochs
   - Evaluate model on test set
   - Report accuracy, precision, recall, and F1-score

### Required Libraries
```python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

### Deliverables
1. Python notebook (.ipynb) containing:
   - All preprocessing implementations
   - Model architecture and training
   - Evaluation results
2. Brief report (max 2 pages) explaining:
   - Preprocessing decisions
   - Model architecture choices
   - Analysis of results
   - Please note that in addition to completing your assignment, you are required to present your work and record the presentation. The recorded presentation must be shared via an unlisted YouTube video, and the link should be submitted as part of your deliverables. Failure to meet this requirement will result in a score of 0 for the assignment.

### Grading Criteria
- Correct implementation of all preprocessing steps (40%)
- Proper integration of GloVe embeddings (20%)
- Model architecture and implementation (25%)
- Model performance and evaluation (15%)

### Additional Notes
- Comments and documentation are required
- Code should be well-organized and efficient
- Include error handling for edge cases
- Use vectorized operations where possible
- Include visualization of training progress
