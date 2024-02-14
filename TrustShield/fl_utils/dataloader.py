import json
import torch
import pandas as pd
import numpy as np
import random

from pathlib        import Path
from sklearn.model_selection \
                    import train_test_split
from TrustShield.fl_utils.tokenizer      import CustomTokenizer
from torch.utils.data \
                    import TensorDataset, DataLoader, RandomSampler, \
                           SequentialSampler

label_config = {"Books" : 0,
                "Clothing & Accessories" : 1,
                "Electronics" : 2,
                "Household"   : 3}

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class DataLoaderForClassification:
    def __init__(self, file, new) -> None:
        """
        Initializes the data loader from the file to read data from
        """
        self.df = pd.read_csv(file)
        self.parse_data(new=new)
        self.tokenize()
        self.prepare_tensors()
        self.batch_size       = 32
        self.train_data       = TensorDataset(self.train_seq,
                                              self.train_mask,
                                              self.train_y)
        self.train_sampler    = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data,
                                           sampler=self.train_sampler,
                                           batch_size=self.batch_size)
        self.val_data         = TensorDataset(self.val_seq,
                                              self.val_mask,
                                              self.val_y)
        self.val_sampler      = SequentialSampler(self.val_data)
        self.val_dataloader   = DataLoader(self.val_data,
                                           sampler=self.val_sampler,
                                           batch_size=self.batch_size)

    def parse_data(self, new=False):
        """
        Parses text classification data.
        """
        if new:
            self.df.dropna(inplace = True) # Dropping observations with missing values
            self.df.drop_duplicates(inplace = True) # Dropping duplicate observations
            self.df.reset_index(drop = True, inplace = True) # Resetting index
            for i, label in enumerate(np.unique(self.df['label'])):
                print(i, label)
                self.df['label'] = self.df['label'].replace(label, label_config[label] if label_config.get(label) is not None else i)

        self.train_text, self.temp_text, self.train_labels, self.temp_labels = \
            train_test_split(self.df['text'], self.df['label'], 
                             random_state=2018, 
                             test_size=0.19, 
                             stratify=self.df['label'])
        # we will use temp_text and temp_labels to create validation and test set
        self.val_text, self.test_text, self.val_labels, self.test_labels = \
            train_test_split(self.temp_text, self.temp_labels, 
                             random_state=2018, 
                             test_size=0.7, 
                             stratify=self.temp_labels)

    def tokenize(self):
        """
        Tokenizes the input.
        """
        self.seq_len      = [len(i.split()) for i in self.train_text]
        self.max_seq_len  = 25
        self.tokenizer    = CustomTokenizer(self.max_seq_len)
        self.tokens_train = self.tokenizer.tokenize_text(self.train_text)
        self.tokens_val   = self.tokenizer.tokenize_text(self.val_text)
        self.tokens_test  = self.tokenizer.tokenize_text(self.test_text)

    def prepare_tensors(self):
        """
        Creates appropriate tensors for the required data.
        """
        # for train set
        self.train_seq  = torch.tensor(self.tokens_train['input_ids'])
        self.train_mask = torch.tensor(self.tokens_train['attention_mask'])
        self.train_y    = torch.tensor(self.train_labels.tolist())

        # for validation set
        self.val_seq  = torch.tensor(self.tokens_val['input_ids'])
        self.val_mask = torch.tensor(self.tokens_val['attention_mask'])
        self.val_y    = torch.tensor(self.val_labels.tolist())

        # for test set
        self.test_seq  = torch.tensor(self.tokens_test['input_ids'])
        self.test_mask = torch.tensor(self.tokens_test['attention_mask'])
        self.test_y    = torch.tensor(self.test_labels.tolist())

class DataLoaderForQA:
    def __init__(self, train_file, dev_file, seed=42, load=True, deberta=False) -> None:
        train_path = Path(train_file)
        val_path   = Path(dev_file)
        with open(train_path, 'rb') as f:
            train_dict = json.load(f)
        with open(val_path, 'rb') as f:
            val_dict = json.load(f)
        self.train_texts, self.train_queries, self.train_answers = self.parse_dict(train_dict, seed)
        self.val_texts, self.val_queries, self.val_answers       = self.parse_dict(val_dict, seed)
        self.tokenizer = CustomTokenizer(512, deberta)
        if load:
            self.find_end_positions(self.train_texts, self.train_answers)
            self.find_end_positions(self.val_texts, self.val_answers)
            self.train_encodings, self.val_encodings = self.tokenize()
            self.add_token_positions(self.train_encodings, self.train_answers)
            self.add_token_positions(self.val_encodings, self.val_answers)
            self.train_dataset = SquadDataset(self.train_encodings)
            self.val_dataset   = SquadDataset(self.val_encodings)
            self.train_loader  = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
            self.val_loader    = DataLoader(self.val_dataset, batch_size=16, shuffle=True)
            print("Total Data Size : {}".format(len(self.train_loader)))
        # Create the tokenizer

    def parse_dict(self, dict, seed):
        """
        Parse the SQuAD2.0 dataset.
        """
        texts = []
        queries = []
        answers = []

        # Search for each passage, its question and its answer
        for group in dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        # Store every passage, query and its answer to the lists
                        texts.append(context)
                        queries.append(question)
                        answers.append(answer)
        
        # Randomly shuffle them
        temp = list(zip(texts, queries, answers))
        random.seed(seed)
        random.shuffle(temp)
        texts, queries, answers = zip(*temp)

        return texts, queries, answers

    def find_end_positions(self, texts, answers):
        """
        Find accurate ending positions for answers in the context.
        """
        for answer, text in zip(answers, texts):
            real_answer = answer['text']
            start_idx   = answer['answer_start']
            end_idx     = start_idx + len(real_answer)
            # Take all cases into consideration
            if text[start_idx:end_idx] == real_answer:
                answer['answer_end'] = end_idx
            elif text[start_idx-1:end_idx-1] == real_answer:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  
            elif text[start_idx-2:end_idx-2] == real_answer:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2
            else:
                print("Couldn't find stuff")
                print(text)
                print(real_answer)
                print(start_idx)
                print(text[start_idx:end_idx])

    def tokenize(self, train_texts=None, train_queries=None, val_texts=None, val_queries=None):
        """
        Tokenize the respective data provided.
        """
        if train_texts is None:
            train_texts = self.train_texts
        
        if train_queries is None:
            train_queries = self.train_queries
    
        if val_texts is None:
            val_texts = self.val_texts

        if val_queries is None:
            val_queries = self.val_queries

        train_encodings = self.tokenizer.tokenizer(train_texts,
                                                   train_queries,
                                                   truncation=True,
                                                   padding=True)
        val_encodings = self.tokenizer.tokenizer(val_texts,
                                                 val_queries,
                                                 truncation=True,
                                                 padding=True)
        
        return train_encodings, val_encodings

    def add_token_positions(self, encodings, answers):
        """
        Find the respective ending positions in tokenized space.
        """
        start_positions = []
        end_positions   = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.tokenizer.model_max_length    
            # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
            if end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            # if end position is still None the answer passage has been truncated
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.tokenizer.model_max_length
        # Update the data in dictionary
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
