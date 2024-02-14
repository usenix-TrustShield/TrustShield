from transformers import BertTokenizerFast, DebertaTokenizerFast

class CustomTokenizer:
    def __init__(self, max_seq_len, deberta=False) -> None:
        if not deberta:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.max_seq_len = max_seq_len

    def tokenize_text(self, text):
        return  self.tokenizer.batch_encode_plus(
                    text.tolist(),
                    max_length=self.max_seq_len,
                    pad_to_max_length=True,
                    truncation=True,
                    return_token_type_ids=False
                )