import torch.nn as nn
import re
import string

class IndoBERTForSTS(nn.Module):
    def __init__(self, bert_model):
        super(IndoBERTForSTS, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

class IndoBERTDatasetTokenizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, answer, student):
        answer_encoding = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        student_encoding = self.tokenizer.encode_plus(
            student,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'answer_text': answer_encoding,
            'answer_input_ids': answer_encoding['input_ids'].flatten(),
            'answer_attention_mask': answer_encoding['attention_mask'].flatten(),
            'student_text': student_encoding,
            'student_input_ids': student_encoding['input_ids'].flatten(),
            'student_attention_mask': student_encoding['attention_mask'].flatten(),
        }

class BERTScoring:
    def __init__(self, bert_model, bert_tokenizer):
        self.model = bert_model
        self.tokenizer = bert_tokenizer

    def __cleaning(self, text: str):
        # Replace punctuations with space
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # Clear multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Replace newlines with space
        text = text.replace("\n", " ")

        return text.lower()

    def __cosine_sim(self, a, b):
        cos_sim = nn.CosineSimilarity(dim=1)
        return cos_sim(a, b)

    def predict(self, reference_answer: str, student_answer: str):
        reference_answer = self.__cleaning(reference_answer)
        student_answer = self.__cleaning(student_answer)

        if not student_answer.strip():
            return [0]

        tokenizer = IndoBERTDatasetTokenizer(self.tokenizer)

        encoded_text = tokenizer.tokenize(reference_answer, student_answer)
        answer_input_ids = encoded_text["answer_input_ids"].unsqueeze(0)
        answer_attention_mask = encoded_text["answer_attention_mask"].unsqueeze(0)
        student_input_ids = encoded_text["student_input_ids"].unsqueeze(0)
        student_attention_mask = encoded_text["student_attention_mask"].unsqueeze(0)

        modelPrediction = IndoBERTForSTS(self.model)

        answer_embedding = modelPrediction(
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask
            )
        

        student_embedding = modelPrediction(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            )
        

        cosine_scores = self.__cosine_sim(answer_embedding, student_embedding)

        return cosine_scores.tolist()

