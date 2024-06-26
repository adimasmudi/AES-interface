import pandas as pd
import re
import string
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold,train_test_split
from tqdm import tqdm
import numpy as np
import time

answer = 'gak tau saya bu'
ReferenceAnswer = "5 Dampak Negatif Perkembangan Teknologi:\nKetergantungan Berlebihan: Orang-orang menjadi terlalu bergantung pada teknologi dan kehilangan kemampuan untuk menyelesaikan masalah sendiri.\nKecanduan Gadget: Penggunaan gadget yang berlebihan dapat menyebabkan kecanduan dan mengganggu kesehatan mental dan fisik.\nKesenjangan Digital: Tidak semua orang memiliki akses yang sama terhadap teknologi, sehingga terjadi kesenjangan antara yang kaya dan yang miskin.\nKejahatan Siber: Teknologi dapat digunakan untuk melakukan kejahatan, seperti penipuan, pencurian data, dan penyebaran berita bohong.\nPenyalahgunaan Informasi: Informasi yang beredar di internet tidak selalu akurat dan dapat disalahgunakan untuk menyebarkan kebencian dan propaganda."

# Dataset Creation (Tokenizer)
class EssayDataTokenizer(Dataset):
    def __init__(self, answers, students, tokenizer, max_length):
        self.answers = answers
        self.students = students
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        answer = self.answers[idx]
        student = self.students[idx]

        answer_encoding = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        student_encoding = self.tokenizer.encode_plus(
            student,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'answer_text' : answer,
            'answer_input_ids': answer_encoding['input_ids'].flatten(),
            'answer_attention_mask': answer_encoding['attention_mask'].flatten(),
            'student_text' : student,
            'student_input_ids': student_encoding['input_ids'].flatten(),
            'student_attention_mask': student_encoding['attention_mask'].flatten(),
        }

class IndoBERTForSTS(nn.Module):
    def __init__(self, bert_model):
        super(IndoBERTForSTS, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[1]

# Function to calculate cosine similarity using PyTorch operations
def cosine_sim(a, b):
    cos_sim = nn.CosineSimilarity(dim=1)
    return cos_sim(a, b)

# function to test the model
def test_model(model, data_loader,device):
    with torch.no_grad():
        for d in data_loader:
            answer_input_ids = d["answer_input_ids"].to(device)
            answer_attention_mask = d["answer_attention_mask"].to(device)
            student_input_ids = d["student_input_ids"].to(device)
            student_attention_mask = d["student_attention_mask"].to(device)

            answer_outputs = model(
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask
            )

            student_outputs = model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask
            )

            cosine_scores = cosine_sim(answer_outputs, student_outputs)

    return cosine_scores

def clear(text):
      # Replace punctuations with space
      text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

      # Clear multiple spaces
      text = re.sub(r'\s+', ' ', text).strip()

      # Replace newlines with space
      text = text.replace("\n", " ")

      return text.lower()

def runExperiment(tokenizer, bert_model,device):
    # create essay dataset of test
    test_dataset = EssayDataTokenizer(
        [clear(ReferenceAnswer)],
        [clear(answer)],
        tokenizer,
        512
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=True
    )


    test_result = test_model(bert_model, test_data_loader,device)

    return test_result

# Load IndoBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p2')

model = IndoBERTForSTS(bert_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used :',device)
model.to(device)

result = runExperiment(tokenizer=tokenizer, bert_model=model,device=device)
print('result', result)
