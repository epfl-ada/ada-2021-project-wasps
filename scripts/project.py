# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from tqdm import tqdm
import argparse

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.quotation
        self.targets = self.data.label_num
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
	    # padding='max_length',
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        # mask = inputs['special_tokens_mask']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.long)
        }


class DistilBERTClass(torch.nn.Module):
    def __init__(self, distilBERT, num_classes):
        super(DistilBERTClass, self).__init__()
        self.embedding = distilBERT
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.embedding(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooler = output_1[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def train_model(model, training_loader, loss_CE, optimizer, epoch):
    model.train()
    for i, data in tqdm(enumerate(training_loader, 0)):
        optimizer.zero_grad()
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        outputs = model(ids, mask)
        loss = loss_CE(outputs, targets)
        if i % 100 == 0:
          clear_output(wait=True)
          print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        loss.backward()
        optimizer.step()

def validation(model, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets']
            outputs = model(ids, mask)
            fin_targets.extend(np.array(targets))
            fin_outputs.extend(np.argmax(outputs.cpu().detach().numpy().tolist(), axis=1))
    return fin_outputs, fin_targets

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, dest="data_path", help="Path to data folder", required=True)
	parser.add_argument("--num_classes", type=int, dest="num_classes", help="Amount of clasfier classes", default=10)
	parser.add_argument("--freeze_bert", type=bool, dest="freeze_bert", help="Freezing BERT weights", default=True)
	parser.add_argument("--model_path", type=str, dest="model_path", help="Path to model weights for loading", default=None)
	parser.add_argument("--save_path", type=str, dest="save_path", help="Path to model weights for saving", default='model_new.pth')
	parser.add_argument("--epochs", type=int, dest="epochs", help="Amount of the epochs", default=10)
	args = parser.parse_args()

	"""# Loading and preparing data"""

	DATA_PATH = args.data_path
	MODEL_PATH = args.model_path
	SAVE_PATH = args.save_path
	df = pd.read_csv(DATA_PATH)

	num_classes = args.num_classes
	labels = df.urls.value_counts()[:num_classes].index.tolist()

	data_qoutations = df[df.urls.isin(labels)]
	data_qoutations = data_qoutations.rename(columns={'urls': 'label'})
	data_qoutations.reset_index(drop=True, inplace=True)
	data_qoutations = data_qoutations.assign(label_num = pd.Series(pd.factorize(data_qoutations['label'])[0]).values)

	"""# Dataloader"""
	# max_len = data_qoutations.quotation.str.split().str.len().max()
	MAX_LEN = 64
	TRAIN_BATCH_SIZE = 32
	VALID_BATCH_SIZE = 32

	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
	train_dataset, test_dataset = train_test_split(data_qoutations, test_size=0.2)

	training_set = MultiLabelDataset(train_dataset, tokenizer, MAX_LEN)
	testing_set = MultiLabelDataset(test_dataset, tokenizer, MAX_LEN)
	train_params = {'batch_size': TRAIN_BATCH_SIZE,
		        'shuffle': True
		        }

	test_params = {'batch_size': VALID_BATCH_SIZE,
		        'shuffle': True
		        }

	training_loader = DataLoader(training_set, **train_params)
	testing_loader = DataLoader(testing_set, **test_params)

	"""# Loading model"""
    
	distilBERT = DistilBertModel.from_pretrained("distilbert-base-uncased")

	model = DistilBERTClass(distilBERT, num_classes)
	
	if MODEL_PATH is not None:
	    model.load_state_dict(torch.load(MODEL_PATH))
	
	if args.freeze_bert:
		for parameter in model.embedding.parameters():
		    parameter.requires_grad = False
	
	model = model.to(device)

	"""# Training"""

	EPOCHS = args.epochs
	LEARNING_RATE = 1e-04

	loss_CE = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
	optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

	for epoch in range(EPOCHS):
	    train_model(model, training_loader, loss_CE, optimizer, epoch)
	    # Evaluation results
	    outputs, targets = validation(model, testing_loader)
	    accuracy = metrics.accuracy_score(targets, outputs)
	    f1_score_micro = metrics.f1_score(targets, outputs, labels=np.ndarray(num_classes), average='micro')
	    f1_score_macro = metrics.f1_score(targets, outputs, labels=np.ndarray(num_classes), average='macro')
	    print(f"Accuracy Score = {accuracy}")
	    print(f"F1 Score (Micro) = {f1_score_micro}")
	    print(f"F1 Score (Macro) = {f1_score_macro}")
	    print("Saving checkpoint")
	    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
	main()

