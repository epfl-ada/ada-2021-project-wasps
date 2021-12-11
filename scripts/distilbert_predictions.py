# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from IPython.display import clear_output
from tqdm import tqdm
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.quoteID = dataframe.quoteID
        self.text = dataframe.quotation
        self.targets = self.data.label
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
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.long),
            'quoteID': self.quoteID.iloc[index]
        }


class DistilBERTClass(torch.nn.Module):
    def __init__(self, distilBERT, num_classes):
        super(DistilBERTClass, self).__init__()
        self.embedding = distilBERT
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.embedding(input_ids=input_ids, attention_mask=attention_mask)[0]
        embed = output_1[:, 0]
        out = self.pre_classifier(embed)
        out = torch.nn.ReLU()(out)
        out = self.dropout(out)
        output = self.classifier(out)
        # output = self.softmax(output)
        return output, embed

def train_model(model, training_loader, loss_, optimizer, scheduler, epoch):
    losses = []
    model.train()
    for i, data in tqdm(enumerate(training_loader, 0)):
        optimizer.zero_grad()
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        outputs, _ = model(ids, mask)
        loss = loss_(outputs, targets)
        if i % 100 == 0:
          clear_output(wait=True)
          print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()
    return np.mean(losses)

def validation(model, testing_loader, loss, classes, num_classes):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    fin_targets=[]
    fin_outputs=[]
    fin_embeds = []
    losses = []
    with torch.no_grad():
        for data in tqdm(testing_loader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs, embed = model(ids, mask)
            losses.append(loss(outputs, targets).item())
            outputs = softmax(outputs)
            fin_embeds.extend(np.array(embed.cpu().detach()))
            fin_targets.extend(np.array(targets.cpu().detach()))
            fin_outputs.extend(np.argmax(outputs.cpu().detach().numpy().tolist(), axis=1))
    
    print('Plotting...')
    plot_multiclass_roc(fin_outputs, fin_targets, classes)
    plot_confusion_matrix(fin_outputs, fin_targets, classes, cmap=None, normalize=True)
    # cluster_plotting(fin_embeds)
    
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, labels=np.ndarray(num_classes), average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, labels=np.ndarray(num_classes), average='macro')
    f1_score_weighted = metrics.f1_score(fin_targets, fin_outputs, labels=np.ndarray(num_classes), average='weighted')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"F1 Score (weighted) = {f1_score_weighted}")
    return accuracy, np.mean(losses)
    
def test(model, none_loader, classes, feature_name):
    softmax = torch.nn.Softmax(dim=1)
    classes = np.array(classes)
    path = feature_name + '_predictions.csv'
    df = {'quoteID': [], feature_name: []}
    model.eval()
    with torch.no_grad():
        for data in tqdm(none_loader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            df['quoteID'].extend(data['quoteID'])
            outputs, _ = model(ids, mask)
            outputs = softmax(outputs)
            outputs = outputs.cpu().detach().numpy().tolist()
            for output in outputs:
                output = np.array(output)
                if feature_name == 'gender':
                    inds = np.argmax(output)
                else:
                    inds = np.argsort(output)[-3:][::-1]
                classes_ = classes[inds]
                probs = output[inds]
                feature = list(zip(classes_, probs))
                df[feature_name].append(feature)
    df = pd.DataFrame(df)
    df.to_csv(path)
    
def cluster_plotting(embeddings):
    # embeddings = embeddings[:10]
    plt.figure(figsize=(8, 6))
    pca = PCA(n_components=50)
    h_i = pca.fit(embeddings).transform(embeddings)
    h_i = TSNE(n_components=2, learning_rate=200, init='pca').fit_transform(h_i)
    for point in h_i:
        plt.scatter(point[0], point[1])
    plt.savefig('clusters.png')
    
def plot_confusion_matrix(outputs, targets, target_names, cmap=None, normalize=True):
    """
    Arguments
    ---------
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    """
    cm = confusion_matrix(targets, outputs)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.png')
    
def plot_multiclass_roc(outputs, targets, classes, figsize=(8, 6)):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # calculate dummies once
    y_test_dummies = pd.get_dummies(targets, drop_first=False).values
    y_pred_dummies = pd.get_dummies(outputs, drop_first=False).values
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred_dummies[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(len(classes)):
        ax.plot(fpr[i], tpr[i], label = classes[i]+' (area = %0.2f)' % roc_auc[i])
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.savefig('roc_curve.png')
    
def plot_losses(loss_tr, loss_te, acc):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('train and test losses', fontsize=20)
  axes[0].set_xlabel('epoch', fontsize=20)
  axes[1].set_ylabel('test accuracy', fontsize=20)
  axes[1].set_xlabel('epoch', fontsize=20)
  axes[0].plot(loss_tr, label='train')
  axes[0].plot(loss_te, label='test')
  axes[1].plot(acc, label='Accuracy')
  axes[0].legend(prop={'size': 15})
  axes[1].legend(prop={'size': 15})
  axes[0].grid()
  axes[1].grid()
  plt.savefig('train_test_losses.png')

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, dest="data_path", help="Path to data folder", required=True)
	parser.add_argument("--feature_name", type=str, dest="feature_name", help="Name of the feature to be predicted", required=True)
	parser.add_argument("--freeze_bert", type=bool, dest="freeze_bert", help="Freezing BERT weights", default=True)
	parser.add_argument("--model_path", type=str, dest="model_path", help="Path to model weights for loading", default=None)
	parser.add_argument("--save_path", type=str, dest="save_path", help="Path to model weights for saving", default='model_new.pth')
	parser.add_argument("--epochs", type=int, dest="epochs", help="Amount of the epochs", default=10)
	parser.add_argument("--test_only", type=bool, dest="test_only", help="Training or validation modes", default=False)
	parser.add_argument("--prediction_mode", type=bool, dest="prediction_mode", help="Predict labels for real test", default=False)
	args = parser.parse_args()

	"""# Loading and preparing data"""
	print('Loading data...')
	MODEL_PATH = args.model_path
	TEST_ONLY = args.test_only
	PREDICTION_MODE = args.prediction_mode
	feature_name = args.feature_name
	SAVE_PATH = feature_name + '_' + args.save_path
	print('Feature to be predicted:', feature_name)
	if feature_name in ['gender']:
	    num_classes = 2
	elif feature_name in ['occupation', 'religion', 'ethnic_group']:
	    num_classes = 10
	elif feature_name in ['date_of_birth']:
	    num_classes = 8
	elif feature_name in ['nationality']:
	    num_classes = 5
	print('Number of classes:', num_classes)
	
	DATA_PATH = os.path.join(args.data_path, feature_name)
	dataset_train = ds.dataset(os.path.join(DATA_PATH, 'train.parquet'), format="parquet")
	df_train = dataset_train.to_table(columns = ['quoteID', 'quotation', feature_name]).to_pandas()
	dataset_val = ds.dataset(os.path.join(DATA_PATH, 'validation.parquet'), format="parquet")
	df_val = dataset_val.to_table(columns = ['quoteID', 'quotation', feature_name]).to_pandas()
	dataset_test = ds.dataset(os.path.join(DATA_PATH, 'test.parquet'), format="parquet")
	df_test = dataset_test.to_table(columns = ['quoteID', 'quotation', feature_name]).to_pandas()
	
	if PREDICTION_MODE:
	    TEST_ONLY = True
	    dataset_none = ds.dataset(os.path.join(DATA_PATH, 'none.parquet'), format="parquet")
	    df_none = dataset_none.to_table(columns = ['quoteID', 'quotation']).to_pandas()
	    df_none['label'] = -1
    
	labels = df_train[feature_name].value_counts()[:num_classes].index.tolist()
	train_dataset = df_train.assign(label = pd.Series(pd.factorize(df_train[feature_name], sort = True)[0]).values)
	val_dataset = df_val.assign(label = pd.Series(pd.factorize(df_val[feature_name], sort = True)[0]).values)
	test_dataset = df_test.assign(label = pd.Series(pd.factorize(df_test[feature_name], sort = True)[0]).values)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	_, test_dataset = train_test_split(test_dataset, test_size=0.01)
	
	classes = pd.factorize(train_dataset[feature_name], sort = True)[1].tolist()
	print('Classes:', classes)
	weights = []
	min_value = (train_dataset[feature_name].value_counts()).min()
	for clas in classes:
	    weights.append(min_value/train_dataset[feature_name].value_counts().loc[clas])
	weights = torch.Tensor(weights)

	"""# Dataloader"""
	# max_len = data_qoutations.quotation.str.split().str.len().max()
	MAX_LEN = 32
	TRAIN_BATCH_SIZE = 64
	VALID_BATCH_SIZE = 64

	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

	training_set = MultiLabelDataset(train_dataset, tokenizer, MAX_LEN)
	validation_set = MultiLabelDataset(val_dataset, tokenizer, MAX_LEN)
	test_set = MultiLabelDataset(test_dataset, tokenizer, MAX_LEN)
	train_params = {'batch_size': TRAIN_BATCH_SIZE,
		        'shuffle': False
		        }

	test_params = {'batch_size': VALID_BATCH_SIZE,
		        'shuffle': False
		        }

	training_loader = DataLoader(training_set, **train_params)
	validation_loader = DataLoader(validation_set, **test_params)
	test_loader = DataLoader(test_set, **test_params)
	if PREDICTION_MODE:
	    none_set = MultiLabelDataset(df_none, tokenizer, MAX_LEN)
	    none_loader = DataLoader(none_set, **test_params)
	"""# Loading model"""
    
	distilBERT = DistilBertModel.from_pretrained("distilbert-base-uncased")

	model = DistilBERTClass(distilBERT, num_classes)
	
	if MODEL_PATH is not None:
	    print('Loading model checkpoint...')
	    model.load_state_dict(torch.load(MODEL_PATH))
	
	if args.freeze_bert:
		print('Freeze BERT parameters')
		for parameter in model.embedding.parameters():
		    parameter.requires_grad = False
	
	model = model.to(device)
	
	if not TEST_ONLY:
	    """# Training"""
	    EPOCHS = args.epochs
	    LEARNING_RATE = 1e-04
	    
	    loss_nll = torch.nn.CrossEntropyLoss(weight=weights).to(device)
	   # loss_nll = torch.nn.NLLLoss(weight=weights).to(device)
	    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)
	    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
	    print('Training config:')
	    print('epochs:', EPOCHS)
	    print('lr:', LEARNING_RATE)
	    print('batch size:', TRAIN_BATCH_SIZE)
	    print('Training...')
	    loss_tr = []
	    loss_te = []
	    acc = []
	    for epoch in range(EPOCHS):
	        train_loss = train_model(model, training_loader, loss_nll, optimizer, scheduler, epoch)
	        loss_tr.append(train_loss)
	        # Evaluation results
	        print('Validation...')
	        accuracy, val_loss = validation(model, validation_loader, loss_nll, classes, num_classes)
	        loss_te.append(val_loss)
	        acc.append(accuracy)
	        plot_losses(loss_tr, loss_te, acc)
	        # Saving model
	        if len(loss_te) > 1:
	            if loss_te[-1] <= np.min(loss_te):
	                print("Saving best checkpoint...")
	                torch.save(model.state_dict(), SAVE_PATH)
	        elif len(loss_te) == 1:
	            print("Saving first checkpoint...")
	            torch.save(model.state_dict(), SAVE_PATH)
	else:
		# Evaluation results
		if PREDICTION_MODE:
		    print('Predicting...')
		    test(model, none_loader, classes, feature_name)
		else:
		    print('Testing...')
		    loss_nll = torch.nn.NLLLoss(weight=weights).to(device)
		    accuracy, val_loss = validation(model, test_loader, loss_nll, classes, num_classes)

if __name__ == "__main__":
	main()

