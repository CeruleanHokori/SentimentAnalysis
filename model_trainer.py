from data_proc import samples_labels
import numpy as np
import torch
from tqdm import trange
from transformers import BertTokenizer,BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler,RandomSampler
from metrics import b_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_file = "data/twitter_training.csv"




#Data samples and labels for training
X_train,y_train = samples_labels(training_file)

#Tokenizer
tokenizer = tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )
#Preprocessing text
def preprocessing(input_text, tokenizer):
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 280, #twitter max length of 280
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

#Ids and masks
token_id,attention_masks = [],[]

for tweet in X_train:
    encoding_dict = preprocessing(tweet,tokenizer)
    token_id.append(encoding_dict['input_ids'])
    attention_masks.append(encoding_dict['attention_mask'])
token_id = torch.cat(token_id,dim=0)
attention_masks = torch.cat(attention_masks,dim=0)
y_train = torch.tensor(y_train)

#Data splitting
ratio,batch_size = 0.2,16

train_idx, val_idx = train_test_split(
    np.arange(len(y_train)),
    test_size = ratio,
    shuffle = True,
    stratify = y_train)

# Train and validation sets
train_set = TensorDataset(token_id[train_idx], 
                          attention_masks[train_idx], 
                          y_train[train_idx])

val_set = TensorDataset(token_id[val_idx], 
                        attention_masks[val_idx], 
                        y_train[val_idx])
#Loaders
train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )


#MODEL

# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# Run on GPU
model.cuda()

if __name__ == "__main__":
    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    epochs = 3

    for _ in trange(epochs, desc = 'Epoch'):
        
        # ========== Training ==========
        
        # Set model to training mode
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_labels)
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output.loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        # ========== Validation ==========

        # Set model to evaluation mode
        model.eval()

        # Tracking variables 
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                eval_output = model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask)
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)

        print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
    torch.save(model,"sentiment_v0")