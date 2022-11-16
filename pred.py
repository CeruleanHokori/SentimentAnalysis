import torch
import numpy as np
from data_proc import decoder
from transformers import BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("sentiment_v0")

new_sentence = "I've waited in a queue for hours"

tokenizer = tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )
def preprocessing(input_text, tokenizer):
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 280, #twitter max length of 280
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )
# We need Token IDs and Attention Mask for inference on the new sentence
test_ids = []
test_attention_mask = []

# Apply the tokenizer
encoding = preprocessing(new_sentence, tokenizer)

# Extract IDs and Attention Mask
test_ids.append(encoding['input_ids'])
test_attention_mask.append(encoding['attention_mask'])
test_ids = torch.cat(test_ids, dim = 0)
test_attention_mask = torch.cat(test_attention_mask, dim = 0)

# Forward pass, calculate logit predictions
with torch.no_grad():
  output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

prediction = decoder(np.argmax(output.logits.cpu().numpy()).flatten().item())

print('Input Sentence: ', new_sentence)
print('Predicted Class: ', prediction)