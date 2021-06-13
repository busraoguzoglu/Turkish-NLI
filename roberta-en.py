import torch
from fairseq.models.roberta import XLMRModel, RobertaModel

# roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
# xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr = XLMRModel.from_pretrained('xlmr.large', checkpoint_file='model.pt')
xlmr.eval()
# print(xlmr.eval())  # disable dropout (or leave in train mode to finetune)

# en_tokens = xlmr.encode('Hello world!')
# assert en_tokens.tolist() == [0, 35378,  8999, 38, 2]
# print(xlmr.decode(en_tokens)) # 'Hello world!'
#
# # Extract the last layer's features
# last_layer_features = xlmr.extract_features(en_tokens)
# assert last_layer_features.size() == torch.Size([1, 5, 1024])
#
# # Extract all layer's features (layer 0 is the embedding layer)
# all_layers = xlmr.extract_features(en_tokens, return_all_hiddens=True)
# assert len(all_layers) == 25
# assert torch.all(all_layers[-1] == last_layer_features)

roberta_large_mnli = RobertaModel.from_pretrained('roberta.large.mnli', checkpoint_file='model.pt')
roberta_large_mnli.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta_large_mnli.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
roberta_large_mnli.decode(tokens)  # 'Hello world!'

# Extract the last layer's features
last_layer_features = roberta_large_mnli.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta_large_mnli.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)

# Encode a pair of sentences and make a prediction
tokens = roberta_large_mnli.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
roberta_large_mnli.predict('mnli', tokens).argmax()  # 0: contradiction
print(roberta_large_mnli.predict('mnli', tokens).argmax())

# Encode another pair of sentences
tokens = roberta_large_mnli.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
roberta_large_mnli.predict('mnli', tokens).argmax()  # 2: entailment
print(roberta_large_mnli.predict('mnli', tokens).argmax())

#Evaluating roberta large mnli
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
# roberta_large_mnli.cuda()
roberta_large_mnli.eval()
with open('MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = roberta_large_mnli.encode(sent1, sent2)
        prediction = roberta_large_mnli.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# Expected output: 0.9060, obtained output:0.9059602649006623

