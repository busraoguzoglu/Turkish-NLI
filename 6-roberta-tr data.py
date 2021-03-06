import torch
from fairseq.models.roberta import XLMRModel, RobertaModel

roberta_large_mnli = RobertaModel.from_pretrained('roberta.large.mnli', checkpoint_file='model.pt')
roberta_large_mnli.eval()  # disable dropout (or leave in train mode to finetune)

# tokens = roberta_large_mnli.encode('Hello world!')
# assert tokens.tolist() == [0, 31414, 232, 328, 2]
# roberta_large_mnli.decode(tokens)  # 'Hello world!'
#
# # Extract the last layer's features
# last_layer_features = roberta_large_mnli.extract_features(tokens)
# assert last_layer_features.size() == torch.Size([1, 5, 1024])
#
# # Extract all layer's features (layer 0 is the embedding layer)
# all_layers = roberta_large_mnli.extract_features(tokens, return_all_hiddens=True)
# assert len(all_layers) == 25
# assert torch.all(all_layers[-1] == last_layer_features)


#Evaluating roberta large mnli
#label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
ncorrect, nsamples = 0, 0
# roberta_large_mnli.cuda()
roberta_large_mnli.eval()
with open('multinli-tr/valid_matched_noIndex.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        # print("at line ", index)
        tokens = line.strip().split('\t')
        # sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        sent1, sent2, target = tokens[1], tokens[2], tokens[0]
        tokens = roberta_large_mnli.encode(sent1, sent2)
        prediction = roberta_large_mnli.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        # ncorrect += int(prediction_label == target)
        # ncorrect += int(prediction == target)
        if (int(prediction) == int(target)):
            ncorrect = ncorrect + 1
        print("prediction: ", prediction)
        print("target: ", target)
        print("correct pred num: ", ncorrect)

        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))

#ROBERTA MNLI ILE TURKCE DENERSEK
#training data[0]
tokens = roberta_large_mnli.encode('??r??n ve co??rafya krem kayma????n?? i??e yar??yor.', 'Kavramsal olarak krem kayma????n??n iki temel boyutu vard??r - ??r??n ve co??rafya.')
roberta_large_mnli.predict('mnli', tokens).argmax()  #1 predict etti - datasette:1
print(roberta_large_mnli.predict('mnli', tokens).argmax())

#training data[1]
tokens = roberta_large_mnli.encode('E??er insanlar hat??rlarsa, bir sonraki seviyeye d????ersin.', 'Mevsim boyunca ve san??r??m senin seviyendeyken onlar?? bir sonraki seviyeye d??????r??rs??n. E??er ebeveyn tak??m??n?? ??a????rmaya karar verirlerse Braves ????l?? A\'dan birini ??a????rmaya karar verirlerse ??ifte bir adam onun yerine ge??meye gider ve bekar bir adam gelir.')
roberta_large_mnli.predict('mnli', tokens).argmax()  # 2 predict etti - datasette:0
print(roberta_large_mnli.predict('mnli', tokens).argmax())

#training data[2]
tokens = roberta_large_mnli.encode('Ekibimin bir ??yesi emirlerinizi b??y??k bir hassasiyetle yerine getirecektir.', 'Numaram??zdan biri talimatlar??n??z?? birazdan yerine getirecektir.')
roberta_large_mnli.predict('mnli', tokens).argmax()  # 2 predict etti - datasette:0
print(roberta_large_mnli.predict('mnli', tokens).argmax())

#contradiction 2
#entailment 0
#neutral 1
