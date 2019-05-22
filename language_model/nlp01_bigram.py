from nltk.tokenize import word_tokenize
import math

with open('C:\\Users\\19843\\Desktop\\natural_language_processing\\Experiment1\\train_LM.txt', 'rb') as f:
    trainText = f.read().decode("utf8")
        # .replace('__eou__', '<B&EOS>')

w2 = word_tokenize(trainText)
# print('word_tokenize of trainText: ', w1)

#去除标点
# punctuation = [',', '.', '!', '?', ';']
# w2 = [i for i in w1 if i not in punctuation]

bigram_list1 = []
bigram_list1.append('__eou__' + '_' + w2[0])

for i in range(len(w2)-1):
    bigram_list1.append(w2[i] + '_' + w2[i+1])

print(bigram_list1)
bigram_num = len(bigram_list1)
trainText_dict = {}
for b in bigram_list1:
    if b not in trainText_dict.keys():
        trainText_dict[b] = 1
    else:
        trainText_dict[b] +=1



with open('C:\\Users\\19843\\Desktop\\natural_language_processing\\Experiment1\\test_LM.txt', 'rb') as f:
    testText = f.read().decode("utf8")
        # .replace('__eou__', '<B&EOS>')

w4 = word_tokenize(testText)
# w4 = [i for i in w3 if i not in punctuation]

bigram_list2 = []
bigram_list2.append('__eou__' + '_' + w4[0])

for i in range(len(w4)-1):
    bigram_list2.append(w4[i] + '_' + w4[i+1])

bigram_list2word = list(set(bigram_list2))
smoothing = False
for b in bigram_list2:
    if b not in trainText_dict:
        smoothing = True
        break

if smoothing:
    for b in bigram_list2word:
        if b not in trainText_dict.keys():
            trainText_dict[b] = 1
        else:
            trainText_dict[b] = trainText_dict[b] + 1

bigram_num = bigram_num + len(bigram_list2word)

perplexity = 0
for b in bigram_list2:
    perplexity += math.log2(trainText_dict[b]/bigram_num)

print(perplexity)
perplexity = pow(2, perplexity * (-1/len(bigram_list2)))
print('perplexity = ', perplexity)

