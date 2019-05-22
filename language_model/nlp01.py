from nltk.tokenize import word_tokenize
import math

with open('C:\\Users\\19843\\Desktop\\natural_language_processing\\Experiment1\\train_LM.txt', 'rb') as f:
    trainText = f.read().decode("utf8").replace('__eou__', '')

w1 = word_tokenize(trainText)
# print('word_tokenize of trainText: ', w1)

word_num = len(w1)
trainText_dict = {}
for word1 in w1:
    if word1 not in trainText_dict.keys():
        trainText_dict[word1] = 1
    else:
        trainText_dict[word1] += 1
# print(trainText_dict)



with open('C:\\Users\\19843\\Desktop\\natural_language_processing\\Experiment1\\test_LM.txt', 'rb') as f:
    testText = f.read().decode("utf8").replace('__eou__', '')

w2 = word_tokenize(testText)
# print('word_tokenize of testText: ', w2)

w2word = list(set(w2))
smoothing = False
for word in w2:
    if word not in trainText_dict.keys():
        smoothing = True
        break

if smoothing:
    for word in w2word:
        if word not in trainText_dict.keys():
            trainText_dict[word] = 1
        else:
            trainText_dict[word] = trainText_dict[word] + 1

word_num = word_num + len(w2word)

perplexity = 0
for word in w2:
    perplexity += math.log2(trainText_dict[word]/word_num)   #乘法算出来太小，须用加法

print(perplexity)
perplexity = pow(2, perplexity * (-1/len(w2)))
print('perplexity = ', perplexity)

