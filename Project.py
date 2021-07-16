from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

top_words_list1 = []
top_doc_freq = []
top_doc_freq_index = []
list2 = []
list3 = []
list4 = []
list5 = []
list_copy = []
a = 0
list11 = []
list12 = []

list21 = []
list22 = []
list23 = []

list31 = []
list32 = []
list33 = []

data = pd.read_csv('training2.csv', encoding="ISO-8859-1")
column6 = pd.DataFrame(data['Column6'])
all_stopwords = stopwords.words('english')
sw_list = ['I', '.', ',', '...', '!', "'s", '&', '*',
           'u', 'U', '?', "n't", '..', 'Is', '-', "'m",
           '#', "'ve", ':', 'd', 'D', '4', '--', '/',
           '0g', 'It', 'im', '....', "'", "'d", '(',
           ';', 'IM', 'wo', 'ca', 'Ca', '[', 'Im', '2',
           'A', '@', 'IS', 'cant', 'r', 'rt', 'If', 'Um',
           'Mo', '.............', '......', '~', '$']

all_stopwords.extend(sw_list)

for idx, row in column6.iterrows():
    text_tokens = re.sub('@[^\s]+', '', row['Column6'])
    text_tokens = word_tokenize(text_tokens)

    token_without_sw = [word for word in text_tokens if not word in all_stopwords]
    token_without_sw = list(map(lambda x: x.lower(), token_without_sw))
    list3.append(token_without_sw)

    # Counts of words in every document
    for i in token_without_sw:
        list2.append(token_without_sw.count(i))

    if (len(list2) == 0):
        continue
    else:
        max_list2 = (max(list2))
        max_list2_index = list2.index(max_list2)

        # List of top words
        top_words_list1.append(token_without_sw[max_list2_index])
        # print(top_words_list1)
        list2.clear()

# Document frequency of top words in list 5
for i in top_words_list1:
    for j in list3:
        for k in j:
            if (i == k):
                a = a + 1
                list4.append(j)
                break
    list5.append(a)
    list_copy.append(a)
    a = 0

data2 = []
# Top 50 Document frequency words
for i in range(50):
    top_freq = max(list_copy)
    data2.append(top_freq)
    top_freq_index = list5.index(top_freq)
    freq_word = top_words_list1[top_freq_index]
    top_doc_freq.append(freq_word)
    list_copy.remove(top_freq)

# Removing similar words in top_doc_freq list
# Selecting unique values
sep = set(top_doc_freq)
separate = (list(sep))
# print(separate)

separate2 = []
for i in separate:
    for j in top_words_list1:
        if i == j:
            separate2.append(top_words_list1.index(j))
            break
# print(separate2)

# List 11 = Document Frequency
# List 12 = All Documents

Sum = 0
start = 0
end = 0
counter = 0
for i in separate2:
    for j in list5:
        if (counter != i):
            Sum = j + Sum
            counter = counter + 1
        else:
            start = Sum
            end = j + Sum
            list11.append(j)
            for k in list4[start: end]:
                list12.append(k)
            Sum = 0
            counter = 0
            break

# print(list11)
# print(list12)


# De-Tokenizing
start2 = 0
end2 = 0
doc2 = ""
doc3 = []

for i in list11:
    end2 = end2 + i
    for j in list12[start2:end2]:
        doc = TreebankWordDetokenizer().detokenize(j)
        if doc2 == "":
            doc2 = doc
        else:
            doc2 = doc2 + " " + doc

    doc3.append(doc2)
    start2 = start2 + i

# Counting all words
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(doc3)
doc_term_matrix = sparse_matrix.todense()

df = pd.DataFrame(doc_term_matrix,
                  columns=count_vectorizer.get_feature_names())

# cosine_similarity
cos = cosine_similarity(df, df)
list_cos = cos.tolist()

index = -1
maximum = 0
loop = 0

# Max & Min cosine_similarity
for i in list_cos:
    for j in i:
        index = index + 1
        if loop == i.index(j):
            continue
        else:
            if j > maximum:
                maximum = j
                list21.append(index)

    loop = loop + 1
    index = -1
    maximum = 0
    list22.append(list21[-1])
    list21.clear()

for i in list22:
    list23.append(separate[i])


some_list = []
some_list2 = []
some_list3 = []
minimum = 100

for i in list_cos:
    for j in i:
        index = index + 1
        if j < minimum:
            minimum = j
            some_list.append(index)

    index = -1
    minimum = 100
    some_list2.append(some_list[-1])
    some_list.clear()

for i in some_list2:
    some_list3.append(separate[i])

# print(list22)
# print(list23)
# print(some_list2)
# print(some_list3)


# Incomplete Data
count = 0
count2 = 0
begin = 0
fin = 0

for i in list11:
    fin = fin + i
    for j in list12[begin : fin]:
        for k in j:
            if k == list23[count] or k == some_list3[count]:
                j[count2] = "?"

            count2 = count2 + 1

        count2 = 0
        list31.append(j)
        # print(j)

    count = count + 1
    begin = begin + i
    # print("-----------------------------------------------------------------------------------------------------------------------------------")
    # print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    # print("-----------------------------------------------------------------------------------------------------------------------------------")

# print(list31)
w = 0
w2 = 0

for i in list31:
    for j in i:
        if j == "?":
            w = w + 1   # w = 1
            break

    if w == 0:
        for k in i:
            if k in separate:
                w2 = w2 + 1
            else:
                i[w2] = "?"
                break

    w = 0
    w2 = 0

# print(list31)

s1 = 8
s2 = 2
start = 0
end = 0
count = 0
count2 = 0
for i in list11:
    end = end + i
    for j in list31[start:end]:
        for k in j:
            if k == "?":
                if s1 == 3 and s2 == 7:
                    s1 = 8
                    s2 = 2

                if not(s2 == 2):
                    j[count] = list23[count2]
                    s2 = s2 + 1
                    s1 = s1 - 1
                else:
                    j[count] = some_list3[count2]
                    s2 = s2 + 1
                    s1 = s1 - 1

            count = count + 1
        count = 0
    count2 = count2 + 1

print(list31)





























