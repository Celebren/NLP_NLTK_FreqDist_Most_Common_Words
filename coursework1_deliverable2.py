# This program fully complies with Pyhton2 syntax and PEP8 rules. The program is not compatible with Python3
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords

'''
Note about comments:
Comments with answers to questions begin with "Answer to x question" and only they should count towards comments word
limit for marking. All other comments provide important code documentation and should not be included to word limit for 
marking.
------------------------------------------------------------------------------------------------------------------------
using pandas open the text file as a dataframe (df), indicating tab separators (sep='\t') and naming the 
columns score and text
df = pd.read_csv('training.txt', sep='\t', names=['score', 'text'])

print df.head()
print df.tail()
print "length =", len(df)
------------------------------------------------------------------------------------------------------------------------
I abandon the pandas.read_csv() method as for some reason, it does not read into Python all lines of the training
dataset, so for accuracy I decide to read the file using Python's with/open/as method instead.
'''

# initialise lists
two_dimensional_list = []
words_list = []

with open("training.txt") as f:
    # read each line of the file
    for line in f:
        # splite the line by tabs and save into array
        split_words_list = line.split('\t')
        # append the array of each line into a two dimensional list
        two_dimensional_list.append(split_words_list)

'''
------------------------------------------------------------------------------------------------------------------------
Answer to first question:
I chose pandas dataframes to store the dataset into Python as dataframes are easy to use and the
ability to apply functions to the entire data frame makes the task a lot easier looping through a
traditional list or dictionary and appling functions in each loop. Pandas dataframes also display a lot nicer than
classic Python data collections. Pandas also contain useful built in functions.
------------------------------------------------------------------------------------------------------------------------
'''
# convert the two dimensional Python list into a pandas data frame with columns named score and text
df = pd.DataFrame(two_dimensional_list, columns=["score", "text"])

# define a list of non-letter characters to be removed from the data frame
patern_list_of_unwanted_characters = [
    '\,', '\.', '\\n', ':', '\*', '\(', '\)', "'", '\!', '\?', '"', '\/', '\~', '\-',
    '\[', '\]', '\=', '\&', '\<', '\>', '\^', '\$', '\\xa0', '\\x92', u'\u2018',
    u'\u2019', u'\ufffd', u'\u2013', u'\u201d', u'\xa3', u'\xa9', u'\xc3', u'\u201c', u'\u2026'
]

# using pandas' replace() function, replace unwanted characters from the list defined earlier, with an empty string
df = df.replace(patern_list_of_unwanted_characters, '', regex=True).astype(str)

# convert all capital letters to lower case using pandas' apply()
df = df.apply(lambda x: x.astype(str).str.lower())

# now that the dataset is clean of unwanted characters and all capital letters are converted into lowercase,
# "flatten" the dataframe into a simple list that will be tokenised and searched through for most common words
flat_list = df['text'].values.flatten()

# import NLTK's stopwords into a list.
stopwords_list = stopwords.words('english')
# create a list of aditional common stopwords that are missed by the NLTK stopwords list
# due to dataset's character encoding. These words were identified by running the NLTK FrewDist most_common() function
custom_stopwords_list = ["i", "a", "the", "it", "is", "one", "3", "am"]

# loop through the flat list and apply NLTK's tokenizer to tokenise each word in each sentence
tokenized_sentences = [word_tokenize(i) for i in flat_list]
# take each tokenised word in the entire data set and add it to a new list. This list will be searched
for sentence in tokenized_sentences:
    for word in sentence:
        words_list.append(word)


'''
I make the assumption that in this dataset, "da" will always be followed by "vinci" and "code", "harry" by "potter"
"brokeback" by "mountain" and "mission" by "impossible". I loop through the list of words in the dataset and each time
"da", "harry", "brokeback" and "mission" are found, they are replaced by "da vinci code", "harry potter", "brokeback 
mountain" and "mission impossible" respectivelly. To be safe, I also asssume that "harry" and "mission" are not always
followed by "potter" and "impossible" so those are only replaced if the string in the next index position matches my
expectations. Then the strings in the following indexes that were added to the new string, are removed from the list.
'''


def words_combiner(function_words_list):
    # initialise an index counter
    words_list_index = 0
    for function_word in function_words_list:
    
        if function_word == "da":
            function_word = "da vinci code"
            function_words_list[words_list_index] = function_word
            function_words_list.remove(function_words_list[words_list_index + 1])
            function_words_list.remove(function_words_list[words_list_index + 1])

        if function_word == "harry":
            if function_words_list[words_list_index + 1] == "potter":
                function_word = "harry potter"
                function_words_list[words_list_index] = function_word
                function_words_list.remove(function_words_list[words_list_index + 1])

        if function_word == "brokeback":
            function_word = "brokeback mountain"
            function_words_list[words_list_index] = function_word
            function_words_list.remove(function_words_list[words_list_index + 1])

        if function_word == "mission":
            if function_words_list[words_list_index + 1] == "impossible":
                function_word = "mission impossible"
                function_words_list[words_list_index] = function_word
                function_words_list.remove(function_words_list[words_list_index + 1])
            
        # increase the index by 1 each
        words_list_index += 1
    return function_words_list


# run combiner function to combine names and movie titles in dataset
# OR comment out to get results for individual words only
# Running this method REDUCES EXECUTION TIME
#words_list = words_combiner(words_list)

# create a copy of the words list that will be iterrated through and stopwords will be removed
filtered_word_list = words_list[:]
# iterate through the list
for word in filtered_word_list:  # iterate over word_list
    # the "word" string needs to be decoded with ISO-8859-1 as there are unexpected characters in the data that
    # prevent execution without decoding
    if word.decode('ISO-8859-1') in stopwords_list:
        # remove any word that exists in the stopwords list
        filtered_word_list.remove(word)

# repeat for words in custom list
for word in filtered_word_list:
    if word in custom_stopwords_list:
        filtered_word_list.remove(word)

# create nltk.FreqDist() of the final clean list of words with all unwanted words and symbols removed
fdist = FreqDist(filtered_word_list)
# create a list of the 20 most common words in the data set
most_common_words = fdist.most_common(20)

# convert the list of most common words into a data frame for nicer display.
most_common_df = pd.DataFrame(most_common_words, columns=["word", "occurences"])
# print the dataframe
print most_common_df

words_list = words_combiner(words_list)

# create a copy of the words list that will be iterrated through and stopwords will be removed
filtered_word_list = words_list[:]
# iterate through the list
for word in filtered_word_list:  # iterate over word_list
    # the "word" string needs to be decoded with ISO-8859-1 as there are unexpected characters in the data that
    # prevent execution without decoding
    if word.decode('ISO-8859-1') in stopwords_list:
        # remove any word that exists in the stopwords list
        filtered_word_list.remove(word)

# repeat for words in custom list
for word in filtered_word_list:
    if word in custom_stopwords_list:
        filtered_word_list.remove(word)

# create nltk.FreqDist() of the final clean list of words with all unwanted words and symbols removed
fdist = FreqDist(filtered_word_list)
# create a list of the 20 most common words in the data set
most_common_words = fdist.most_common(20)

# convert the list of most common words into a data frame for nicer display.
most_common_df = pd.DataFrame(most_common_words, columns=["word", "occurences"])
# print the dataframe
print most_common_df

'''
------------------------------------------------------------------------------------------------------------------------
Answer to second question:
- Most common word without combined words: "harry", occuring 2091 times
- Most common string with combined words: "harry potter", occuring 2086 times
The difference in occurences is because not every "harry" occurence is followed by "potter"
------------------------------------------------------------------------------------------------------------------------
'''
