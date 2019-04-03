import textblob

import argparse

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

# Get Sentences from Text File
# Input file name, output list of sentences
def GetFileSentences(file_name):
    file = open(file_name,"r")
    text = file.read()
    text = text.decode("utf-8").encode("ascii", "ignore")
    blobby = textblob.TextBlob(text)
    all_sentences = blobby.raw_sentences
    return all_sentences

# Get tags from sentence
# Input Sentence, Output Tags list
def GetTagsFromSentence(sentence):
    sentence = sentence.lower()
    words = textblob.TextBlob(sentence)
    tags = words.tags
    return tags

def findNNPForPRP(PRPIndex,SentenceIndex,TaggedAllSentences):
    NNPList = list()
    tags = TaggedAllSentences[SentenceIndex]
    if PRPIndex == -1:
        return findNNPForPRP(len(TaggedAllSentences[SentenceIndex-1]),SentenceIndex-1,TaggedAllSentences)
    else:
        # print "PRP : ", PRPIndex,"\nTagged:\n",tags
        if PRPIndex > 0:
            for NNPFinder in range(0,PRPIndex):
                if "NNP" in tags[NNPFinder]:
                    # print "NNPFinder",NNPFinder,"\ntags_len:",len(tags)
                    if NNPFinder < len(tags)-1:
                        if ("NNP" in tags[NNPFinder+1] or "CC" in tags[NNPFinder+1]):
                            NNPList.append(tags[NNPFinder])
                            # print "Append",tags[NNPFinder],"\nLIST NOW\n",tempNNPList
                        else:
                            NNPList.append(tags[NNPFinder])
                            # print "Append",tags[NNPFinder],"\nLIST NOW\n",tempNNPList
                            # print "Returning list\n",NNPList
                            return NNPList
        else:
            return findNNPForPRP(-1,SentenceIndex,TaggedAllSentences)
    return NNPList

def replacePRPWithNNP(PRPIndexInSentence,tagged_sentence,NNPList):
    # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    # print "\n\nNNPList : ",NNPList
    # print "\n\nPRP : ", tagged_sentence[PRPIndexInSentence],"\nPRP Index :", PRPIndexInSentence
    # print "\n\nSentence : ", tagged_sentence
    # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    NNPList.reverse()
    tagged_sentence.pop(PRPIndexInSentence)
    for NNP in NNPList:
        tagged_sentence.insert(PRPIndexInSentence,NNP)
    return tagged_sentence

def getSentenceFromTags(tags):
    OP = ""
    for word in tags:
        if not word[0][0] == '\'':
            OP = OP + " " 
        OP = OP + word[0]
    OP = OP + ". "
    return OP

def remakeSentences(tagged_sentences):
    new_sentences = list()
    OP = ""
    for sentence in tagged_sentences:
        OP = getSentenceFromTags(sentence)
        new_sentences.append(OP)
    return new_sentences

def summarizeText(text):
    content = text
    content = sanitize_input(content)

    sentence_tokens, word_tokens = tokenize_content(content)  
    sentence_ranks = score_tokens(word_tokens, sentence_tokens)

    return summarize(sentence_ranks, sentence_tokens, 6)


# Summarization Functions

def parse_arguments():
    """ Parse command line arguments """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='File name of text to summarize')
    parser.add_argument('-l', '--length', default=4, help='Number of sentences to return')
    args = parser.parse_args()

    return args

def read_file(path):
    """ Read the file at designated path and throw exception if unable to do so """ 
    try:
        with open(path, 'r') as file:
            return file.read()

    except IOError as e:
        print("Fatal Error: File ({}) could not be locaeted or is not readable.".format(path))

def sanitize_input(data):
    """ 
    Currently just a whitespace remover. More thought will have to be given with how 
    to handle sanitzation and encoding in a way that most text files can be successfully
    parsed
    """
    replace = {
        ord('\f') : ' ',
        ord('\t') : ' ',
        ord('\n') : ' ',
        ord('\r') : None
    }

    return data.translate(replace)

def tokenize_content(content):
    """
    Accept the content and produce a list of tokenized sentences, 
    a list of tokenized words, and then a list of the tokenized words
    with stop words built from NLTK corpus and Python string class filtred out. 
    """
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(content.lower())
    
    return [
        sent_tokenize(content),
        [word for word in words if word not in stop_words]    
    ]

def score_tokens(filterd_words, sentence_tokens):
    """
    Builds a frequency map based on the filtered list of words and 
    uses this to produce a map of each sentence and its total score
    """
    word_freq = FreqDist(filterd_words)

    ranking = defaultdict(int)

    for i, sentence in enumerate(sentence_tokens):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                ranking[i] += word_freq[word]

    return ranking

def summarize(ranks, sentences, length):
    """
    Utilizes a ranking map produced by score_token to extract
    the highest ranking sentences in order after converting from
    array to string.  
    """
    if int(length) > len(sentences): 
        print("Error, more sentences requested than available. Use --l (--length) flag to adjust.")
        exit()

    indexes = nlargest(length, ranks, key=ranks.get)
    final_sentences = [sentences[j] for j in sorted(indexes)]
    return ' '.join(final_sentences)
#

def main():
    all_sentences = GetFileSentences("input.txt")
    tagged_sentences = list()
    for sentence in all_sentences:
        tagged_sentences.append((GetTagsFromSentence(sentence)))
    for SentenceIndex,tagged_sentence in enumerate(tagged_sentences):
        # print "\n\n\n NEW SENTENCE",getSentenceFromTags(tagged_sentence)
        for PRPIndexInSentence,tagged_word in enumerate(tagged_sentence):
            if "PRP" in tagged_word:
                if not ("thank" in tagged_sentence[PRPIndexInSentence-1] or "Thank" in tagged_sentence[PRPIndexInSentence-1]):
                    NNPList = findNNPForPRP(PRPIndexInSentence,SentenceIndex,tagged_sentences)
                    tagged_sentence = replacePRPWithNNP(PRPIndexInSentence,tagged_sentence,NNPList)
                    tagged_sentences[SentenceIndex] = tagged_sentence
    all_sentences = remakeSentences(tagged_sentences)
    character = raw_input("Enter Character Name : (one word, First name or Last name) : ")
    character = character.lower()
    relevant = ""
    for sentence in all_sentences:
        if character in sentence:
            relevant = relevant + ". " + sentence
    print summarizeText(relevant)










if __name__ == '__main__':
    main()
        