from sys import argv
from pickle import load
from collections import Counter
from nltk import word_tokenize
import re
import numpy as np
from nltk.util import everygrams


def loadObjectFromPickleFile(filePath):
    """Load an python object from a pickle file."""
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)


def readLinesFromFile(filePath):
    """Read lines from a file."""
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def findSentencesFromText(text, lang='te'):
    """Find no of sentences from a piece of text."""
    sentences = list()
    if lang == 'te':
        sentences = [item for item in re.findall('(.*?\. )', text)]
    elif lang == 'hi':
        sentences = [item for item in re.findall('(.*?। )', text)]
    if ''.join(sentences) == text:
        pass
    else:
        # print(text)
        if text.find(''.join(sentences)) == 0:
            if len(''.join(sentences)) == len(text):
                pass
            else:
                lastSent = text[len(''.join(sentences)):]
                sentences.append(lastSent)
    return sentences


def findAlignmentBetweenTextUsingTransDict(sentenceList1, sentenceList2, transDict):
    """Find alignment of sentences between aligned paragraphs using a bilingual dictionary."""
    backPointers = np.zeros(
        (len(sentenceList2), len(sentenceList1)), dtype='int8')
    alignmentMatrix = np.ones((len(sentenceList2), len(sentenceList1)))
    parallelSentences = list()
    if len(sentenceList1) > 1 and len(sentenceList2) > 1:
        scores = findScoreForAlignmentUsingnGrams(
            sentenceList1[0], sentenceList2, transDict)
        # scores = findScoreForAlignment(
        #     sentenceList1[0], sentenceList2, transDict)
        for index, score in enumerate(scores):
            alignmentMatrix[index][0] = score[0] * score[1]
            backPointers[index][0] = index
        for col in range(1, len(sentenceList1)):
            scores = findScoreForAlignmentUsingnGrams(
                sentenceList1[col], sentenceList2, transDict)
            # scores = findScoreForAlignment(
            #     sentenceList1[col], sentenceList2, transDict)
            for j in range(len(sentenceList2)):
                scoresFromPrev = scores[j][0] * \
                    scores[j][1] + alignmentMatrix[:, col - 1]
                alignmentMatrix[j][col] = np.max(scoresFromPrev)
                maxIndex = np.argmax(scoresFromPrev)
                backPointers[j][col] = maxIndex
        maxIndex = np.argmax(alignmentMatrix[:, col])
        parallelSentences.append(sentenceList1[col].strip(
        ) + '\t' + sentenceList2[maxIndex].strip() + '\n')
        for col in range(-1, - len(sentenceList1), -1):
            maxIndex = backPointers[maxIndex][col]
            parallelSentences.append(
                sentenceList1[col - 1].strip() + '\t' + sentenceList2[maxIndex].strip() + '\n')
    else:
        try:
            parallelSentences.append(sentenceList1[0].strip(
            ) + '\t' + sentenceList2[0].strip() + '\n')
        except IndexError:
            print("it contains zero sentences")
            return
    return parallelSentences


def create_string_ngrams(ngrams_all):
    """Create string ngrams where each ngram is a tuple."""
    return list(map(lambda x: ' '.join(x), ngrams_all))


def findScoreForAlignment(srcSent, tgtList, transDict):
    """Find alignment score for a source sentence with a list of target sentences."""
    wordsInSourceSent = word_tokenize(srcSent.lower())
    wordsInSrc = len(wordsInSourceSent)
    srcDict = Counter(wordsInSourceSent)
    scores = list()
    tgtDicts = [Counter(word_tokenize(tgt.lower())) for tgt in tgtList]
    print(len(transDict))
    for tgtDict in tgtDicts:
        count = 0
        matchedWords = list()
        wordInTgt = sum(tgtDict.values())
        for src in srcDict:
            if re.search('\d+(\.\d+)?', src):
                if src in tgtDict:
                    matchedWords.append((src, src))
                    count += 1
            elif src in transDict:
                foundTgt = transDict[src]
                for word in foundTgt:
                    if word in tgtDict and tgtDict[word] == srcDict[src]:
                        matchedWords.append((src, word))
                        count += tgtDict[word]
                        break
                    elif word in tgtDict and tgtDict[word] != srcDict[src]:
                        break
        if abs(wordsInSrc - wordInTgt) in range(5):
            lengthValue = 0.2
        else:
            lengthValue = 1 / abs(wordsInSrc - wordInTgt)
        print(count)
        print(matchedWords)
        if count == 0.:
            scores.append((1e-5, lengthValue))
        else:
            scores.append((count / len(wordsInSourceSent), lengthValue))
    print(scores)
    return np.array(scores)


def findScoreForAlignmentUsingnGrams(srcSent, tgtList, transDict):
    """Find alignment score for a source sentence with a list of target sentences."""
    wordsInSourceSent = word_tokenize(srcSent.lower())
    wordsInSrc = len(wordsInSourceSent)
    sourceNgrams = list(everygrams(wordsInSourceSent, max_len=2))
    sourceNgrams = create_string_ngrams(sourceNgrams)
    srcDict = Counter(sourceNgrams)
    scores = list()
    tgtDicts, allTgtWords = list(), list()
    for tgt in tgtList:
        wordsInTgt = word_tokenize(tgt.lower())
        allTgtWords.append(wordsInTgt)
        tgtNgrams = list(everygrams(wordsInTgt, max_len=2))
        tgtNgrams = create_string_ngrams(tgtNgrams)
        tgtDicts.append(Counter(tgtNgrams))
    print(len(transDict))
    for index, tgtDict in enumerate(tgtDicts):
        count = 0
        matchedNgrams = list()
        for src in srcDict:
            if re.search('\d+(\.\d+)?', src):
                if src in tgtDict:
                    matchedNgrams.append((src, src))
                    count += 1
            elif src in transDict:
                foundTgt = transDict[src]
                for ngrm in foundTgt:
                    if ngrm in tgtDict and tgtDict[ngrm] == srcDict[src]:
                        matchedNgrams.append((src, ngrm))
                        count += tgtDict[ngrm]
                        break
                    elif ngrm in tgtDict and tgtDict[ngrm] != srcDict[src]:
                        break
        wordsInTgt = len(allTgtWords[index])
        if abs(wordsInSrc - wordsInTgt) in range(5):
            lengthValue = 0.2
        else:
            lengthValue = 1 / abs(wordsInSrc - wordsInTgt)
        print(count)
        print(matchedNgrams)
        if count == 0.:
            scores.append((1e-5, lengthValue))
        else:
            scores.append((count / len(sourceNgrams), lengthValue))
    print(scores)
    return np.array(scores)


def writeListToFile(filePath, dataList):
    """Write list to a file."""
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write(''.join(dataList) + '\n')


def main():
    englishFile = argv[1]
    hindiFile = argv[2]
    transPickle = argv[3]
    outputFile = argv[4]
    parallelSentences = list()
    englishPars = readLinesFromFile(englishFile)
    print(len(englishPars))
    hindiPars = readLinesFromFile(hindiFile)
    print(len(hindiPars))
    assert len(hindiPars) == len(englishPars)
    transDict = loadObjectFromPickleFile(transPickle)
    for index, hindiPar in enumerate(hindiPars):
        print(index)
        englishPar = englishPars[index]
        hindiSentences = findSentencesFromText(hindiPar, 'hi')
        englishSentences = findSentencesFromText(englishPar, 'te')
        got_sent = findAlignmentBetweenTextUsingTransDict(
            hindiSentences, englishSentences, transDict)
        if got_sent:
            parallelSentences += got_sent
    parallelSentences = parallelSentences[::-1]
    writeListToFile(outputFile, parallelSentences)


if __name__ == '__main__':
    main()
