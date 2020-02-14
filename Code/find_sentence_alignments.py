from sys import argv
from pickle import load
from collections import Counter
from nltk import word_tokenize
import re
import numpy as np


def loadObjectFromPickleFile(filePath):
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def findSentencesFromText(text, lang='en'):
    if not re.search('.+\n$', text):
        text += '\n'
    if lang == 'en':
        return [item[0] if not item[1].strip() else item[1] for item in re.findall('(.*?\. )|(.*?\.\n)', text)]
    elif lang == 'hi':
        return [item[0] if not item[1].strip() else item[1] for item in re.findall('(.*?। )|(.*?।\n)', text)]


def findAlignmentBetweenTextUsingTransDict(sentenceList1, sentenceList2, transDict):
    backPointers = np.zeros((len(sentenceList2), len(sentenceList1)), dtype='int8')
    alignmentMatrix = np.ones((len(sentenceList2), len(sentenceList1)))
    scores = findScoreForAlignment(sentenceList1[0], sentenceList2, transDict)
    parallelSentences = list()
    for index, score in enumerate(scores):
        alignmentMatrix[index][0] = score[0] * score[1]
        backPointers[index][0] = index
    for col in range(1, len(sentenceList1)):
        scores = findScoreForAlignment(sentenceList1[col], sentenceList2, transDict)
        for j in range(len(sentenceList2)):
            scoresFromPrev = scores[j][0] * scores[j][1] + alignmentMatrix[:, col - 1]
            alignmentMatrix[j][col] = np.max(scoresFromPrev)
            maxIndex = np.argmax(scoresFromPrev)
            backPointers[j][col] = maxIndex
    maxIndex = np.argmax(alignmentMatrix[:, col])
    print('Pointer-->', maxIndex)
    print(sentenceList1[col], '\n--aligned to--\n', sentenceList2[maxIndex])
    parallelSentences.append(sentenceList1[col].strip() + '\t' + sentenceList2[maxIndex].strip() + '\n')
    for col in range(-1, - len(sentenceList1), -1):
        print('Pointer-->', backPointers[maxIndex][col])
        maxIndex = backPointers[maxIndex][col]
        print(sentenceList1[col - 1], '\n--aligned to--\n', sentenceList2[maxIndex])
        parallelSentences.append(sentenceList1[col - 1].strip() + '\t' + sentenceList2[maxIndex].strip() + '\n')
    return parallelSentences


def findScoreForAlignment(srcSent, tgtList, transDict):
    wordsInSourceSent = word_tokenize(srcSent)
    wordsInSrc = len(wordsInSourceSent)
    srcDict = Counter(wordsInSourceSent)
    scores = list()
    tgtDicts = [Counter(word_tokenize(tgt)) for tgt in tgtList]
    for tgtDict in tgtDicts:
        count = 0
        wordInTgt = sum(tgtDict.values())
        for src in srcDict:
            if re.search('\d+(\.\d+)?', src):
                if src in tgtDict:
                    count += 1
            elif src in transDict:
                foundTgt = transDict[src]
                for word in foundTgt:
                    if word in tgtDict and tgtDict[word] == srcDict[src]:
                        count += tgtDict[word]
                        break
                    elif word in tgtDict and tgtDict[word] != srcDict[src]:
                        break
        lengthValue = 1 if wordInTgt == wordsInSrc else 1 / abs(wordsInSrc - wordInTgt)
        scores.append((count / len(wordsInSourceSent), lengthValue))
    return np.array(scores)


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write(''.join(dataList) + '\n\n')


def main():
    englishFile = argv[1]
    hindiFile = argv[2]
    transPickle = argv[3]
    outputFile = argv[4]
    parallelSentences = list()
    englishPars = readLinesFromFile(englishFile)
    hindiPars = readLinesFromFile(hindiFile)
    assert len(hindiPars) == len(englishPars)
    transDict = loadObjectFromPickleFile(transPickle)
    for index, hindiPar in enumerate(hindiPars):
        englishPar = englishPars[index]
        hindiSentences = findSentencesFromText(hindiPar, 'hi')
        englishSentences = findSentencesFromText(englishPar, 'en')
        parallelSentences += findAlignmentBetweenTextUsingTransDict(hindiSentences, englishSentences, transDict)
    writeListToFile(outputFile, parallelSentences)


if __name__ == '__main__':
    main()
