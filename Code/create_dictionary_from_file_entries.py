from pickle import dump
from sys import argv


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def createDictionaryFromEntries(lines):
    translationDict = dict()
    for line in lines:
        hi, en = line.split()
        translationDict.setdefault(hi, list()).append(en)
    return translationDict


def dumpObjectIntoFile(dataObject, filePath):
    with open(filePath, 'wb') as fileWrite:
        dump(dataObject, fileWrite)


def main():
    inputFile = argv[1]
    pickleFile = argv[2]
    inputLines = readLinesFromFile(inputFile)
    translationDict = createDictionaryFromEntries(inputLines)
    dumpObjectIntoFile(translationDict, pickleFile)
    print(translationDict['की'])


if __name__ == '__main__':
    main()
