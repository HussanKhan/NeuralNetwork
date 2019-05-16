import numpy

class RNNDataHandle():

    """ This class prepares sequence text data for a RNN, by one-hot-encoding the
        data into inputs and targets"""

    def __init__(self, textData):
        self.textData = textData
        
        # Create vocab list by going through text data
        self.allVocab = []
        self.vocabMap = {}
        self.vocabLen = 0
        self.totalInputs = 0

        self.inputs = []
        self.targets = []

    # Builds vocab map
    def buildVocab(self):
        # Every word -> next word combo
        totalInputs = 0
        # Id for word mapping
        wordId = 0
        for prompt in self.textData:
            words = prompt.split()

            for word in words:

                totalInputs += 1

                if word not in self.allVocab:
                    self.allVocab.append(word)
                    self.vocabMap[word] = wordId
                    wordId += 1

        self.totalInputs = totalInputs
        self.vocabLen = wordId
        print(self.vocabMap)
        return self.vocabLen

    def buildInputTarget(self):
        # Aligned by index
        allInputs = numpy.zeros((self.totalInputs, self.vocabLen))
        allTargets = numpy.zeros((self.totalInputs, self.vocabLen))

        currentInput = 0

        # Over all prompts
        for sequence in self.textData:

            wordList = sequence.split()
            # maxLen = len(wordList) - 1

            for word in wordList:
                wordIndex = self.vocabMap[word]
                allInputs[currentInput, wordIndex] = 1
                currentInput += 1

        # Fills targets with next word value
        targetIndex = 0
        for i in range(1, self.totalInputs):
            allTargets[targetIndex] = allInputs[i]
            targetIndex += 1

        self.inputs = allInputs
        self.targets = allTargets

        return (allInputs, allTargets)

    def getString(self, wordVector):
        wordCode = numpy.argmax(wordVector)
        return self.allVocab[wordCode]

# def buildVocab(textData):
#     # Every word -> next word combo
#     totalInputs = 0
#     # Id for word mapping
#     wordId = 0
#     for prompt in textData:
#         words = prompt.split()
    
#         for word in words:

#             totalInputs += 1
            
#             if word not in allVocab:
#                 allVocab.append(word)
#                 vocabMap[word] = wordId
#                 wordId += 1
    
#     return totalInputs

# totalInputs = buildVocab(textData) # Every combo of input to next word

# # One hot encode input
# def oneHotSeq(textData, totalInputs, vocabLen):
#     # Aligned by index
#     allInputs = numpy.zeros((totalInputs, vocabLen))
#     allTargets = numpy.zeros((totalInputs, vocabLen))

#     currentInput = 0

#     # Over all prompts
#     for sequence in textData:
        
#         wordList = sequence.split()
#         # maxLen = len(wordList) - 1

#         for word in wordList:
#             wordIndex = vocabMap[word]
#             allInputs[currentInput, wordIndex] = 1
#             currentInput += 1

#     # Fills targets with next word value
#     targetIndex = 0
#     for i in range(1, totalInputs):
#         allTargets[targetIndex] = allInputs[i]
#         targetIndex += 1

#     return (allInputs, allTargets)
    
# inputs, targets = oneHotSeq(textData, totalInputs, len(allVocab))

# print(inputs)
# print(targets)