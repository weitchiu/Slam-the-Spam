import pandas

msgs = pandas.read_csv('spam.csv', encoding = "ISO-8859-1")

###################
# IMPORT DATA
##################

X = msgs.iloc[:, 1]
y = msgs.iloc[:, 0]

###################
# TRAIN/TEST SPLIT
##################

# total rows = 5571
# training - 80% of 5571 is 4457
# testing - other 20% = 1114

X_train = X[:4457]
y_train = y[:4457]
X_test = X[4457:]
y_test = y[4457:]

spamKeywords = ['available', 'crazy', 'free', 'win', 'text', 'apply', 'now', 'winner', 'call', 'claim',
                'upgrade', 'chance', 'txt', 'send', 'credit', 'click', 'miss', 'reply', 'confirm', 'congrats',
                'valued', 'urgent', 'award', 'won', 'guarantee', 'expire', 'today', 'freemsg', 'stop', 'msg',
                'service', 'find out', 'must', 'congratulations', 'subscribe', 'customer', 'important', 'lucky',
                'sex', 'please', 'discount', 'last', 'message', 'award', 'bill', 'contact', 'dating', 'private',
                'account', 'code', 'mobile', 'sale', 'rate', 'money', 'only', 'for you', 'we', 'invite', 'cash',
                'chat']

words = ['free', 'text', 'txt', 'miss', 'mobile', 'reply', 'please', 'call', 'now', 'new', 'service', 'just',
         'customer', 'only', 'win', 'msg', 'message', 'home', 'love', 'ok', 'tell', 'want', 'will', 'shall', 'were',
         'know', 'still', 'got', 'need', 'come', 'go', 'urgent', 'today', 'congrats', 'congratulations', 'confirm',
         'we', 'think', 'good', 'happy', 'lor', 'thought', 'yup', 'life', 'cool', 'also', 'shall', 'where', 'very', "won't",
         'what', 'one', 'problem', 'probably', 'anyway', 'asap', 'again', 'lol', 'really', 'always', 'ass', 'no',
         'have', 'if', 'why', "don't", 'anything', 'this', 'something', 'then', 'nope', 'yup', 'yeah', 'ye', 'are',
         "can't", "haven't", 'cant', 'my', 'me', "i'm", 'am', 'maybe', 'babe', ':)', 'sure', 'cuz', 'gonna', 'how',
         'when', 'so', 'thanks', 'can', 'finish', 'yes', 'dot', 'at', 'hello', 'sometimes', 'already', 'but', 'sorry',
         'finally', 'doing', 'did', 'day', 'its', 'prob', 'there', 'too', 'said', 'stuff', 'oh', 'shit', 'meet',' might',
         'been', 'took', ', time', 'that', 'like', 'cos', 'bring', 'any', 'ask', 'hi', 'gotta', 'class', 'right', 'dear',
         'sex', 'hot', 'friend', 'wait', 'won', 'person', 'leave', 'here', 'maybe', 'was', 'were', 'wish', 'hope', 'you',
         'us', 'upgrade', 'account', 'money', 'rate', 'cash', 'stop', 'subscribe', 'private', 'bill', 'chat', 'award', 'claim']

traits = [[] for _ in range(4)] # allCaps, hasNumbers, singleKeyword, multipleKeywords
containWords = [[] for _ in range(len(words))]

# check if a string contains numbers
def containsNumbers(string):
    return any(char.isdigit() for char in string)

# read information of messages into attribute vectors
for msg in X_train:
    if msg.isupper():
        traits[0].append(1)
    else:
        traits[0].append(0)

    if containsNumbers(msg):
        traits[1].append(1)
    else:
        traits[1].append(0)

    keywordCount = 0
    for keyword in spamKeywords:
        if keyword in msg.lower():
            keywordCount += 1
        if keywordCount == 2:
            break

    if keywordCount == 0:
        traits[2].append(0)
        traits[3].append(0)
    elif keywordCount == 1:
        traits[2].append(1)
        traits[3].append(0)
    else:
        traits[2].append(1)
        traits[3].append(1)

    for i in range(len(words)):
        if words[i] in msg.lower():
            containWords[i].append(1)
        else:
            containWords[i].append(0)

spam = 0
notSpam = 0

traitAndSpam = [0 for _ in range(4)]    # allCaps, hasNumbers, singleKeyword, multipleKeywords
traitAndNotSpam = [0 for _ in range(4)]
noTraitAndSpam = [0 for _ in range(4)]
noTraitAndNotSpam = [0 for _ in range(4)]

wordAndSpam = [0 for _ in range(len(words))]
wordAndNotSpam = [0 for _ in range(len(words))]
noWordAndSpam = [0 for _ in range(len(words))]
noWordAndNotSpam = [0 for _ in range(len(words))]

# obtain information for calculating probabilities
for i in range(len(y_train)):
    result = y_train[i]
    if result == 0:
        notSpam += 1
    else:
        spam += 1

    for j in range(4):
        trait = traits[j][i]

        if not trait and not result:
            noTraitAndNotSpam[j] += 1
        elif not trait and result:
            noTraitAndSpam[j] += 1
        elif trait and not result:
            traitAndNotSpam[j] += 1
        else:
            traitAndSpam[j] += 1

    for k in range(len(words)):
        containWord = containWords[k][i]

        if not containWord and not result:
            noWordAndNotSpam[k] += 1
        elif not containWord and result:
            noWordAndSpam[k] += 1
        elif containWord and not result:
            wordAndNotSpam[k] += 1
        else:
            wordAndSpam[k] += 1

total = len(y_train)
pSpam = spam / total
pNotSpam = notSpam / total

pTraitGivenSpam = [0 for _ in range(4)]
pTraitGivenNotSpam = [0 for _ in range(4)]
pNoTraitGivenSpam = [0 for _ in range(4)]
pNoTraitGivenNotSpam = [0 for _ in range(4)]

pWordGivenSpam = [0 for _ in range(len(words))]
pWordGivenNotSpam = [0 for _ in range(len(words))]
pNoWordGivenSpam = [0 for _ in range(len(words))]
pNoWordGivenNotSpam = [0 for _ in range(len(words))]

for i in range(4):
    pTraitGivenSpam[i] = (traitAndSpam[i] / total) / pSpam
    pTraitGivenNotSpam[i] = (traitAndNotSpam[i] / total) / pNotSpam
    pNoTraitGivenSpam[i] = (noTraitAndSpam[i] / total) / pSpam
    pNoTraitGivenNotSpam[i] = (noTraitAndNotSpam[i] / total) / pNotSpam

for i in range(len(words)):
    pWordGivenSpam[i] = (wordAndSpam[i] / total) / pSpam
    pWordGivenNotSpam[i] = (wordAndNotSpam[i] / total) / pNotSpam
    pNoWordGivenSpam[i] = (noWordAndSpam[i] / total) / pSpam
    pNoWordGivenNotSpam[i] = (noWordAndNotSpam[i] / total) / pNotSpam

def naiveBayes(string):
    pTraitGivenSpam = [0 for _ in range(4)]
    pTraitGivenNotSpam = [0 for _ in range(4)]
    pNoTraitGivenSpam = [0 for _ in range(4)]
    pNoTraitGivenNotSpam = [0 for _ in range(4)]

    pWordGivenSpam = [0 for _ in range(len(words))]
    pWordGivenNotSpam = [0 for _ in range(len(words))]
    pNoWordGivenSpam = [0 for _ in range(len(words))]
    pNoWordGivenNotSpam = [0 for _ in range(len(words))]

    for i in range(4):
        pTraitGivenSpam[i] = (traitAndSpam[i] / total) / pSpam
        pTraitGivenNotSpam[i] = (traitAndNotSpam[i] / total) / pNotSpam
        pNoTraitGivenSpam[i] = (noTraitAndSpam[i] / total) / pSpam
        pNoTraitGivenNotSpam[i] = (noTraitAndNotSpam[i] / total) / pNotSpam

    for i in range(len(words)):
        pWordGivenSpam[i] = (wordAndSpam[i] / total) / pSpam
        pWordGivenNotSpam[i] = (wordAndNotSpam[i] / total) / pNotSpam
        pNoWordGivenSpam[i] = (noWordAndSpam[i] / total) / pSpam
        pNoWordGivenNotSpam[i] = (noWordAndNotSpam[i] / total) / pNotSpam

    msgTraits = [0 for _ in range(4)]
    msgTraits[0] = string.isupper()
    msgTraits[1] = containsNumbers(string)

    numKeywords = 0
    for keyword in spamKeywords:
        if keyword in string.lower():
            numKeywords += 1
        if numKeywords == 2:
            break

    if numKeywords == 1:
        msgTraits[2] = 1
    if numKeywords == 2:
        msgTraits[3] = 1

    msgWords = [0 for _ in range(len(words))]

    for i in range(len(words)):
        if words[i] in string.lower():
            msgWords[i] = 1

    pMsgTraitsGivenSpam = [0 for _ in range(4)]
    pMsgTraitsGivenNotSpam = [0 for _ in range(4)]

    pMsgWordsGivenSpam = [0 for _ in range(len(words))]
    pMsgWordsGivenNotSpam = [0 for _ in range(len(words))]

    for i in range(4):
        if msgTraits[i]:
            pMsgTraitsGivenSpam[i] = pTraitGivenSpam[i]
            pMsgTraitsGivenNotSpam[i] = pTraitGivenNotSpam[i]
        else:
            pMsgTraitsGivenSpam[i] = pNoTraitGivenSpam[i]
            pMsgTraitsGivenNotSpam[i] = pNoTraitGivenNotSpam[i]

    for i in range(len(words)):
        if msgWords[i]:
            pMsgWordsGivenSpam[i] = pWordGivenSpam[i]
            pMsgWordsGivenNotSpam[i] = pWordGivenNotSpam[i]
        else:
            pMsgWordsGivenSpam[i] = pNoWordGivenSpam[i]
            pMsgWordsGivenNotSpam[i] = pNoWordGivenNotSpam[i]

    pSpamGivenTraits = pSpam
    pNotSpamGivenTraits = pNotSpam

    for pTraitGivenSpam in pMsgTraitsGivenSpam:
        pSpamGivenTraits *= pTraitGivenSpam
    for pTraitGivenNotSpam in pMsgTraitsGivenNotSpam:
        pNotSpamGivenTraits *= pTraitGivenNotSpam

    for pWordGivenSpam in pMsgWordsGivenSpam:
        pSpamGivenTraits *= pWordGivenSpam
    for pWordGivenNotSpam in pMsgWordsGivenNotSpam:
        pNotSpamGivenTraits *= pWordGivenNotSpam

    if pSpamGivenTraits > pNotSpamGivenTraits:
        return 1
    else:
        return 0

def testNaiveBayes():
    yHat = []
    error = 0
    for i in range(4457, 4457 + len(X_test)):
        result = naiveBayes(X_test[i])
        yHat.append(result)
        if result != y_test[i]:
            error += 1
    print(1 - (error / len(y_test)))

testNaiveBayes()

