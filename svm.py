"""
SVM model of Group 11 project
"""

###################
# IMPORT DATA
##################
import pandas as pd

# check if a string contains numbers
def containsNumbers(string):
    return any(char.isdigit() for char in string)

msgs = pd.read_csv('spam.csv', encoding = "ISO-8859-1")

# shuffle the data
msgs = msgs.sample(frac=1)

X = msgs.iloc[:, 1]
y = msgs.iloc[:, 0]

###################
# CREATING NUMERICAL ATTRIBUTE VECTORS FOR X
###################

spamKeywords = ['available', 'crazy', 'free', 'win', 'text', 'apply', 'now', 'winner', 'call', 'claim',
                'upgrade', 'chance', 'txt', 'send', 'credit', 'click', 'miss', 'reply', 'confirm', 'congrats',
                'valued', 'urgent', 'award', 'won', 'guarantee', 'expire', 'today', 'freemsg', 'stop', 'msg',
                'service', 'find out', 'must', 'congratulations', 'subscribe', 'customer', 'important', 'lucky',
                'sex', 'please', 'discount', 'last', 'message', 'award', 'bill', 'contact', 'dating', 'private',
                'account']


msgLength = []  # continuous attribute
allCaps = []    # binary attribute
hasNumbers = [] # binary attribute
singleKeyword = []  # binary attribute
multipleKeywords = []   # binary attribute

# read information of messages into attribute vectors
for msg in X:
    msgLength.append(len(msg))

    if msg.isupper():
        allCaps.append(1)
    else:
        allCaps.append(0)

    if containsNumbers(msg):
        hasNumbers.append(1)
    else:
        hasNumbers.append(0)

    keywordCount = 0
    for keyword in spamKeywords:
        if keyword in msg.lower():
            keywordCount += 1
        if keywordCount == 2:
            break

    if keywordCount == 0:
        singleKeyword.append(0)
        multipleKeywords.append(0)
    elif keywordCount == 1:
        singleKeyword.append(1)
        multipleKeywords.append(0)
    else:
        singleKeyword.append(1)
        multipleKeywords.append(1)

# Making X Dataframe out of these numerical vectors

attributeData = {
    'msgLength' : msgLength,
    'allCaps': allCaps,
    'hasNumbers': hasNumbers,
    'singleKeyword': singleKeyword,
    'multipleKeyword': multipleKeywords
}

df = pd.DataFrame(attributeData)

###################
# TRAIN/TEST SPLIT
##################

# total rows = 5571
# training - 80% of 5571 is 4457
# testing - other 20% = 1114

X_train = df[:4457]
y_train = y[:4457]
X_test = df[4457:]
y_test = y[4457:]

#############################
# BUILDING THE SVM WITH TRAINING SET
#############################

from sklearn import svm
from sklearn import metrics

# I built the Support Vector Machine using appropriate functions from sklearn
# libraries
vectorMachine = svm.SVC(kernel='linear')

#Train the model using the training sets
vectorMachine.fit(X_train, y_train)

def testSVM():
    # TESTING SVM ON TEST SET

    #Predict the response for test dataset
    y_pred = vectorMachine.predict(X_test)

    # Model Accuracy

    print("Total SVM model accuracy for message dataset: ", metrics.accuracy_score(y_test, y_pred))

# use inputted machine on new String data (SMS data)
def runMachine(sms):

    # Initialize attribute vectors to fit attributes of sms in

    msgLength = []  # continuous attribute
    allCaps = []    # binary attribute
    hasNumbers = [] # binary attribute
    singleKeyword = []  # binary attribute
    multipleKeywords = []   # binary attribute

    msgLength.append(len(sms))

    if sms.isupper():
        allCaps.append(1)
    else:
        allCaps.append(0)

    if containsNumbers(sms):
        hasNumbers.append(1)
    else:
        hasNumbers.append(0)

    keywordCount = 0
    for keyword in spamKeywords:
        if keyword in sms.lower():
            keywordCount += 1
        if keywordCount == 2:
            break

    if keywordCount == 0:
        singleKeyword.append(0)
        multipleKeywords.append(0)
    elif keywordCount == 1:
        singleKeyword.append(1)
        multipleKeywords.append(0)
    else:
        singleKeyword.append(1)
        multipleKeywords.append(1)

    # Making X Dataframe out of these numerical vectors

    attributeData = {
        'msgLength' : msgLength,
        'allCaps': allCaps,
        'hasNumbers': hasNumbers,
        'singleKeyword': singleKeyword,
        'multipleKeyword': multipleKeywords
    }

    df = pd.DataFrame(attributeData)

    X_test = df

    predictions = vectorMachine.predict(X_test)

    print(predictions)
