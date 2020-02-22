import pandas

msgs = pandas.read_csv('spam.csv', encoding = "ISO-8859-1")

###################
# IMPORT DATA
##################

x = msgs.iloc[:, 1]
y = msgs.iloc[:, 0]

###################
# TRAIN/TEST SPLIT
##################

# total rows = 5571
# training - 80% of 5571 is 4457
# testing - other 20% = 1114

x_train = x[:4457]
y_train = y[:4457]
x_test = x[4457:]
y_test = y[4457:]

k_value = 67  # Could change this (just used sqrt(training))

# We'll calculate similarity by checking % of words they have in common
def message_similarity(train_msg, test_msg):
    common = 0
    words = set()

    for word in train_msg.split():
        words.add(word)

    for word in test_msg.split():
        if word in words: # We have a match!
            common += 1 

    return float( common / (len(train_msg.split()) + len(test_msg.split())) )

def knn(test_msg):
    # Map msg index to a similarity value
    similarity_dict = {}

    for i in range(0, len(x_train)):
        similarity = message_similarity(x_train[i], test_msg)
        similarity_dict[i] = similarity

    nearest_neighbors = []

    # Sort similarities from greatest to least
    sorted_dict = sorted(similarity_dict.items(), key = lambda entry: entry[1], reverse = True)

    # Get the k-nearest neighbors
    for i in range(k_value):
        entry = sorted_dict[i]
        nearest_neighbors.append(entry[0])

    # At the end, we have a list of indexes for most similar messages
    return nearest_neighbors

def classification(knn_array):
    non_spam = 0
    spam = 0

    for index in knn_array:
        if y_train[index] == 0:
            non_spam += 1
        else:
            spam += 1

    if non_spam > spam:
        return 0
    else:
        return 1

# Run KNN on single test input
def knn_string(string):
    knn_array = knn(string)
    result = classification(knn_array)

    print(result)

# Test KNN on test data
def knn_test():
    accurate = 0

    for i in range(4457, 4457 + len(x_test)):
        knn_array = knn(x_test[i])
        result = classification(knn_array)

        if result == y_test[i]:
            accurate += 1

    print(accurate / len(y_test))
