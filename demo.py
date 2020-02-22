from naiveBayes import naiveBayes
from knn import knn_string
from svm import runMachine

test_messages = ["hey dude what's up", "HEY BRO, WHATSUP! REPLY NOW AT 83147661764. WIN WIN WIN", "i Like pie", "TEXT ME to win a FREE car free 4 lyfe yo 374237432"]

# Demo Naive Bayes
print("Naive Bayes")
print("------------")
for msg in test_messages:
    print(naiveBayes(msg))

print("K-Nearest Neighbors")
print("--------------------")
for msg in test_messages:
    knn_string(msg)

print("Support Vector Machine")
print("----------------------")
for msg in test_messages:
    runMachine(msg)
