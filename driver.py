from knn import knn_test
from naiveBayes import testNaiveBayes
from svm import testSVM

if __name__ == "__main__":
    # Naive Bayes Test
    print("Running Naive Bayes.")
    print("Accuracy: ")
    testNaiveBayes()

    # KNN Test
    print("Running KNN. This will take some time.")
    print("Accuracy: ")
    knn_test()

    # SVM Test
    print("Running SVM.")
    print("Accuracy: ")
    testSVM()
