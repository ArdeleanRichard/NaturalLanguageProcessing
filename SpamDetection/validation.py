from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

from preprocess import preprocess_text


def is_spam(vectorizer, logreg, text, clean=True):
    if clean==True:
        cleaned_text = preprocess_text(text)
    else:
        cleaned_text = text

    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = logreg.predict(text_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"


def get_performance(true, pred):
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)

    # Performance
    print(f'\n\nPerformance:')
    print(f"-> Accuracy: {accuracy:.4f}")
    print(f"-> Precision: {precision:.4f}")
    print(f"-> Recall: {recall:.4f}")
    print(f"-> F1-Score: {f1:.4f}")

def plot_confusion_matrix(true, pred):
    # After making predictions
    conf_matrix = confusion_matrix(true, pred)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()