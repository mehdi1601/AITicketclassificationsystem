#### Student Name: Mehdi Dahlouk  ###
### Student ID:001440096 #####
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# Download necessary NLTK resources for text processing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words list for text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Define a function for preprocessing text data
def classify_ticket_type(text):
    text = text.lower()
    # Added Lists of keywords for each category to enhance text recongnition
    billing_keywords = [
        "bill", "charge", "payment", "invoice", "fees", "charges", "billing",
        "account charge", "subscription fee", "statement", "billing issue", "overcharge",
        "billing problem", "financial", "cost", "price", "expense", "pay", "due", "overbilling",
        "auto-pay", "autopay", "billing cycle", "monthly charge", "balance", "billing query",
        "billing dispute", "chargeback", "credit", "debit", "financial query", "payment issue",
        "payment problem", "pay invoice", "pay bill", "direct debit", "financial assistance",
        "payment method", "payment option", "payment plan", "pricing", "rate", "renewal",
        "subscription", "tariff", "transaction", "billing adjustment", "billing error",
        "charge dispute", "credit card", "payment failure", "payment error",
        "account balance", "payment confirmation", "late fee", "service charge",
        "payment processing", "payment terms", "payment history", "billing contact",
        "automated billing", "recurring charge", "billing support", "payment receipt",
        "payment deadline", "outstanding balance", "billing cycle end", "pro-rated charge",
        "billing statement", "annual fee", "quarterly billing", "monthly billing",
        "billing account", "payment dispute", "billing clarification", "e-billing",
        "online payment", "secure payment", "payment gateway", "billing notification",
        "payment reminder", "past due", "arrears", "financial statement", "account statement",
        "payment status", "unauthorized charge", "billing adjustment request", "payment extension",
        "payment deferment", "billing cycle date", "billing frequency", "invoice number",
        "invoice date", "invoice amount", "payment breakdown", "itemized bill", "billing inquiry"
    ]
    cancellation_keywords = [
        "cancel", "terminate", "unsubscribe", "stop", "end", "revoke", "abort", "discontinue",
        "cancel order", "cancel subscription", "cancel service", "cancel booking", "cancel reservation",
        "cancel membership", "cancel account", "cancel application", "cancel registration",
        "halt", "suspend", "withdraw", "rescind", "annul", "cease", "nullify", "void",
        "cancelation request", "termination request", "discontinuation", "opt out", "drop out",
        "break off", "call off", "cut off", "delete", "erase", "remove", "undo", "cancel policy",
        "cancellation policy", "cancellation fee", "cancellation charges", "no cancel",
        "cannot cancel", "unable to cancel", "cancel immediately", "urgent cancellation",
        "expedite cancellation", "cancel confirmation", "confirm cancellation", "cancellation confirmed",
        "cancel without penalty", "cancel without charge", "cancel with refund", "early cancellation",
        "late cancellation", "cancel due to", "reason for cancellation", "cancellation conditions"
    ]
    product_keywords = [
        "product", "item", "purchase", "order", "buy", "goods", "merchandise", "product info",
        "product details", "product query", "product question", "product issue", "product information",
        "specifications", "features", "availability", "stock", "in stock", "out of stock",
        "backorder", "release date", "new product", "latest product", "upcoming product",
        "product comparison", "compare products", "product range", "product selection",
        "product catalog", "catalogue", "product line", "product model", "model number",
        "product version", "product upgrade", "accessories", "product accessories",
        "product quality", "warranty", "guarantee", "product support", "customer support",
        "product guide", "user guide", "manual", "instruction manual", "product instructions",
        "usage", "how to use", "how to operate", "product operation", "product troubleshooting",
        "product help", "product assistance", "product advice", "recommend product",
        "product recommendation", "best product", "top product", "product review", "product feedback"
    ]
    refund_keywords = refund_keywords = [
        "refund", "return", "money back", "reimbursement", "exchange", "repayment", "compensation",
        "refund policy", "refund request", "refund status", "refund process", "refund query",
        "refund issue", "refund procedure", "refund eligibility", "refund terms", "refund conditions",
        "return process", "return request", "return policy", "return item", "return order",
        "cancel order", "cancel purchase", "reverse payment", "reverse charge", "refund transaction",
        "refund amount", "refund confirmation", "refund application", "refund approval", "refund denied",
        "return goods", "return merchandise", "return product", "refund for return", "refund window",
        "refund period", "refund deadline", "refund timeframe", "refund method", "refund option",
        "full refund", "partial refund", "no refund", "chargeback", "dispute charge",
        "cancel transaction", "revoke payment", "refund dispute", "refund claim", "refund appeal"
    ]
    technical_keywords = [
        "problem", "issue", "not working", "crash", "error", "bug", "fault", "glitch", "technical",
        "defect", "malfunction", "broken", "faulty", "trouble", "difficulty", "complication",
        "hindrance", "obstacle", "setback", "snag", "not starting", "not working right",
        "crashing", "failure", "system down", "unresponsive", "freeze", "frozen", "hang", "stuck",
        "slow", "performance issue", "connectivity problem", "network issue", "connection issue",
        "downtime", "outage", "disruption", "interruption", "corruption", "data loss",
        "sync issue", "synchronization problem", "login issue", "access problem", "authentication issue",
        "security breach", "vulnerability", "exploit", "hack", "cyber attack", "malware", "virus",
        "spyware", "adware", "trojan", "worm", "ransomware", "phishing", "spam", "scam", "fraud"
    ]
    if any(keyword in text for keyword in billing_keywords):
        return "Billing Inquiry"
    elif any(keyword in text for keyword in cancellation_keywords):
        return "Cancellation Request"
    elif any(keyword in text for keyword in product_keywords):
        return "Product Inquiry"
    elif any(keyword in text for keyword in refund_keywords):
        return "Refund Request"
    elif any(keyword in text for keyword in technical_keywords):
        return "Technical Issue"
    else:
        return "Unknown"

# Load the dataset
df = pd.read_csv('data_set/customer_support_tickets.csv')

# Preprocess the text data from the dataset
df['Ticket Description'] = df['Ticket Description'].astype(str).apply(preprocess_text)

X = df['Ticket Description']
y = df['Ticket Type']

# Split the dataset into training and testing sets
# Here, 40% of the data is used for testing and the rest for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a machine learning pipeline for training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Define hyperparameters for grid search
param_grid = {
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'classifier__max_iter': [100, 200, 500]
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Predict and evaluate the model on the test set
predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))

# Example interactive loop for classifying new tickets
while True:
    user_input = input("Enter your ticket text (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_category = classify_ticket_type(user_input)
    print(f"The ticket is classified as: {predicted_category}")

print("Exiting the program. Thank you!")