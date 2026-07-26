"""
Rychle porovnani FinBERT vs. model natrenovany na social media textu
(cardiffnlp/twitter-roberta-base-sentiment-latest) na vetach, kde FinBERT
selhal (viz notebook 02).

Cil: zjistit, jestli model urceny pro neformalni/slangovy text zvladne
Reddit-specificke vyrazy ("diamond hands", "to the moon") lip nez FinBERT,
ktery je trenovany na formalnich financnich textech.
"""

from transformers import pipeline

test_texts = [
    "GME to the moon! Diamond hands, never selling",
    "Lost my entire life savings on TSLA puts, guess I'll die",
    "AAPL reports Q3 earnings tomorrow, expecting inline results",
    "This stock is going to zero, sold everything",
    "Absolutely printing money on NVDA calls today",
]

print("=" * 70)
print("FinBERT (financni texty)")
print("=" * 70)
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
for text, result in zip(test_texts, finbert(test_texts)):
    print(f"{result['label']:10s} ({result['score']:.3f})  {text}")

print()
print("=" * 70)
print("Twitter-RoBERTa (social media texty)")
print("=" * 70)
social = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
)
for text, result in zip(test_texts, social(test_texts)):
    print(f"{result['label']:10s} ({result['score']:.3f})  {text}")