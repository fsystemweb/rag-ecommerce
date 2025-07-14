import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import StringIO


stopwords_es = set([
    'para', 'siguiente', 'esta', 'desde', 'todos', 'que', 'los', 'del', 'con', 'las', 'por', 'una', 'caso'
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Eliminar URLs
    text = re.sub(r'[^\w\s]', '', text) # Eliminar puntuación
    text = re.sub(r'\d+', '', text) # Eliminar números
    text = re.sub(r'\s+', ' ', text).strip() # Eliminar espacios extra
    return text

def tokenize_and_filter_words(text):
    words = text.split()
    # Filtrar palabras que están en la lista de stopwords o son muy cortas
    filtered_words = [word for word in words if word not in stopwords_es and len(word) > 2]
    return filtered_words

# Cargar data
with open('../data/data.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

cleaned_data = clean_text(md_text)
tokens = tokenize_and_filter_words(cleaned_data)

term_frequencies = Counter(tokens)

# Obtener los 20 términos más comunes
most_common_terms = term_frequencies.most_common(20)
df_terms = pd.DataFrame(most_common_terms, columns=['Termino', 'Frecuencia'])

# Crear el gráfico de barras
plt.figure(figsize=(12, 7))
sns.barplot(x='Frecuencia', y='Termino', data=df_terms, palette='viridis')
plt.title('Top 20 Términos Más Frecuentes', fontsize=16)
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Término', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Análisis de Frecuencia de Términos Completado.")
print("\nTop 20 Términos y sus Frecuencias:")
print(df_terms)
