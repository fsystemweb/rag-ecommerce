import json
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from query_executor import load_data, process_batch_query
from datetime import datetime
import os
from pathlib import Path

metrics_path = Path("../metrics")

# Configuración inicial
validation_data = {}

def load_val_data():
    with open("../data/val.json") as f:
        return json.load(f)

def evaluate_response(ai_response, ideal_answer, threshold=0.6):
    """Evalúa la respuesta usando métricas de similitud semántica"""
    # Add context prompt
    vectorizer = TfidfVectorizer().fit_transform([ai_response, ideal_answer])
    similarity = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    y_true = [1]
    y_pred = [1 if similarity >= threshold else 0]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def process_data(data):
    results = []
    product_metrics = []

    chunks, embeddings = load_data()
    
    for section in data['questions']:
        section_name = section['section']
        section_scores = {'precision': [], 'recall': [], 'f1': []}
        
        for q in section['questions']:
            # Generar respuesta de IA
            ai_response = process_batch_query(q['question'], chunks, embeddings)
            print(f"LOG | query: '{q['question']}' | ai_response: '{ai_response}'")
            
            # Evaluar respuesta
            metrics = evaluate_response(ai_response, q['answer'])
            
            # Almacenar resultados
            result = {
                'section': section_name,
                'question': q['question'],
                'metrics': metrics,
                'product': None
            }
            
            # Identificar productos mencionados
            if "Información de Productos" in section_name:
                product_match = re.search(r"('.*?')", q['question'])
                if product_match:
                    product = product_match.group(1)
                    result['product'] = product
                    product_metrics.append({
                        'product': product,
                        **metrics
                    })
            
            results.append(result)
            # Acumular métricas de sección
            for k in metrics:
                section_scores[k].append(metrics[k])
    
    return results, product_metrics

def calculate_aggregates(results):
    aggregates = {}
    for result in results:
        section = result['section']
        if section not in aggregates:
            aggregates[section] = {'precision': [], 'recall': [], 'f1': []}
        
        for metric in result['metrics']:
            aggregates[section][metric].append(result['metrics'][metric])
    
    # Calcular promedios
    for section, metrics in aggregates.items():
        for metric, values in metrics.items():
            metrics[metric] = np.mean(values)
    
    return aggregates

def generate_plots(aggregates, product_metrics):
    # 1. Gráfico por secciones
    sections = list(aggregates.keys())
    metrics = ['precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    x = np.arange(len(sections))
    
    for i, metric in enumerate(metrics):
        values = [aggregates[section][metric] for section in sections]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_title('Efectividad por Sección')
    ax.set_ylabel('Puntuación')
    ax.set_xticks(x + width)
    ax.set_xticklabels(sections, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(metrics_path / 'efectividad_secciones.png')
    
    # 2. Gráfico de productos
    if product_metrics:
        products = list(set(pm['product'] for pm in product_metrics))
        product_data = {product: {'precision': [], 'recall': [], 'f1': []} for product in products}
        
        for pm in product_metrics:
            for metric in metrics:
                product_data[pm['product']][metric].append(pm[metric])
        
        # Calcular promedios por producto
        for product, metrics_dict in product_data.items():
            for metric, values in metrics_dict.items():
                product_data[product][metric] = np.mean(values)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.25
        x = np.arange(len(products))
        
        for i, metric in enumerate(metrics):
            values = [product_data[product][metric] for product in products]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_title('Desempeño por Producto')
        ax.set_ylabel('Puntuación')
        ax.set_xticks(x + width)
        ax.set_xticklabels(products, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(metrics_path / 'desempeno_productos.png')
    
    # 3. Gráfico de distribución de F1
    all_f1 = [result['metrics']['f1'] for result in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_f1, bins=15, color='skyblue', edgecolor='black')
    ax.set_title('Distribución de Puntuaciones F1')
    ax.set_xlabel('Puntuación F1')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(metrics_path / 'distribucion_f1.png')


def save_metrics(aggregates, log_file="log.txt"):
    os.makedirs(os.path.dirname(metrics_path / log_file), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "metrics": aggregates
    }

    with open(metrics_path / log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")  

    for section, metrics in aggregates.items():
        print(f"\n{section}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

# Ejecución principal
if __name__ == "__main__":
    validation_data = load_val_data()

    # Procesar datos
    results, product_metrics = process_data(validation_data)
    
    # Calcular agregados
    aggregates = calculate_aggregates(results)
    
    # Generar reporte
    print("\nReporte de Métricas por Sección:")
    save_metrics(aggregates)
    
    # Generar gráficos
    generate_plots(aggregates, product_metrics)
    print("\nGráficos generados: efectividad_secciones.png, desempeno_productos.png, distribucion_f1.png")
