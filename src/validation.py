import json
import re
import numpy as np
import matplotlib.pyplot as plt
from query_executor import load_data, process_batch_query
from datetime import datetime
import os
from pathlib import Path
from config.openai_client import client
from dotenv import load_dotenv
from util.generate_response import cosine_similarity

load_dotenv()


metrics_path = Path("../metrics")

# Configuración inicial
validation_data = {}



def load_val_data():
    with open("../data/val.json") as f:
        return json.load(f)

def evaluate_response(ai_response, ideal_answer, threshold=0.6):
    """Evalúa la respuesta usando métricas de similitud semántica"""
    inputs = [f"passage: {ai_response}", f"passage: {ideal_answer}"]
    response = client.embeddings.create(model=os.getenv("EMBEDDINGS_MODEL"), input=inputs)

    emb_ai = response.data[0].embedding
    emb_ideal = response.data[1].embedding

    similarity = cosine_similarity(emb_ai, emb_ideal)
    is_pass = similarity >= threshold
    # similarity	How close the two texts are semantically
    # pass_threshold	Whether this closeness is enough to count as "correct" or "acceptable"

    return {
        "similarity": similarity,
        "pass_threshold": is_pass
    }


def process_data(data):
    results = []
    product_metrics = []

    chunks, embeddings = load_data()
    
    for section in data['questions']:
        section_name = section['section']
        section_scores = {'similarity': [], 'pass_threshold': []}
        
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
            aggregates[section] = {'similarity': [], 'pass_threshold': []}
        
        for metric in result['metrics']:
            aggregates[section][metric].append(result['metrics'][metric])
    
    # Calcular promedios
    for section, metrics in aggregates.items():
        for metric, values in metrics.items():
            metrics[metric] = np.mean(values)
    
    return aggregates

def generate_plots(aggregates, product_metrics):
    # 1. Bar plot of average similarity per section
    sections = list(aggregates.keys())
    avg_similarity = [aggregates[sec]['similarity'] for sec in sections]

    plt.figure(figsize=(10, 6))
    plt.bar(sections, avg_similarity, color='skyblue')
    plt.title("Average Similarity per Section")
    plt.xlabel("Section")
    plt.ylabel("Average Similarity")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(metrics_path / "efectividad_secciones.png")
    plt.close()

    # 2. Bar plot of pass_threshold rate per product (percentage passing threshold)
    if product_metrics:
        products = list(set(pm['product'] for pm in product_metrics if pm['product'] is not None))
        products.sort()

        pass_rates = []
        for p in products:
            p_metrics = [pm for pm in product_metrics if pm['product'] == p]
            pass_rate = sum(pm['pass_threshold'] for pm in p_metrics) / len(p_metrics)
            pass_rates.append(pass_rate)

        plt.figure(figsize=(10, 6))
        plt.bar(products, pass_rates, color='lightgreen')
        plt.title("Pass Threshold Rate per Product")
        plt.xlabel("Product")
        plt.ylabel("Pass Rate")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(metrics_path / "desempeno_productos.png")
        plt.close()

    # 3. Histogram of similarity scores distribution across all results
    all_similarities = []
    for section in aggregates.keys():
        for result in product_metrics:
            if result['product'] is not None:
                all_similarities.append(result['similarity'])

    if not all_similarities:  # fallback: collect from results if product_metrics empty
        all_similarities = [res['metrics']['similarity'] for res in results]

    plt.figure(figsize=(8, 6))
    plt.hist(all_similarities, bins=20, color='salmon', edgecolor='black')
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(metrics_path / "distribucion_similaridad.png")
    plt.close()



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
