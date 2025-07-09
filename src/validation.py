import json
import numpy as np
import matplotlib.pyplot as plt
from query_executor import load_data, process_batch_query
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
    chunks, embeddings = load_data()
    
    for section in data['questions']:
        section_name = section['section']
        section_scores = {'similarity': [], 'pass_threshold': []}
        
        for q in section['questions']:
            # Generar respuesta de IA
            ai_response = process_batch_query(q['question'], chunks, embeddings)
            log_line = f"LOG | query: '{q['question']}' | ai_response: '{ai_response}'"
            log_entry(log_line)
            
            # Evaluar respuesta
            metrics = evaluate_response(ai_response, q['answer'])
            
            # Almacenar resultados
            result = {
                'section': section_name,
                'question': q['question'],
                'metrics': metrics,
                'product': None
            }
            
            results.append(result)
            # Acumular métricas de sección
            for k in metrics:
                section_scores[k].append(metrics[k])
    
    return results

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

def generate_plot(aggregates):
    # 1. Bar plot of average similarity per section
    sections = list(aggregates.keys())
    avg_similarity = [aggregates[sec]['similarity'] for sec in sections]

    plt.figure(figsize=(10, 6))
    plt.bar(sections, avg_similarity, color='skyblue')
    plt.title("Promedio de Similitud por Sección")
    plt.xlabel("Sección")
    plt.ylabel("Similitud")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

    similarities = [v['similarity'] for v in aggregates.values()]
    global_avg = sum(similarities) / len(similarities)

    plt.figtext(0.5, 0.01, f"Promedio Global Similitud: {global_avg:.4f}",
                ha="center", fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Make room for bottom text
    plt.savefig(metrics_path / "efectividad_secciones.png")
    plt.close()

def generate_table(aggregates):
    headers = ["Secciones", "Similitud", "Pass Threshold"]
    table_data = []

    for section, values in aggregates.items():
        similarity = f"{values['similarity']:.3f}" 
        pass_threshold = f"{values['pass_threshold']:.3f}"
        table_data.append([section, similarity, pass_threshold])


    table_data.sort(key=lambda x: float(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(8, 6)) 

    ax.axis('off')
    ax.axis('tight')


    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center') 


    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) 


    for (row, col), cell in table.get_celld().items():
        if row == 0: 
            cell.set_facecolor("#406882") 
            cell.set_text_props(color='white', fontweight='bold')
        elif float(table_data[row-1][1]) >= float(table_data[row-1][2]):
            cell.set_facecolor("#F8D7DA")
        else:
            cell.set_facecolor("#D4EDDA")


        if col == 0:
            cell.set_width(0.3)
        elif col in [1, 2]:
            cell.set_width(0.2)


    ax.set_title("Similitud por Seccion", fontsize=16, pad=20)

    plt.tight_layout() 
    plt.savefig(metrics_path / "tabla_efectividad_secciones.png")
    plt.close()    


def log_entry(data, log_file="log.txt"):
    os.makedirs(metrics_path, exist_ok=True)

    entry = {
        "data": data
    }

    with open(metrics_path / log_file, "a", encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_metrics(aggregates, log_file="log.txt"):
    log_entry(aggregates, log_file)

    for section, metrics in aggregates.items():
        print(f"\n{section}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

# Ejecución principal
if __name__ == "__main__":
    validation_data = load_val_data()

    # Procesar datos
    results = process_data(validation_data)
    
    # Calcular agregados
    aggregates = calculate_aggregates(results)
    
    # Generar reporte
    print("\nReporte de Métricas por Sección:")
    save_metrics(aggregates)
    
    # Generar gráficos
    generate_plot(aggregates)
    generate_table(aggregates)
    print("\nGráficos generados: efectividad_secciones.png, tabla_efectividad_secciones")
