import networkx as nx
import matplotlib.pyplot as plt

products = [
  {
    "name": "1808 Colombia Natural",
    "Tipo": "100% Arábica",
    "Origen": "Colombia (>1200m altitud)",
    "Tueste": "Artesanal en Móstoles",
    "Destacado": "Café de especialidad para cafeteros exigentes",
    "Formatos": {
      "1kg": "23.75€",
      "500g": "11.90€",
      "250g": "6.45€"
    },
    "Molido": [
      "Express",
      "Filtro",
      "Italiana",
      "Prensa francesa",
      "Grano"
    ],
    "Prep": "Cafetera italiana: llenar sin prensar"
  },
  {
    "name": "1808 Descafeinado Natural",
    "Tipo": "Blend Arábica/Robusta",
    "Proceso": "Descafeinado artesanal",
    "Destacado": "Café de especialidad para cafeteros exigentes",
    "Formatos": {
      "1kg": "23.75€",
      "500g": "11.90€",
      "250g": "6.45€"
    },
    "Molido": [
      "Express",
      "Filtro",
      "Italiana",
      "Prensa francesa",
      "Grano"
    ]
  },
  {
    "name": "1808 Arábica Natural",
    "Origen": "Brasil/Honduras/Colombia",
    "Destacado": "Sabor potente con postgusto prolongado",
    "Formatos": {
      "1kg": "21.90€",
      "500g": "11.00€",
      "250g": "5.95€"
    },
    "Molido": [
      "Express",
      "Filtro",
      "Italiana",
      "Prensa francesa",
      "Grano"
    ]
  },
  {
    "name": "1808 Arábica 80/20",
    "Mezcla": "80% Natural + 20% Torrefacto",
    "Destacado": "Crema voluminosa",
    "Formatos": {
      "1kg": "21.90€",
      "500g": "11.00€",
      "250g": "5.95€"
    },
    "Molido": [
      "Express",
      "Filtro",
      "Italiana",
      "Prensa francesa",
      "Grano"
    ]
  },
  {
    "name": "Cápsulas Máxima",
    "Contenido": "50 uds (Etiopía/México/Guatemala/Brasil)",
    "Precio": "13.75€",
    "Uso": "Exclusivo máquinas 1808",
    "Formatos": ["Capsula"]
  },
  {
    "name": "Cápsulas Italiano",
    "Contenido": "50 uds",
    "Mezcla": "Arábica Brasil + Robusta India",
    "Tueste": "Italiano intenso",
    "Precio": "13.75€",
    "Formatos": ["Capsula"],
    "Uso": "Exclusivo máquinas 1808"
  },
  {
    "name": "Cápsulas Descafeinado",
    "Contenido": "50 uds",
    "Origen": "Colombia/Brasil/Vietnam",
    "Proceso": "Descafeinado natural",
    "Precio": "13.75€",
    "Formatos": ["Capsula"],
    "Uso": "Exclusivo máquinas 1808"
  }
]


G = nx.Graph()

# Add nodes and edges product -> attribute
for product in products:
    product_name = product["name"]
    G.add_node(product_name, type='product')
    
    # Iterate over attributes except 'name'
    for attr_name, attr_value in product.items():
        if attr_name == "name":
            continue
        
        if isinstance(attr_value, dict):
            # For nested dict like 'Formatos'
            for k, v in attr_value.items():
                attr_node = f"{attr_name}: {k} = {v}"
                G.add_node(attr_node, type='attribute')
                G.add_edge(product_name, attr_node, label=attr_name)
        
        elif isinstance(attr_value, list):
            # For list attributes like 'Molido' or 'Formatos' (capsula)
            for v in attr_value:
                attr_node = f"{attr_name}: {v}"
                G.add_node(attr_node, type='attribute')
                G.add_edge(product_name, attr_node, label=attr_name)
        
        else:
            attr_node = f"{attr_name}: {attr_value}"
            G.add_node(attr_node, type='attribute')
            G.add_edge(product_name, attr_node, label=attr_name)

# Layout and draw
pos = nx.spring_layout(G, k=0.5)

plt.figure(figsize=(8, 6)) 


node_colors = ['lightblue' if G.nodes[n]['type'] == 'product' else 'lightgreen' for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=node_colors, font_size=8, node_size=500)

plt.title("Grafo de Atributos de Productos")
plt.show()

