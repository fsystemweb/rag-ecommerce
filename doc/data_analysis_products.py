import networkx as nx
import matplotlib.pyplot as plt
import re


def read_md_file_and_parsing_products():
  # Step 1: Read Markdown File
  with open('../data/data.md', 'r', encoding='utf-8') as f:
      md_content = f.read()

  # Step 2: Extract Product Sections
  info_productos_block = re.search(
      r"## \*\*Información de Productos\*\*\n(.*?)(?=\n## |\Z)", 
      md_content, 
      re.DOTALL
  )

  productos_text = info_productos_block.group(1)

  product_blocks = re.findall(
    r"### (.*?)\n(.*?)(?=\n### |\Z)", 
    productos_text, 
    re.DOTALL
  )

  # Step 3: Parse Each Product Block
  for title, block in product_blocks:
    product = {"name": title.strip()}
    lines = block.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^- \*\*(.*?)\*\*: (.*)", line):
            key, value = re.findall(r"^- \*\*(.*?)\*\*: (.*)", line)[0]
            product[key.strip()] = value.strip()

        elif re.match(r"^- `.*?`: .*?€", line):
            if "Tamaños" not in product:
                product["Tamaños"] = []
                product["TamañosPrecios"] = {}
            size, price = re.findall(r"`(.*?)`: (.*?)€", line)[0]
            product["Tamaños"].append(size)
            product["TamañosPrecios"][size] = f"{price}€"

        elif re.match(r"^- \[x\] (.+)", line):
            if "Formatos" not in product:
                product["Formatos"] = []
            formato = re.findall(r"^- \[x\] (.+)", line)[0]
            product["Formatos"].append(formato.strip())

        elif ": " in line:
            key, value = line.split(": ", 1)
            product[key.strip()] = value.strip()
    products.append(product)



products = []

G = nx.Graph()

read_md_file_and_parsing_products()

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

