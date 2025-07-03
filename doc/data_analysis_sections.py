import re
import matplotlib.pyplot as plt
import numpy as np

# Load markdown file content
with open('../data/data.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

# Sections you want to extract (exact names from your md file)
sections_to_extract = [
    "Sobre Nosotros",
    "Devoluciones y Reembolsos",
    "Envíos",
    "Garantía",
    "Información de Productos",
    "Forma de Pago",
    "Contacto"
]

# Prepare dictionary to hold extracted content
extracted = {section: "" for section in sections_to_extract}

# Regex to find section headers like ## **Section Name**
pattern = re.compile(r'## \*\*(.+?)\*\*')

# Find all matches and their positions
matches = list(pattern.finditer(md_text))

# Extract content for each section
for i, match in enumerate(matches):
    section_name = match.group(1).strip()
    start_pos = match.end()
    end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
    content = md_text[start_pos:end_pos].strip()

    if section_name in extracted:
        extracted[section_name] = content

# Calculate lengths of each section's content (character count)
sections_length = {k: len(v) for k, v in extracted.items()}

# Filter out zero-length or invalid values to avoid pie chart errors
filtered_sections = {k: v for k, v in sections_length.items() if v > 0 and not np.isnan(v)}

# Plotting
plt.figure(figsize=(8,6))
plt.pie(filtered_sections.values(), labels=filtered_sections.keys(), autopct='%1.1f%%')
plt.title('Contenido extraído por secciones')
plt.show()
