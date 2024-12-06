# relabel_annotations.py

import os

# Define Paths
# Update these paths based on your project structure
ANNOT_DIRS = {
    'train': 'C:/Users/Aloof/Desktop/waste detection/train/labels',
    'valid': 'C:/Users/Aloof/Desktop/waste detection/valid/labels',
    'test': 'C:/Users/Aloof/Desktop/waste detection/test/labels'
}

OUTPUT_DIR_BASE = 'C:/Users/Aloof/Desktop/Projectt/relabelled_labels'

# Create Output Directories if they don't exist
for split in ANNOT_DIRS.keys():
    output_dir = os.path.join(OUTPUT_DIR_BASE, split, 'labels')
    os.makedirs(output_dir, exist_ok=True)

# Mapping Dictionary
# Original class IDs (0-12) mapped to new class IDs (0 or 1)
mapping = {
    0: 0,  # banana -> biodegradable
    1: 0,  # chilli -> biodegradable
    2: 1,  # drinkcan -> non_biodegradable
    3: 1,  # drinkpack -> non_biodegradable
    4: 1,  # foodcan -> non_biodegradable
    5: 0,  # lettuce -> biodegradable
    6: 0,  # paperbag -> biodegradable
    7: 1,  # plasticbag -> non_biodegradable
    8: 1,  # plasticbottle -> non_biodegradable
    9: 1,  # plasticcontainer -> non_biodegradable
    10: 0, # sweetpotato -> biodegradable
    11: 0, # teabag -> biodegradable
    12: 0  # tissueroll -> biodegradable
}

# Process Each Annotation File in All ANNOT_DIRS
for split, annot_dir in ANNOT_DIRS.items():
    output_dir = os.path.join(OUTPUT_DIR_BASE, split, 'labels')
    for filename in os.listdir(annot_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(annot_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Skip invalid lines
                    original_class = int(parts[0])
                    new_class = mapping.get(original_class, original_class)  # Default to original if not mapped
                    # Replace the class_id with new_class
                    new_line = f"{new_class} {' '.join(parts[1:])}\n"
                    outfile.write(new_line)

print("Relabeling completed successfully.")
