# verify_labels.py

import os

def verify_labels(directory):
    invalid_entries = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    parts = line.strip().split()
                    if len(parts) < 1:
                        continue
                    try:
                        class_id = int(parts[0])
                        if class_id not in [0, 1]:
                            invalid_entries.append((filepath, line_num, class_id))
                    except ValueError:
                        invalid_entries.append((filepath, line_num, 'Non-integer class ID'))
    return invalid_entries

# Define directories
base_dir = "C:/Users/Aloof/Desktop/waste detection"
splits = ['train', 'valid', 'test']

all_invalid_entries = []

for split in splits:
    label_dir = os.path.join(base_dir, split, 'labels')
    print(f"Checking {label_dir}...")
    invalid = verify_labels(label_dir)
    if invalid:
        all_invalid_entries.extend(invalid)

if all_invalid_entries:
    print("\nFound invalid class IDs:")
    for entry in all_invalid_entries:
        filepath, line_num, class_id = entry
        print(f"File: {filepath}, Line: {line_num}, Class ID: {class_id}")
    print("\nPlease ensure all class IDs are either 0 or 1.")
else:
    print("\nAll annotation files are correctly relabelled with class IDs 0 and 1.")
