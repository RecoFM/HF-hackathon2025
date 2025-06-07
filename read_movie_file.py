import json
from collections import Counter

input_path = 'data/Movies_and_TV.jsonl'

parent_asins = []
lines_processed = 0

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        lines_processed += 1
        if lines_processed % 100000 == 0:
            print(f"Processed {lines_processed} lines...")

        try:
            row = json.loads(line)
            parent_asin = row.get('parent_asin')
            if parent_asin:
                parent_asins.append(parent_asin)
        except json.JSONDecodeError as e:
            print(f"Skipping line {lines_processed} due to JSON error: {e}")

# Count occurrences
counter = Counter(parent_asins)

# Find duplicates
duplicates = {asin: count for asin, count in counter.items() if count > 1}

# Print results
print(f"\nProcessed {lines_processed} total lines.")
print(f"Total unique parent_asin values: {len(counter)}")
print(f"Number of duplicate parent_asins: {len(duplicates)}")
print("\nTop 10 most common parent_asins:")
for asin, count in counter.most_common(10):
    print(f"{asin}: {count} occurrences")