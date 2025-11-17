import json

# Load embeddings data
with open('data/embeddings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data['documents']
metadata = data['metadata']

print("Checking for HDFC Flexi Cap Fund risk level data...")

# Look for risk level information
found = False
for i, doc in enumerate(docs):
    if 'HDFCFlexi Cap Fund' in doc and 'Very High' in doc:
        print(f"Found risk level information in document {i}")
        print(f"Document preview: {doc[:300]}...")
        found = True
        break

if found:
    print("✓ Risk level chunk found")
else:
    print("✗ Risk level chunk not found")

# Look for NAV information (should be None/missing)
print("\nChecking for NAV information...")
nav_found = False
for i, doc in enumerate(docs):
    if 'HDFCFlexi Cap Fund' in doc and 'NAV' in doc and '₹' in doc:
        print(f"Found NAV information in document {i}")
        print(f"Document preview: {doc[:200]}...")
        nav_found = True
        break

if nav_found:
    print("⚠️  NAV information found (unexpected)")
else:
    print("✓ NAV correctly not available in chunks")

# Look for expense ratio information
print("\nChecking for expense ratio information...")
ter_found = False
for i, doc in enumerate(docs):
    if 'HDFCFlexi Cap Fund' in doc and '0.67%' in doc:
        print(f"Found expense ratio in document {i}")
        print(f"Document preview: {doc[:200]}...")
        ter_found = True
        break

if ter_found:
    print("✓ Expense ratio found")
else:
    print("✗ Expense ratio not found")