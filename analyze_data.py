"""Analyze scraped data to see what we have"""
import json

records = [json.loads(line) for line in open('data/raw_scheme_pages.jsonl', encoding='utf-8')]

print(f"Total records: {len(records)}\n")
print("="*100)

# Group by source
from collections import defaultdict
by_source = defaultdict(list)
for r in records:
    url = r.get('source_url', '')
    if 'hdfcfund.com' in url:
        by_source['HDFC Website'].append(r)
    elif 'groww.in' in url:
        by_source['Groww.in'].append(r)
    else:
        by_source['Unknown'].append(r)

for source, recs in by_source.items():
    print(f"\n{source}: {len(recs)} records")
    print("-"*100)
    
    for i, r in enumerate(recs, 1):
        # Check if record has any actual data
        has_data = any([
            r.get('scheme_name'),
            r.get('nav_value'),
            r.get('pe_ratio'),
            r.get('fund_size_cr'),
            r.get('returns_1y_percent')
        ])
        
        status = "✓ HAS DATA" if has_data else "✗ NO DATA"
        
        print(f"\n  {i}. {status}")
        print(f"     Scheme: {r.get('scheme_name') or 'None'}")
        print(f"     NAV: ₹{r.get('nav_value') or 'None'}")
        print(f"     PE Ratio: {r.get('pe_ratio') or 'None'}")
        print(f"     Fund Size: ₹{r.get('fund_size_cr') or 'None'} Cr")
        print(f"     1Y Return: {r.get('returns_1y_percent') or 'None'}%")
        print(f"     URL: {r.get('source_url')}")
        scraped = r.get('last_scraped_at', '')[:19] if r.get('last_scraped_at') else 'Unknown'
        print(f"     Scraped: {scraped}")

print("\n" + "="*100)
print("\nSUMMARY:")
print(f"  Total records: {len(records)}")
print(f"  HDFC Website: {len(by_source['HDFC Website'])}")
print(f"  Groww.in: {len(by_source['Groww.in'])}")

# Count records with actual data
records_with_data = sum(1 for r in records if any([
    r.get('scheme_name'),
    r.get('nav_value'),
    r.get('pe_ratio')
]))
records_without_data = len(records) - records_with_data

print(f"  Records WITH data: {records_with_data}")
print(f"  Records WITHOUT data (all null): {records_without_data}")
