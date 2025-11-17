"""Test Quant AMC website for data availability"""
import requests
from bs4 import BeautifulSoup
import json

url = "https://quantmutual.com/equity/quant-growth"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

print(f"Testing: {url}\n")
print("="*100)

try:
    response = requests.get(url, headers=headers, timeout=10, verify=False)
    
    print(f"\nHTTP Status: {response.status_code}")
    
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        print(f"HTML Length: {len(html)} chars")
        
        # Extract all visible text
        text = soup.get_text(separator=' ', strip=True).lower()
        
        print("\n" + "="*100)
        print("CHECKING FOR KEY DATA FIELDS:")
        print("="*100)
        
        # Check for required fields
        fields_to_check = {
            'NAV': ['nav', '₹', 'rupee'],
            'Returns (1Y/3Y/5Y)': ['1 year', '3 year', '5 year', '1y', '3y', '5y', 'returns', 'cagr'],
            'P/E Ratio': ['p/e', 'pe ratio', 'price to earning', 'price earning'],
            'Fund Size/AUM': ['aum', 'fund size', 'assets under management', 'corpus'],
            'Expense Ratio': ['expense ratio', 'ter', 'total expense'],
            'Exit Load': ['exit load', 'redemption'],
            'Minimum Investment': ['minimum', 'min investment', 'lumpsum', 'sip'],
            'Fund Manager': ['fund manager', 'managed by'],
            'Risk Level': ['risk', 'volatility', 'riskometer'],
            'Beta': ['beta'],
            'Sharpe Ratio': ['sharpe'],
        }
        
        found_fields = {}
        for field_name, keywords in fields_to_check.items():
            found = any(keyword in text for keyword in keywords)
            found_fields[field_name] = found
            status = "✓ FOUND" if found else "✗ NOT FOUND"
            print(f"  {status:15} - {field_name}")
        
        total_found = sum(found_fields.values())
        total_fields = len(found_fields)
        
        print("\n" + "="*100)
        print(f"SUMMARY: {total_found}/{total_fields} fields found ({total_found/total_fields*100:.0f}%)")
        
        if total_found >= 7:
            print("✓ EXCELLENT - This source has most required data!")
        elif total_found >= 5:
            print("✓ GOOD - This source has enough data to proceed")
        elif total_found >= 3:
            print("⚠ MODERATE - Some data available but might need additional sources")
        else:
            print("✗ POOR - Likely JavaScript-rendered or insufficient data")
        
        print("\n" + "="*100)
        print("SAMPLE TEXT (first 1000 chars):")
        print("="*100)
        print(soup.get_text(separator=' ', strip=True)[:1000])
        
        # Try to extract NAV specifically
        print("\n" + "="*100)
        print("TRYING TO EXTRACT SPECIFIC DATA:")
        print("="*100)
        
        import re
        
        # Look for NAV value
        nav_pattern = r'₹\s*(\d+[\d,]*\.?\d*)'
        nav_matches = re.findall(nav_pattern, html)
        if nav_matches:
            print(f"  NAV candidates: {nav_matches[:3]}")
        else:
            print("  NAV: No matches found")
        
        # Look for returns
        return_pattern = r'(\d+\.?\d*)\s*%'
        return_matches = re.findall(return_pattern, text)
        if return_matches:
            print(f"  Return % candidates: {return_matches[:5]}")
        else:
            print("  Returns: No matches found")
            
except Exception as e:
    print(f"\n✗ ERROR: {e}")

print("\n" + "="*100)
