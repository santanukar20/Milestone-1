"""Test different data sources to find one with static HTML"""
import requests
from bs4 import BeautifulSoup

# Test sources
test_urls = [
    # AMC Direct websites
    ("HDFC AMC", "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct"),
    ("ICICI Prudential", "https://www.icicipruamc.com/mutual-fund/icici-prudential-bluechip-fund/direct-plan"),
    ("SBI MF", "https://www.sbimf.com/en-us/individual/our-products/mutual-funds/equity/sbi-bluechip-fund"),
    
    # Aggregator platforms
    ("Groww", "https://groww.in/mutual-funds/hdfc-flexi-cap-fund-direct-growth"),
    ("Value Research", "https://www.valueresearchonline.com/funds/16593/hdfc-flexi-cap-fund-direct-plan"),
    ("Moneycontrol", "https://www.moneycontrol.com/mutual-funds/hdfc-flexi-cap-fund-direct-plan/portfolio-overview/MHD002"),
    ("Morningstar", "https://www.morningstar.in/mutualfunds/f00000oqgz/hdfc-flexi-cap-fund-direct-growth/portfolio.aspx"),
    ("ETMoney", "https://www.etmoney.com/mutual-funds/hdfc-flexi-cap-fund-direct-plan-18"),
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

print("Testing data sources for static HTML content...\n")
print("="*100)

for name, url in test_urls:
    try:
        print(f"\n{name}: {url}")
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        
        if response.status_code != 200:
            print(f"  ✗ HTTP {response.status_code}")
            continue
        
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check for common fund data indicators
        text = soup.get_text().lower()
        
        indicators = {
            'NAV': any(x in text for x in ['nav', '₹']),
            'Returns': any(x in text for x in ['return', '1y', '3y', '5y', '1 year', '3 year']),
            'PE Ratio': 'p/e' in text or 'pe ratio' in text or 'price to earning' in text,
            'Fund Size': 'aum' in text or 'fund size' in text or 'assets under' in text,
            'Expense': 'expense ratio' in text or 'ter' in text,
        }
        
        found = sum(indicators.values())
        
        if found >= 3:
            print(f"  ✓ Looks promising! Found {found}/5 indicators")
            print(f"    HTML length: {len(html)} chars")
            for key, val in indicators.items():
                print(f"    {key}: {'✓' if val else '✗'}")
        else:
            print(f"  ✗ Likely JavaScript-rendered ({found}/5 indicators)")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*100)
print("\nRECOMMENDATION:")
print("Look for sources with ✓ and 3+ indicators - they likely have static HTML")
