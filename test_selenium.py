"""Test Selenium setup"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

print("Setting up Chrome options...")
chrome_options = Options()
chrome_options.add_argument('--headless=new')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1920,1080')

print("Initializing driver...")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

url = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
print(f"Navigating to {url}...")
driver.get(url)

print("Waiting for page to load...")
time.sleep(5)

html = driver.page_source
print(f"Page loaded! HTML length: {len(html)} chars")

# Check for NAV in the HTML
if "₹" in html:
    print("✓ Found currency symbol in HTML")
else:
    print("✗ No currency symbol found")

if "NAV" in html or "nav" in html.lower():
    print("✓ Found NAV text in HTML")
else:
    print("✗ No NAV text found")

# Save a snippet for debugging
print("\nFirst 500 chars of HTML:")
print(html[:500])

driver.quit()
print("\nTest complete!")
