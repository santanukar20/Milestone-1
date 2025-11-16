"""
Production-grade data extraction script for Mutual Fund FAQ RAG chatbot.
Scrapes factual data from AMC HTML pages and stores it in structured format.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Allowed AMC domains (easily extensible)
ALLOWED_DOMAINS: Set[str] = {
    'www.hdfcfund.com',
    # Add more AMC domains here as needed:
    # 'www.sbimf.com',
}

# User-Agent to avoid blocking
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Session for connection pooling
session = requests.Session()
session.headers.update(DEFAULT_HEADERS)


def redirect_hook(response, **kwargs):
    """Hook to prevent redirects to non-allowed domains."""
    if response.is_redirect:
        redirect_url = response.headers.get('Location')
        if redirect_url:
            # Resolve relative URLs to absolute URLs
            if not redirect_url.startswith('http'):
                redirect_url = urljoin(response.url, redirect_url)
            
            if redirect_url and not is_allowed_domain(redirect_url):
                logger.warning(f"Blocked redirect to non-allowed domain: {redirect_url}")
                # Raise an exception to prevent the redirect
                raise requests.exceptions.RequestException(
                    f"Redirect blocked to non-allowed domain: {redirect_url}"
                )
    return response


def is_allowed_domain(url: str) -> bool:
    """Check if the URL belongs to an allowed AMC domain."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Normalize domain for comparison
        # Check if domain matches any allowed domain (with or without www.)
        domain_variants = [domain]
        if domain.startswith('www.'):
            domain_variants.append(domain[4:])
        else:
            domain_variants.append(f'www.{domain}')
        
        return any(var in ALLOWED_DOMAINS for var in domain_variants)
    except Exception as e:
        logger.error(f"Error parsing domain from URL {url}: {e}")
        return False


def fetch_html(url: str, timeout: int = 30, max_retries: int = 3) -> Optional[str]:
    """
    Fetch HTML content from a URL with validation and retry logic.
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        HTML content as string, or None if fetch fails
    """
    if not is_allowed_domain(url):
        logger.error(f"FAIL {url} - Domain not in allowed list")
        return None
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff: wait 2^attempt seconds before retry
            if attempt > 0:
                wait_time = 2 ** (attempt - 1)
                logger.info(f"Retry {attempt}/{max_retries-1} for {url} after {wait_time}s")
                time.sleep(wait_time)
            
            # Disallow redirects to non-whitelisted domains
            response = session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                stream=False,  # Don't stream to check content-type early
                hooks={'response': redirect_hook}
            )
            
            # Double-check final URL after redirects
            final_url = response.url
            if not is_allowed_domain(final_url):
                logger.error(f"FAIL {url} - Redirected to non-allowed domain: {final_url}")
                return None
            
            # Validate HTTP status
            if response.status_code != 200:
                logger.error(f"FAIL {url} - HTTP {response.status_code}")
                return None
            
            # Check Content-Type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type:
                logger.error(f"FAIL {url} - Content-Type is PDF, not HTML")
                return None
            
            # Ignore images and JS
            if any(x in content_type for x in ['image/', 'javascript', 'application/javascript']):
                logger.error(f"FAIL {url} - Content-Type is {content_type}, not HTML")
                return None
            
            if 'html' not in content_type and not content_type.startswith('text/'):
                logger.warning(f"WARNING {url} - Unexpected Content-Type: {content_type}, proceeding anyway")
            
            return response.text
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {url}")
                continue
            logger.error(f"FAIL {url} - Request timeout after {max_retries} attempts")
            return None
            
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"Network error on attempt {attempt + 1}/{max_retries} for {url}: {e}")
                continue
            logger.error(f"FAIL {url} - Network error after {max_retries} attempts: {e}")
            return None
            
        except Exception as e:
            last_exception = e
            logger.error(f"FAIL {url} - Unexpected error: {e}")
            return None
    
    return None


def extract_visible_text(soup: BeautifulSoup) -> str:
    """
    Extract all visible text from the HTML, cleaned and normalized.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Clean visible text as a single string
    """
    # Remove script and style elements
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    
    # Get text
    text = soup.get_text(separator=' ', strip=True)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def find_section_text(soup: BeautifulSoup, labels: List[str], case_sensitive: bool = False) -> Optional[str]:
    """
    Find section text by searching for common labels.
    
    Args:
        soup: BeautifulSoup parsed HTML
        labels: List of possible label strings to search for
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        Text content of the section if found, None otherwise
    """
    text_content = soup.get_text()
    if not case_sensitive:
        text_content_lower = text_content.lower()
        labels = [label.lower() for label in labels]
    
    for label in labels:
        # Try to find label in various ways
        # Method 1: Look for label in headings
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b', 'span', 'div']):
            tag_text = tag.get_text(strip=True)
            if not tag_text:
                continue
            
            tag_text_to_match = tag_text if case_sensitive else tag_text.lower()
            label_to_match = label if case_sensitive else label.lower()
            
            if label_to_match in tag_text_to_match:
                # Get text from next sibling or parent's next siblings
                next_sibling = tag.find_next_sibling()
                if next_sibling:
                    text = next_sibling.get_text(separator=' ', strip=True)
                    if text:
                        return text
                
                # Try parent's next sibling
                if tag.parent:
                    next_parent = tag.parent.find_next_sibling()
                    if next_parent:
                        text = next_parent.get_text(separator=' ', strip=True)
                        if text:
                            return text
                
                # Try the parent itself if it contains more text
                parent = tag.parent
                if parent:
                    text = parent.get_text(separator=' ', strip=True)
                    # Remove the label text itself
                    text = text.replace(tag_text, '', 1).strip()
                    if text:
                        return text
        
        # Method 2: Look for label followed by colon or in text
        pattern = re.compile(
            re.escape(label) + r'[:.]?\s*(.+?)(?:\n\n|\n\s*[A-Z][a-z]+[:.]|$)',
            re.IGNORECASE if not case_sensitive else 0,
            re.DOTALL
        )
        match = pattern.search(text_content)
        if match:
            result = match.group(1).strip()
            if result and len(result) > 10:  # Avoid very short matches
                return result
    
    return None


def find_all_section_text(soup: BeautifulSoup, labels: List[str], case_sensitive: bool = False) -> List[str]:
    """
    Find all occurrences of section text by searching for common labels.
    Useful for extracting multiple fund managers.
    
    Args:
        soup: BeautifulSoup parsed HTML
        labels: List of possible label strings to search for
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        List of text content found
    """
    results = []
    text_content = soup.get_text()
    
    if not case_sensitive:
        text_content_lower = text_content.lower()
        labels_lower = [label.lower() for label in labels]
    
    for label in labels:
        label_to_match = label if case_sensitive else label.lower()
        
        # Look for label in headings and extract multiple occurrences
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b', 'span', 'div', 'dt', 'th']):
            tag_text = tag.get_text(strip=True)
            if not tag_text:
                continue
            
            tag_text_to_match = tag_text if case_sensitive else tag_text.lower()
            
            if label_to_match in tag_text_to_match:
                # Get text from next sibling or parent's next siblings
                next_sibling = tag.find_next_sibling()
                if next_sibling:
                    text = next_sibling.get_text(separator=' ', strip=True)
                    if text and text not in results:
                        results.append(text)
                
                # Try parent's next sibling
                if tag.parent:
                    next_parent = tag.parent.find_next_sibling()
                    if next_parent:
                        text = next_parent.get_text(separator=' ', strip=True)
                        if text and text not in results:
                            results.append(text)
                
                # Try the parent itself if it contains more text
                parent = tag.parent
                if parent:
                    text = parent.get_text(separator=' ', strip=True)
                    text = text.replace(tag_text, '', 1).strip()
                    if text and text not in results:
                        results.append(text)
        
        # Also search in text patterns
        pattern = re.compile(
            re.escape(label) + r'[:.]?\s*(.+?)(?:\n\n|\n\s*[A-Z][a-z]+[:.]|$)',
            re.IGNORECASE if not case_sensitive else 0,
            re.DOTALL
        )
        matches = pattern.finditer(text_content)
        for match in matches:
            result = match.group(1).strip()
            if result and len(result) > 10 and result not in results:
                results.append(result)
    
    return results


def extract_fund_managers(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract fund manager(s) information, handling multiple managers.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Fund manager(s) text, or None if not found
    """
    labels = [
        'Fund Manager', 'Fund Managers', 'Manager', 'Managers',
        'Fund Management', 'Managed by', 'Management Team'
    ]
    
    results = find_all_section_text(soup, labels, case_sensitive=False)
    
    if not results:
        return None
    
    # Combine all found managers, removing duplicates
    all_managers = []
    seen = set()
    
    for result in results:
        # Try to split by common delimiters (comma, semicolon, "and", etc.)
        parts = re.split(r'[,;]|\s+and\s+|\s+&\s+', result)
        for part in parts:
            part = part.strip()
            # Clean up common prefixes
            part = re.sub(r'^(Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s*', '', part, flags=re.IGNORECASE)
            if part and len(part) > 3 and part.lower() not in seen:
                seen.add(part.lower())
                all_managers.append(part)
    
    if all_managers:
        return ', '.join(all_managers[:5])  # Limit to 5 managers max
    
    # Fallback: return first result if splitting didn't work
    return results[0]


def extract_suitable_for(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract suitable_for / product_labs information.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Suitable for text, or None if not found
    """
    labels = [
        'Suitable For', 'Suitable for', 'Ideal For', 'Target Investors',
        'Product Label', 'Product Labels', 'Who Should Invest',
        'Investor Profile', 'Target Audience'
    ]
    
    result = find_section_text(soup, labels)
    return result


def extract_factual_text_blocks(soup: BeautifulSoup) -> List[str]:
    """
    Extract all factual text blocks from the page.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        List of factual text blocks
    """
    text_blocks = []
    
    # Find common containers for factual information
    for container in soup.find_all(['section', 'article', 'div', 'main'], class_=re.compile(
        r'(content|fact|info|detail|description|overview|summary)',
        re.IGNORECASE
    )):
        text = container.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Only include substantial text blocks (at least 50 chars)
        if text and len(text) >= 50:
            # Skip if it's mostly navigation or common page elements
            skip_keywords = ['cookie', 'menu', 'navigation', 'footer', 'header', 'sidebar']
            if not any(keyword in text.lower()[:100] for keyword in skip_keywords):
                text_blocks.append(text)
    
    # Also extract from definition lists (common pattern for key-value pairs)
    for dl in soup.find_all('dl'):
        dt_dd_pairs = []
        dt_text = ''
        for child in dl.children:
            if child.name == 'dt':
                dt_text = child.get_text(strip=True)
            elif child.name == 'dd' and dt_text:
                dd_text = child.get_text(strip=True)
                if dd_text:
                    dt_dd_pairs.append(f"{dt_text}: {dd_text}")
        
        if dt_dd_pairs:
            text_blocks.extend(dt_dd_pairs)
    
    # Deduplicate
    seen = set()
    unique_blocks = []
    for block in text_blocks:
        block_lower = block.lower()[:100]  # Use first 100 chars for deduplication
        if block_lower not in seen:
            seen.add(block_lower)
            unique_blocks.append(block)
    
    return unique_blocks


def extract_sections(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """
    Extract structured sections from the parsed HTML.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Dictionary with section keys and extracted values
    """
    sections = {}
    
    # Define label patterns for each section
    section_config = {
        'objective': ['Investment Objective', 'Scheme Objective', 'Objective', 'Fund Objective'],
        'category': ['Category', 'Fund Category', 'Scheme Category', 'Fund Type', 'Scheme Type'],
        'benchmark': ['Benchmark', 'Benchmark Index', 'Index', 'Benchmark Index Name'],
        'riskometer': ['Riskometer', 'Risk Level', 'Risk Rating', 'Risk Profile', 'Risk'],
        'minimum_sip': ['Minimum SIP', 'Minimum Systematic Investment', 'SIP Minimum', 'Minimum Monthly Investment', 'Minimum SIP Investment'],
        'minimum_lumpsum': ['Minimum Investment', 'Minimum Lumpsum', 'Minimum Amount', 'Initial Investment', 'Minimum Lump Sum'],
        'expense_ratio': ['Expense Ratio', 'Total Expense Ratio', 'TER', 'Total Expense Ratio (TER)'],
        'exit_load': ['Exit Load', 'Redemption Load', 'Load', 'Redemption Charge', 'Exit Load Structure'],
        'inception_date': ['Inception Date', 'Launch Date', 'Fund Inception', 'Date of Inception', 'Scheme Inception'],
        'nav_summary': ['NAV', 'Net Asset Value', 'Latest NAV', 'Current NAV'],
    }
    
    for key, labels in section_config.items():
        sections[key] = find_section_text(soup, labels)
    
    # Special handling for fund_manager (extract multiple)
    sections['fund_manager'] = extract_fund_managers(soup)
    
    # Extract suitable_for / product_labs
    sections['suitable_for'] = extract_suitable_for(soup)
    
    # Extract factual text blocks
    factual_blocks = extract_factual_text_blocks(soup)
    sections['factual_text_blocks'] = factual_blocks if factual_blocks else None
    
    return sections


def extract_pdf_links(soup: BeautifulSoup) -> List[str]:
    """
    Extract all PDF links from the page without parsing them.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        List of PDF URLs found on the page
    """
    pdf_links = []
    
    # Find all links
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Handle relative URLs (would need base URL, but storing as-is)
        if href.lower().endswith('.pdf') or 'pdf' in href.lower():
            pdf_links.append(href)
    
    # Also check in iframes, embeds, etc.
    for iframe in soup.find_all(['iframe', 'embed'], src=True):
        src = iframe.get('src', '')
        if 'pdf' in src.lower() or src.lower().endswith('.pdf'):
            pdf_links.append(src)
    
    # Deduplicate
    pdf_links = list(set(pdf_links))
    
    return pdf_links


def parse_scheme_page(html: str, url: str) -> Dict:
    """
    Parse a scheme page HTML and extract structured data.
    
    Args:
        html: Raw HTML content
        url: Source URL
        
    Returns:
        Dictionary with all extracted data
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract page title as fallback scheme name
    title_tag = soup.find('title')
    scheme_name = title_tag.get_text(strip=True) if title_tag else "Unknown Scheme"
    
    # Extract visible text
    text_clean = extract_visible_text(soup)
    
    # Extract structured sections
    sections = extract_sections(soup)
    
    # Build record
    record = {
        "scheme_name": scheme_name,
        "source_url": url,
        "text_clean": text_clean,  # Removed html_raw as per requirements (only store clean text)
        "sections": sections,
        "last_scraped_at": datetime.now(timezone.utc).isoformat()
    }
    
    return record


def save_records(records: List[Dict], output_path: str) -> None:
    """
    Save records to a JSONL file (one JSON object per line).
    
    Args:
        records: List of record dictionaries
        output_path: Path to output JSONL file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(records)} records to {output_path}")


def load_existing_records(output_path: str) -> Dict[str, Dict]:
    """
    Load existing records from JSONL file, keyed by URL.
    
    Args:
        output_path: Path to JSONL file
        
    Returns:
        Dictionary mapping URLs to records
    """
    existing = {}
    output_file = Path(output_path)
    
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        existing[record['source_url']] = record
            logger.info(f"Loaded {len(existing)} existing records from {output_path}")
        except Exception as e:
            logger.warning(f"Could not load existing records: {e}")
    
    return existing


def scrape_scheme_urls(urls: List[str], output_path: str = "data/raw_scheme_pages.jsonl", 
                       update_existing: bool = True) -> Dict[str, int]:
    """
    Main function to scrape multiple scheme URLs.
    
    Args:
        urls: List of scheme URLs to scrape
        output_path: Path to output JSONL file
        update_existing: If True, update existing records; if False, skip
        
    Returns:
        Dictionary with counts: {'attempted': int, 'succeeded': int, 'failed': int}
    """
    stats = {'attempted': 0, 'succeeded': 0, 'failed': 0}
    
    # Load existing records if updating
    existing_records = {}
    if update_existing:
        existing_records = load_existing_records(output_path)
    
    new_records = []
    
    for url in urls:
        stats['attempted'] += 1
        logger.info(f"START {url}")
        
        # Fetch HTML
        html = fetch_html(url)
        if html is None:
            stats['failed'] += 1
            continue
        
        # Parse page
        try:
            record = parse_scheme_page(html, url)
            new_records.append(record)
            logger.info(f"SUCCESS {url}")
            stats['succeeded'] += 1
        except Exception as e:
            logger.error(f"FAIL {url} - Parse error: {e}")
            stats['failed'] += 1
    
    # Merge with existing records (new records overwrite old ones for same URL)
    all_records = list(existing_records.values())
    
    # Create URL set for existing records
    existing_urls = {r['source_url'] for r in all_records}
    
    # Add new/updated records
    for record in new_records:
        if record['source_url'] in existing_urls:
            # Update existing record
            all_records = [r for r in all_records if r['source_url'] != record['source_url']]
        all_records.append(record)
    
    # Save all records
    if all_records:
        save_records(all_records, output_path)
    
    return stats


if __name__ == "__main__":
    # List of HDFC Mutual Fund Equity Scheme URLs
    scheme_urls = [
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-50-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-50-exchange-traded-fund/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-bse-sensex-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-next-50-exchange-traded-fund/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-next-50-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-100-exchange-traded-fund/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-100-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-midcap-150-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-midcap-150-etf/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-smallcap-250-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-smallcap-250-etf/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-bse-500-index-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-bse-500-etf/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty-bank-exchange-traded-fund/regular",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-nifty50-equal-weight-index-fund/direct",
    ]
    
    # Run scraper
    output_file = "data/raw_scheme_pages.jsonl"
    stats = scrape_scheme_urls(scheme_urls, output_path=output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("SCRAPING SUMMARY")
    print("="*60)
    print(f"URLs attempted: {stats['attempted']}")
    print(f"Successful:     {stats['succeeded']}")
    print(f"Failed:         {stats['failed']}")
    print(f"Output file:    {output_file}")
    print("="*60)

