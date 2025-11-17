"""
Production-grade data extraction script for Mutual Fund FAQ RAG chatbot.
Scrapes factual data from Groww.in pages and stores it in structured format.
"""

import json
import logging
import re
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
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

# Allowed domains
ALLOWED_DOMAINS: Set[str] = {
    'hdfcfund.com',
    'www.hdfcfund.com',
}

# User-Agent to avoid blocking
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
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
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff: wait 2^attempt seconds before retry
            if attempt > 0:
                wait_time = 2 ** (attempt - 1)
                logger.info(f"Retry {attempt}/{max_retries-1} for {url} after {wait_time}s")
                time.sleep(wait_time)
            
            # Fetch with requests
            response = session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                stream=False,
                hooks={'response': redirect_hook},
                verify=False,
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
            
            if any(x in content_type for x in ['image/', 'javascript', 'application/javascript']):
                logger.error(f"FAIL {url} - Content-Type is {content_type}, not HTML")
                return None
            
            if 'html' not in content_type and not content_type.startswith('text/'):
                logger.warning(f"WARNING {url} - Unexpected Content-Type: {content_type}, proceeding anyway")
            
            return response.text
            
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries} for {url}")
                continue
            logger.error(f"FAIL {url} - Request timeout after {max_retries} attempts")
            return None
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Network error on attempt {attempt + 1}/{max_retries} for {url}: {e}")
                continue
            logger.error(f"FAIL {url} - Network error after {max_retries} attempts: {e}")
            return None
            
        except Exception as e:
            logger.error(f"FAIL {url} - Unexpected error: {e}")
            return None
    
    return None


def extract_text_by_label(soup: BeautifulSoup, label: str) -> Optional[str]:
    """
    Extract text value by searching for a label in the page.
    
    Args:
        soup: BeautifulSoup parsed HTML
        label: Label text to search for
        
    Returns:
        Extracted text value or None
    """
    # Try to find the label and get adjacent text
    for element in soup.find_all(text=re.compile(re.escape(label), re.IGNORECASE)):
        if element.parent:
            # Try next sibling
            next_elem = element.parent.find_next_sibling()
            if next_elem:
                text = next_elem.get_text(strip=True)
                if text:
                    return text
            
            # Try parent's next sibling
            parent = element.parent
            if parent and parent.parent:
                next_parent = parent.find_next_sibling()
                if next_parent:
                    text = next_parent.get_text(strip=True)
                    if text:
                        return text
    
    return None


def extract_number_from_text(text: str) -> Optional[float]:
    """
    Extract number from text, handling Indian number format.
    
    Args:
        text: Text containing number
        
    Returns:
        Extracted number as float or None
    """
    if not text:
        return None
    
    # Remove common symbols and text
    clean = re.sub(r'[₹,\s%]', '', text)
    clean = clean.replace('Cr', '').replace('cr', '')
    
    # Extract number
    match = re.search(r'-?\d+\.?\d*', clean)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    
    return None


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
        flags = re.DOTALL
        if not case_sensitive:
            flags |= re.IGNORECASE
        pattern = re.compile(
            re.escape(label) + r'[:.]?\s*(.+?)(?:\n\n|\n\s*[A-Z][a-z]+[:.]|$)',
            flags
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
        flags = re.DOTALL
        if not case_sensitive:
            flags |= re.IGNORECASE
        pattern = re.compile(
            re.escape(label) + r'[:.]?\s*(.+?)(?:\n\n|\n\s*[A-Z][a-z]+[:.]|$)',
            flags
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


def parse_groww_scheme_page(html: str, url: str) -> Dict:
    """
    Parse a Groww mutual fund scheme page and extract structured data.
    
    Args:
        html: Raw HTML content
        url: Source URL
        
    Returns:
        Dictionary with all extracted fund data
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Initialize record with all fields
    record = {
        "scheme_name": None,
        "amc_name": "HDFC Asset Management Company Limited",
        "category": None,
        "plan_type": None,
        "option": None,
        "risk_level": None,
        "fund_rating": None,
        
        "nav_date": None,
        "nav_value": None,
        "fund_size_cr": None,
        
        "min_lumpsum_first_investment": None,
        "min_lumpsum_additional": None,
        "min_sip_amount": None,
        
        "returns_1y_percent": None,
        "returns_3y_percent": None,
        "returns_5y_percent": None,
        "returns_all_percent": None,
        
        "category_avg_1y_percent": None,
        "category_avg_3y_percent": None,
        "category_avg_5y_percent": None,
        "category_avg_all_percent": None,
        
        "rank_within_category_1y": None,
        "rank_within_category_3y": None,
        "rank_within_category_5y": None,
        "rank_within_category_all": None,
        
        "top_5_holdings_weight_percent": None,
        "top_20_holdings_weight_percent": None,
        "pe_ratio": None,
        "pb_ratio": None,
        
        "alpha": None,
        "beta": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        
        "expense_ratio_percent": None,
        "expense_ratio_inclusive_of_gst": True,
        "exit_load_text": None,
        "stamp_duty_text": None,
        "tax_implication_text": None,
        
        "fund_manager_name": None,
        "fund_manager_since": None,
        "fund_manager_tenure_text": None,
        
        "source_url": url,
        "last_scraped_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Extract scheme name from h1 or title
    h1 = soup.find('h1')
    if h1:
        record["scheme_name"] = h1.get_text(strip=True)
    else:
        title = soup.find('title')
        if title:
            record["scheme_name"] = title.get_text(strip=True).strip()
    
    # Extract all visible text for full context
    text_all = soup.get_text(separator=' ', strip=True)
    
    # Extract NAV (will be None if page shows NA on HDFC pages)
    # Skip NAV extraction for HDFC as pages show "NA"
    # nav_match = re.search(r'NAV.*?₹\s*([\d,]+\.\d+)', text_all)
    # if nav_match:
    #     record["nav_value"] = extract_number_from_text(nav_match.group(1))
    
    # Extract Fund Size / AUM
    size_match = re.search(r'(?:Fund Size|AUM|Assets Under Management).*?₹\s*([\d,]+(?:\.\d+)?\s*Cr)', text_all, re.IGNORECASE)
    if size_match:
        record["fund_size_cr"] = extract_number_from_text(size_match.group(1))
    
    # Extract Risk Level (prefer structured section near Riskometer)
    risk_text = find_section_text(soup, ['Riskometer', 'Risk Level', 'Risk'])
    if risk_text:
        m = re.search(r'(Very High|High|Moderately High|Moderate|Low)', risk_text, re.IGNORECASE)
        if m:
            record["risk_level"] = m.group(1)
    else:
        risk_match = re.search(r'(Very High|High|Moderately High|Moderate|Low)', text_all, re.IGNORECASE)
        if risk_match:
            record["risk_level"] = risk_match.group(1)
    
    # Extract Category (fallback to Equity if shown near header)
    category_text = find_section_text(soup, ['Category', 'Fund Category', 'Scheme Category', 'Fund Type', 'Scheme Type'])
    if not category_text:
        cat_tag = soup.find(lambda tag: tag.name in ['span', 'div', 'p'] and 'Equity' in tag.get_text())
        if cat_tag:
            category_text = 'Equity'
    if category_text:
        record["category"] = category_text.strip()
    
    # Extract Returns (1Y, 3Y, 5Y)
    returns_1y = re.search(r'1Y.*?([\d.]+)%', text_all)
    if returns_1y:
        record["returns_1y_percent"] = extract_number_from_text(returns_1y.group(1))
    
    returns_3y = re.search(r'3Y.*?([\d.]+)%', text_all)
    if returns_3y:
        record["returns_3y_percent"] = extract_number_from_text(returns_3y.group(1))
    
    returns_5y = re.search(r'5Y.*?([\d.]+)%', text_all)
    if returns_5y:
        record["returns_5y_percent"] = extract_number_from_text(returns_5y.group(1))
    
    # Extract Returns since inception
    since_inception = re.search(r'Returns\s+since\s+inception\s*([\d.]+)%', text_all, re.IGNORECASE)
    if since_inception:
        record["returns_all_percent"] = extract_number_from_text(since_inception.group(1))
    
    # Extract Expense Ratio / TER (HDFC shows TER value before "Lock in" or "NAV")
    # TER appears as "TER ... 0.67" (single decimal place usually)
    ter_match = re.search(r'TER.*?Disclaimer.*?(0\.\d+|1\.\d+)', text_all, re.IGNORECASE)
    if ter_match:
        record["expense_ratio_percent"] = extract_number_from_text(ter_match.group(1))
    else:
        # Fallback to generic expense ratio pattern
        expense_match = re.search(r'(?:Expense Ratio|Total Expense Ratio).*?([\d.]+)%?', text_all, re.IGNORECASE)
        if expense_match:
            record["expense_ratio_percent"] = extract_number_from_text(expense_match.group(1))
    
    # Extract Exit Load from Downloads/Exit Load section
    exit_match = re.search(r'Exit Load\s+(.*?)(?:Product Labelling|Benchmark Riskometer|###|FAQs)', text_all, re.IGNORECASE | re.DOTALL)
    if exit_match:
        exit_text = exit_match.group(1).strip()
        # Clean up and limit to reasonable length
        exit_text = re.sub(r'\s+', ' ', exit_text)
        if len(exit_text) > 500:
            exit_text = exit_text[:500]
        record["exit_load_text"] = exit_text
    
    # Extract Minimum SIP (handle "Min SIP")
    sip_match = re.search(r'(?:Minimum SIP|Min SIP).*?₹\s*([\d,]+)', text_all, re.IGNORECASE)
    if sip_match:
        record["min_sip_amount"] = extract_number_from_text(sip_match.group(1))
    
    # Extract Minimum Lumpsum
    lump_match = re.search(r'Minimum (?:Lumpsum|Investment).*?₹\s*([\d,]+)', text_all, re.IGNORECASE)
    if lump_match:
        record["min_lumpsum_first_investment"] = extract_number_from_text(lump_match.group(1))
    
    # Extract Fund Managers from "Fund Managers" section
    fm_match = re.search(r'Fund Managers?\s+((?:[A-Z][a-z]+\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?.*?)+?)(?:Top 10 Holdings|Downloads|##)', text_all, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1).strip()
        # Extract just names (Ms./Mr. followed by name)
        names = re.findall(r'(?:Ms\.|Mr\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', fm_text)
        if names:
            record["fund_manager_name"] = ', '.join(names[:5])
        else:
            # Fallback: extract capitalized names
            names = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', fm_text)
            if names:
                record["fund_manager_name"] = ', '.join(names[:5])
    
    # Extract Ratios
    pe_match = re.search(r'P/E Ratio.*?([\d.]+)', text_all, re.IGNORECASE)
    if pe_match:
        record["pe_ratio"] = extract_number_from_text(pe_match.group(1))
    
    pb_match = re.search(r'P/B Ratio.*?([\d.]+)', text_all, re.IGNORECASE)
    if pb_match:
        record["pb_ratio"] = extract_number_from_text(pb_match.group(1))
    
    alpha_match = re.search(r'Alpha.*?([\d.]+)', text_all, re.IGNORECASE)
    if alpha_match:
        record["alpha"] = extract_number_from_text(alpha_match.group(1))
    
    beta_match = re.search(r'Beta.*?([\d.]+)', text_all, re.IGNORECASE)
    if beta_match:
        record["beta"] = extract_number_from_text(beta_match.group(1))
    
    sharpe_match = re.search(r'Sharpe.*?([\d.]+)', text_all, re.IGNORECASE)
    if sharpe_match:
        record["sharpe_ratio"] = extract_number_from_text(sharpe_match.group(1))
    
    # Detect plan type and option from URL and name
    if 'direct' in url.lower() or 'direct' in str(record["scheme_name"]).lower():
        record["plan_type"] = "Direct"
    
    if 'growth' in url.lower() or 'growth' in str(record["scheme_name"]).lower():
        record["option"] = "Growth"
    
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
            record = parse_groww_scheme_page(html, url)
            new_records.append(record)
            logger.info(
                f"Extracted | scheme={record.get('scheme_name')} | AUM={record.get('fund_size_cr')} Cr | "
                f"TER={record.get('expense_ratio_percent')}% | SIP={record.get('min_sip_amount')} | "
                f"ExitLoad={'Y' if record.get('exit_load_text') else 'N'} | Risk={record.get('risk_level')} | "
                f"Managers={record.get('fund_manager_name')} | SinceInc={record.get('returns_all_percent')}%"
            )
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
    # List of HDFC AMC scheme URLs
    scheme_urls = [
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct",
        "https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct",
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

