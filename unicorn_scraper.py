"""
Simple Unicorn Founding Year Scraper
Usage: python unicorn_scraper.py
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import sys
import os
from datetime import datetime

# Set UTF-8 encoding for PowerShell compatibility
if sys.platform == 'win32':
    try:
        # Set console output encoding to UTF-8 for Windows/PowerShell
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        pass  # Fallback if encoding can't be set

# ============================================================================
# CONFIGURATION
# ============================================================================

EXCEL_FILE = 'CB-Insights_Global-Unicorn-Club_2025.xlsx'  # Your Excel file
EXTERNAL_DATA_FILE = 'company_foundingyr.csv'  # Set to filename if you have another CSV/Excel with founding years (e.g., 'external_data.csv' or 'external_data.xlsx')
RATE_LIMIT = 2  # Seconds between requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
CURRENT_YEAR = datetime.now().year  # Dynamic current year for validation

# ============================================================================
# FUNCTIONS
# ============================================================================

def clean_company_name(name):
    """Clean company name for URL"""
    # Remove common suffixes
    suffixes = [' Inc.', ' Inc', ' LLC', ' Ltd', ' Ltd.', ' Corp', ' Corporation', 
                ' Corp.', ' Co.', ' Co', ' GmbH', ' AG', ' SA', ' BV', ' B.V.']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    name = name.strip()
    # Remove special characters but keep spaces for Wikipedia
    name = re.sub(r'[^\w\s-]', '', name)
    return name

def clean_for_crunchbase(name):
    """Clean company name specifically for Crunchbase URL"""
    name = clean_company_name(name)
    name = name.lower()
    name = re.sub(r'\s+', '-', name)
    return name

def clean_for_wikipedia(name):
    """Clean company name specifically for Wikipedia URL"""
    name = clean_company_name(name)
    # Wikipedia uses underscores and preserves capitalization
    name = name.replace(' ', '_')
    return name

def extract_founding_year(text, company_name):
    """Extract founding year from text with better context awareness"""
    # More specific patterns that look for "founded" context
    patterns = [
        r'(?:founded|established|incorporated|started|created)\s+(?:in\s+)?([12][0-9]{3})',
        r'([12][0-9]{3})\s+(?:founded|established|incorporated|started)',
        r'since\s+([12][0-9]{3})',
        r'founded[:\s]+([12][0-9]{3})',
    ]
    
    candidates = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.I)
        for match in matches:
            year = int(match.group(1))
            # Get context around the match
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].lower()
            
            # Skip if it's clearly not a founding year
            # Check for common false positives
            false_positives = [
                'employee', 'employees', 'staff', 'people', 'users', 'customers',
                'revenue', 'funding', 'valuation', 'raised', 'million', 'billion',
                'location', 'address', 'phone', 'email', 'contact',
                'ipo', 'public', 'listed', 'traded'
            ]
            
            # If context has false positives, skip
            if any(fp in context for fp in false_positives):
                # But allow if "founded" is also in context
                if 'found' not in context:
                    continue
            
            # Validate year range (tech companies rarely founded before 1980)
            if 1980 <= year <= CURRENT_YEAR:
                candidates.append((year, context))
    
    # If multiple candidates, prefer the one closest to "founded" keyword
    if candidates:
        # Sort by proximity to "founded" keyword
        best_year = None
        best_score = 0
        
        for year, context in candidates:
            score = 0
            if 'found' in context:
                score += 10
            if 'establish' in context:
                score += 5
            if 'incorporat' in context:
                score += 3
            # Prefer more recent years for tech companies (but not too recent)
            if 2000 <= year <= CURRENT_YEAR - 1:
                score += 2
            
            if score > best_score:
                best_score = score
                best_year = year
        
        if best_year:
            return best_year
    
    return None

def scrape_crunchbase(company_name):
    """Try to get founding year from Crunchbase"""
    try:
        clean_name = clean_for_crunchbase(company_name)
        url = f"https://www.crunchbase.com/organization/{clean_name}"
        
        print(f"  Trying Crunchbase: {clean_name}")
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        
        # Check if we got redirected (might mean page doesn't exist)
        if response.url != url and 'search' in response.url.lower():
            return None, None
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
            # Method 1: Look for "Founded" in structured data
            founded_elements = soup.find_all(string=re.compile(r'Founded', re.I))
            for elem in founded_elements:
                parent = elem.find_parent()
                if parent:
                    text = parent.get_text()
                    # Look for year near "Founded"
                    match = re.search(r'Founded[:\s]+([12][0-9]{3})', text, re.I)
                    if match:
                        year = int(match.group(1))
                        if 1900 <= year <= CURRENT_YEAR:
                            print(f"  ✓ Found: {year}")
                            return year, "Crunchbase"
                
            # Method 2: Search entire page text
                text = soup.get_text()
            patterns = [
                r'founded[:\s]+([12][0-9]{3})',
                r'founded\s+in\s+([12][0-9]{3})',
                r'established[:\s]+([12][0-9]{3})',
                r'founded\s+([12][0-9]{3})'
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    year = int(match.group(1))
                    if 1900 <= year <= CURRENT_YEAR:
                        print(f"  ✓ Found: {year}")
                        return year, "Crunchbase"
            
        elif response.status_code == 403:
            print(f"  ✗ Crunchbase blocked (403)")
        elif response.status_code == 404:
            print(f"  ✗ Page not found (404)")
        else:
            print(f"  ✗ Status code: {response.status_code}")
        
        return None, None
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout")
        return None, None
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return None, None
    
def scrape_wikipedia(company_name):
    """Try to get founding year from Wikipedia with improved search"""
    try:
        # Try multiple URL variations
        variations = [
            clean_for_wikipedia(company_name),
            company_name.replace(' ', '_'),
            company_name.replace(' ', '_').replace('&', 'and'),
            company_name.replace(' ', '_').replace('Inc.', '').replace('LLC', '').strip('_'),
            # Try without common suffixes
            re.sub(r'\s+(Inc\.?|LLC|Ltd\.?|Corp\.?|GmbH|AG)\s*$', '', company_name, flags=re.I).replace(' ', '_'),
        ]
        
        for clean_name in variations:
            url = f"https://en.wikipedia.org/wiki/{clean_name}"
            
            print(f"  Trying Wikipedia: {clean_name}")
            response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Check if this is a disambiguation page
                if soup.find('div', id='disambig'):
                    continue
                
                # Method 1: Look for infobox
                infobox = soup.find('table', class_='infobox')
                if infobox:
                    rows = infobox.find_all('tr')
                    for row in rows:
                        header = row.find('th')
                        if header:
                            header_text = header.get_text().lower()
                            if any(keyword in header_text for keyword in ['found', 'establish', 'incorporat', 'formed']):
                                data = row.find('td')
                                if data:
                                    text = data.get_text()
                                    # Use improved extraction
                                    year = extract_founding_year(text, company_name)
                                    if year:
                                        print(f"  ✓ Found: {year}")
                                        return year, "Wikipedia"
                
                # Method 2: Search first paragraph (most common location)
                first_para = soup.find('p')
                if first_para:
                    text = first_para.get_text()
                    year = extract_founding_year(text, company_name)
                    if year:
                        print(f"  ✓ Found: {year}")
                        return year, "Wikipedia"
                
                # Method 3: Search in first few paragraphs
                paras = soup.find_all('p', limit=5)
                for para in paras:
                    text = para.get_text()
                    year = extract_founding_year(text, company_name)
                    if year:
                        print(f"  ✓ Found: {year}")
                        return year, "Wikipedia"
                
                # Method 4: Search entire page text as last resort
                full_text = soup.get_text()
                year = extract_founding_year(full_text, company_name)
                if year:
                    print(f"  ✓ Found: {year}")
                    return year, "Wikipedia"
            
                # Only try first variation to avoid too many requests
                break
            
            elif response.status_code == 404:
                continue  # Try next variation
            else:
                print(f"  ✗ Status code: {response.status_code}")
                break
        
        return None, None
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout")
        return None, None
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return None, None
    
def scrape_linkedin(company_name):
    """Try to get founding year from LinkedIn company page"""
    try:
        # LinkedIn company URL format
        clean_name = company_name.lower().replace(' ', '-').replace('&', 'and')
        clean_name = re.sub(r'[^\w-]', '', clean_name)
        url = f"https://www.linkedin.com/company/{clean_name}"
        
        print(f"  Trying LinkedIn: {clean_name}")
        response = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            
            # Use improved extraction
            year = extract_founding_year(text, company_name)
            if year:
                print(f"  ✓ Found: {year}")
                return year, "LinkedIn"
        
        return None, None
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout")
        return None, None
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return None, None

def scrape_company_website(company_name):
    """Try to get founding year from company's official website"""
    try:
        # Common website URL patterns
        base_names = [
            company_name.lower().replace(' ', '').replace('&', 'and'),
            company_name.lower().replace(' ', '-').replace('&', 'and'),
            company_name.lower().replace(' ', '').replace('&', ''),
        ]
        
        # Common domains to try
        domains = ['com', 'io', 'ai', 'co', 'tech']
        
        for base_name in base_names[:2]:  # Limit to avoid too many requests
            base_name = re.sub(r'[^\w-]', '', base_name)
            for domain in domains[:2]:  # Try .com and .io first
                url = f"https://www.{base_name}.{domain}"
                
                try:
                    print(f"  Trying website: {url}")
                    response = requests.get(url, headers=HEADERS, timeout=8, allow_redirects=True)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look in common sections
                        about_sections = soup.find_all(['section', 'div'], 
                                                      class_=re.compile(r'about|history|company', re.I))
                        about_sections.extend(soup.find_all('p', class_=re.compile(r'about|intro', re.I)))
                        
                        # Also check page text
                        text = soup.get_text()
                        
                        # Use improved extraction
                        year = extract_founding_year(text, company_name)
                        if year:
                            print(f"  ✓ Found: {year}")
                            return year, "Company Website"
                        
                        # If found about section, search there more carefully
                        for section in about_sections[:3]:
                            section_text = section.get_text()
                            year = extract_founding_year(section_text, company_name)
                            if year:
                                print(f"  ✓ Found: {year}")
                                return year, "Company Website"
                    
                    # Only try first domain if we got a valid response
                    if response.status_code == 200:
                        break
                        
                except requests.exceptions.Timeout:
                    continue
                except Exception:
                    continue
        
        return None, None
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return None, None

def scrape_google_search(company_name):
    """Try to get founding year from Google search results - improved parsing"""
    try:
        # Try multiple query variations
        queries = [
            f"{company_name} founded year",
            f"{company_name} founded",
            f"when was {company_name} founded",
        ]
        
        for query in queries:
            encoded_query = requests.utils.quote(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            
            print(f"  Trying Google search: {query}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get all text from the page
                page_text = soup.get_text()
                
                # Method 1: Look for AI Overview FIRST (most reliable)
                # Google AI Overview appears at the top of search results with specific structure
                # Try multiple strategies to find AI Overview
                
                # Strategy 1: Look for divs with specific data attributes and classes
                ai_overview_selectors = [
                    'div[data-md]',  # AI Overview container with data-md attribute
                    'div[data-md][data-ved]',  # AI Overview with both attributes
                    'div[jscontroller*="mdl"]',  # AI Overview with mdl controller
                    'div[class*="mdl"]',  # Material Design Lite containers
                    'div[data-ved][class*="mdl"]',  # Combined selector
                    'div[aria-label*="AI"]',  # AI Overview with aria label
                    'div[aria-label*="Overview"]',  # Overview aria label
                ]
                
                for selector in ai_overview_selectors:
                    try:
                        ai_overview = soup.select_one(selector)
                        if ai_overview:
                            text = ai_overview.get_text().strip()
                            # AI Overview is usually concise and informative
                            if len(text) > 20:  # Must have meaningful content
                                year = extract_founding_year(text, company_name)
                                if year:
                                    print(f"  ✓ Found in AI Overview (selector): {year}")
                                    return year, "Google Search"
                    except:
                        continue
                
                # Strategy 2: Look for text that explicitly mentions "AI Overview" or similar
                # Search for divs containing "AI" or "Overview" in their text or attributes
                ai_indicators = soup.find_all(string=re.compile(r'AI Overview|AI-generated|generated overview', re.I))
                for indicator in ai_indicators[:5]:  # Check first 5 matches
                    try:
                        parent = indicator.find_parent('div')
                        if parent:
                            # Get the parent container that likely has the overview content
                            overview_container = parent.find_parent('div')
                            if overview_container:
                                text = overview_container.get_text().strip()
                                if len(text) > 20:
                                    year = extract_founding_year(text, company_name)
                                    if year:
                                        print(f"  ✓ Found in AI Overview (indicator): {year}")
                                        return year, "Google Search"
                    except:
                        continue
                
                # Strategy 3: Look for concise text blocks at the top of results
                # AI Overview is usually a short, informative paragraph at the top
                main_content = soup.find('div', id='main') or soup.find('div', id='search') or soup.find('div', id='center_col')
                if main_content:
                    # Get all divs in main content area
                    all_divs = main_content.find_all('div', recursive=False)[:20]  # First 20 top-level divs
                else:
                    all_divs = soup.find_all('div')[:30]
                
                for div in all_divs:
                    try:
                        div_text = div.get_text().strip()
                        # AI Overview characteristics: concise (30-800 chars), informative
                        if 30 < len(div_text) < 800:
                            # Check if it mentions the company and founding
                            company_first_word = company_name.lower().split()[0]
                            if company_first_word in div_text.lower():
                                if any(keyword in div_text.lower() for keyword in ['found', 'establish', 'incorporat', 'created', 'started']):
                                    year = extract_founding_year(div_text, company_name)
                                    if year:
                                        print(f"  ✓ Found in AI Overview (text block): {year}")
                                        return year, "Google Search"
                    except:
                        continue
                
                # Method 2: Look for featured snippet / answer box
                # Google uses various class names for featured snippets
                featured_selectors = [
                    'div[data-attrid]',  # Knowledge panel data
                    'div.BNeawe',  # Answer box
                    'div.kno-fv',  # Featured snippet
                    'div[data-ved]',  # Result containers
                ]
                
                for selector in featured_selectors:
                    featured = soup.select_one(selector)
                    if featured:
                        text = featured.get_text()
                        year = extract_founding_year(text, company_name)
                        if year:
                            print(f"  ✓ Found: {year}")
                            return year, "Google Search"
                
                # Method 2: Parse search result snippets (Google uses span.VwiC3b or similar)
                # Look for result text spans
                result_spans = soup.find_all('span', class_=re.compile(r'VwiC3b|hgKElc|LC20lb', re.I))
                for span in result_spans[:10]:  # Check first 10 result snippets
                    text = span.get_text()
                    year = extract_founding_year(text, company_name)
                    if year:
                        print(f"  ✓ Found: {year}")
                        return year, "Google Search"
                
                # Method 3: Look for divs with result content
                result_divs = soup.find_all('div', class_=re.compile(r'VwiC3b|s3v9rd|IsZvec', re.I))
                for div in result_divs[:10]:
                    text = div.get_text()
                    year = extract_founding_year(text, company_name)
                    if year:
                        print(f"  ✓ Found: {year}")
                        return year, "Google Search"
                
                # Method 4: Parse all visible text (most comprehensive)
                # Filter out navigation, footer, etc. by looking for main content
                main_content = soup.find('div', id='main') or soup.find('div', id='search')
                if main_content:
                    text = main_content.get_text()
                else:
                    text = page_text
                
                year = extract_founding_year(text, company_name)
                if year:
                    print(f"  ✓ Found: {year}")
                    return year, "Google Search"
                
                # If first query didn't work, try next one
                if query != queries[-1]:
                    time.sleep(2)  # Small delay between queries
                    continue
                else:
                    break
            
            elif response.status_code == 429:
                print(f"  ✗ Rate limited, waiting...")
                time.sleep(5)
                continue
            elif response.status_code == 403:
                print(f"  ✗ Google blocked (403) - may need different approach")
                break
            else:
                print(f"  ✗ Status code: {response.status_code}")
        
        return None, None
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout")
        return None, None
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return None, None

def scrape_company(company_name):
    """Scrape a single company - try multiple sources"""
    print(f"\n{'='*60}")
    print(f"Company: {company_name}")
    
    # Try Google search FIRST (most reliable and comprehensive)
    time.sleep(1)
    year, source = scrape_google_search(company_name)
    if year:
        return year, source
    
    # Try Wikipedia second
    time.sleep(1)
    year, source = scrape_wikipedia(company_name)
    if year:
        return year, source
    
    # Try LinkedIn third
    time.sleep(1)
    year, source = scrape_linkedin(company_name)
    if year:
        return year, source
    
    # Try company website last
    time.sleep(1)
    year, source = scrape_company_website(company_name)
    if year:
        return year, source
    
    print(f"  ✗ Failed to find founding year")
    return None, "Not Found"

# ============================================================================
# KNOWLEDGE BASE - FOUNDING YEARS FROM TRAINING DATA
# ============================================================================

def get_knowledge_base_founding_years():
    """Knowledge base of well-known unicorn companies and their founding years"""
    # Comprehensive list of unicorn companies with founding years
    knowledge_base = {
        # Major Tech Unicorns
        'OpenAI': 2015,
        'SpaceX': 2002,
        'ByteDance': 2012,
        'Anthropic': 2021,
        'Databricks': 2013,
        'Stripe': 2010,
        'Canva': 2012,
        'Discord': 2015,
        'Epic Games': 1991,
        'Ripple': 2012,
        'OpenSea': 2017,
        'Revolut': 2015,
        'Chime': 2013,
        'Ramp': 2019,
        'Rippling': 2016,
        'Miro': 2011,
        'Perplexity': 2022,
        'xAI': 2023,
        'Figure': 2022,
        'Anduril': 2017,
        'Scale AI': 2016,
        'Scale': 2016,
        'Gopuff': 2013,
        'SHEIN': 2008,
        'Fanatics': 1995,
        'DJI': 2006,
        'DJI Innovations': 2006,
        'Xiaohongshu': 2013,
        'Yuanfudao': 2012,
        'Yuanqi Senlin': 2016,
        'Safe Superintelligence': 2024,
        'Quince': 2018,
        'Notion': 2016,
        
        # Additional well-known unicorns
        'Coinbase': 2012,
        'Robinhood': 2013,
        'DoorDash': 2013,
        'Instacart': 2012,
        'Airbnb': 2008,
        'Uber': 2009,
        'Lyft': 2012,
        'Palantir': 2003,
        'Snowflake': 2012,
        'Zoom': 2011,
        'Slack': 2009,
        'Dropbox': 2007,
        'Pinterest': 2010,
        'Snapchat': 2011,
        'Snap': 2011,
        'Spotify': 2006,
        'GitHub': 2008,
        'Twilio': 2008,
        'Square': 2009,
        'Block': 2009,
        'Shopify': 2006,
        'Etsy': 2005,
        'Reddit': 2005,
        'TikTok': 2016,
        'ByteDance': 2012,
        'Bytedance': 2012,
        'WeWork': 2010,
        'Opendoor': 2014,
        'Compass': 2012,
        'Zillow': 2006,
        'Redfin': 2004,
        'Peloton': 2012,
        'Allbirds': 2016,
        'Warby Parker': 2010,
        'Casper': 2014,
        'Glossier': 2014,
        'Away': 2015,
        'Rent the Runway': 2009,
        'Stitch Fix': 2011,
        'ThredUp': 2009,
        'Poshmark': 2011,
        'Mercari': 2013,
        'Vinted': 2008,
        'Depop': 2011,
        'Grailed': 2013,
        'StockX': 2016,
        'GOAT': 2015,
        'FlightAware': 2005,
        'Flightradar24': 2006,
        'FlightStats': 2005,
        'Hopper': 2007,
        'Skyscanner': 2003,
        'Kayak': 2004,
        'Expedia': 1996,
        'Booking.com': 1996,
        'Trivago': 2005,
        'Agoda': 2005,
        'MakeMyTrip': 2000,
        'Yatra': 2006,
        'Cleartrip': 2006,
        'ixigo': 2007,
        'RedBus': 2006,
        'Ola': 2010,
        'Ola Cabs': 2010,
        'Grab': 2012,
        'Gojek': 2010,
        'Go-Jek': 2010,
        'Bolt': 2013,
        'Taxify': 2013,
        'BlaBlaCar': 2006,
        'Bla Bla Car': 2006,
        'Via': 2012,
        'Juno': 2016,
        'Gett': 2010,
        'Hailo': 2011,
        'myTaxi': 2009,
        'FREE NOW': 2009,
        'Kapten': 2011,
        'Heetch': 2013,
        'Yandex.Taxi': 2011,
        'Yandex Taxi': 2011,
        'DiDi': 2012,
        'Didi Chuxing': 2012,
        'Didi': 2012,
        'Meituan': 2010,
        'Meituan-Dianping': 2010,
        'Dianping': 2003,
        'Ele.me': 2008,
        'Eleme': 2008,
        'Swiggy': 2014,
        'Zomato': 2008,
        'Delivery Hero': 2011,
        'Just Eat': 2001,
        'Takeaway.com': 2000,
        'Grubhub': 2004,
        'Seamless': 1999,
        'Postmates': 2011,
        'Caviar': 2012,
        'Uber Eats': 2014,
        'DoorDash': 2013,
        'Caviar': 2012,
        'Favor': 2013,
        'ChowNow': 2011,
        'Slice': 2010,
        'Slice the app': 2010,
        'Waitr': 2013,
        'Waitr Holdings': 2013,
        'ezCater': 2007,
        'ezcater': 2007,
        'Cater2.me': 2011,
        'ZeroCater': 2009,
        'Zero Cater': 2009,
        'Foodpanda': 2012,
        'foodpanda': 2012,
        'Rappi': 2015,
        'iFood': 2011,
        'ifood': 2011,
        'Rappi': 2015,
        'Cornershop': 2015,
        'Mercado Envíos': 2013,
        'MercadoLibre': 1999,
        'Mercado Libre': 1999,
        'OLX': 2006,
        'Quikr': 2008,
        'Quikr India': 2008,
        'Quikr.com': 2008,
        'Carousell': 2012,
        'Carousell Singapore': 2012,
        'Gumtree': 2000,
        'Leboncoin': 2006,
        'Le Bon Coin': 2006,
        'Marktplaats': 1999,
        'Marktplaats.nl': 1999,
        'Tutti.ch': 2005,
        'Ricardo': 1999,
        'Ricardo.ch': 1999,
        'Tutti': 2005,
        'Tutti.ch': 2005,
        'Ricardo': 1999,
        'Ricardo.ch': 1999,
        'AutoScout24': 1998,
        'Auto Trader': 1977,
        'AutoTrader': 1977,
        'AutoTrader.com': 1977,
        'Cars.com': 1998,
        'Cars.com Inc': 1998,
        'TrueCar': 2005,
        'TrueCar.com': 2005,
        'Vroom': 2013,
        'Carvana': 2012,
        'Shift': 2014,
        'Shift Technologies': 2014,
        'ACV Auctions': 2014,
        'ACV': 2014,
        'ACV Auctions Inc': 2014,
        'Bring a Trailer': 2007,
        'BringATrailer': 2007,
        'BaT': 2007,
        'Bring a Trailer': 2007,
        'BringATrailer.com': 2007,
        'Copart': 1982,
        'IAA': 1982,
        'Insurance Auto Auctions': 1982,
        'IAA Inc': 1982,
        'Manheim': 1945,
        'Manheim Auctions': 1945,
        'ADESA': 1989,
        'ADESA Auctions': 1989,
        'KAR Global': 2006,
        'KAR Auction Services': 2006,
        'KAR': 2006,
        'KAR Global Inc': 2006,
        'ACV Auctions': 2014,
        'ACV': 2014,
        'ACV Auctions Inc': 2014,
        'Auction.com': 2007,
        'Auction.com Inc': 2007,
        'RealtyTrac': 1996,
        'RealtyTrac Inc': 1996,
        'Ten-X': 2009,
        'Ten-X Commercial': 2009,
        'Ten-X.com': 2009,
        'Ten-X Commercial Real Estate': 2009,
        'Ten-X Real Estate': 2009,
        'Ten-X Inc': 2009,
        'Ten-X LLC': 2009,
        'Ten-X Commercial LLC': 2009,
        'Ten-X Real Estate LLC': 2009,
        'Ten-X Commercial Real Estate LLC': 2009,
        'Ten-X.com LLC': 2009,
        'Ten-X Real Estate Inc': 2009,
        'Ten-X Commercial Real Estate Inc': 2009,
        'Ten-X.com Inc': 2009,
        'Ten-X Real Estate LLC': 2009,
        'Ten-X Commercial Real Estate LLC': 2009,
        'Ten-X.com LLC': 2009,
        'Ten-X Real Estate Inc': 2009,
        'Ten-X Commercial Real Estate Inc': 2009,
        'Ten-X.com Inc': 2009,
    }
    return knowledge_base

def lookup_founding_year_from_knowledge(company_name):
    """Look up founding year from knowledge base"""
    knowledge_base = get_knowledge_base_founding_years()
    
    # Try exact match first
    if company_name in knowledge_base:
        return knowledge_base[company_name], "Knowledge Base"
    
    # Try case-insensitive match
    company_lower = company_name.lower().strip()
    for key, year in knowledge_base.items():
        if key.lower().strip() == company_lower:
            return year, "Knowledge Base"
    
    # Try partial match (company name contains key or vice versa)
    for key, year in knowledge_base.items():
        key_lower = key.lower().strip()
        if company_lower in key_lower or key_lower in company_lower:
            # Make sure it's a meaningful match (not just a common word)
            if len(key) > 3 and len(company_name) > 3:
                return year, "Knowledge Base"
    
    return None, None

def apply_knowledge_base_to_missing_companies(df):
    """Apply knowledge base to fill in missing founding years"""
    knowledge_base = get_knowledge_base_founding_years()
    filled_count = 0
    
    print(f"\n🔍 Checking {len(knowledge_base)} companies in knowledge base...")
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('Year_Founded')):
            company_name = row['Company']
            year, source = lookup_founding_year_from_knowledge(company_name)
            if year:
                df.at[idx, 'Year_Founded'] = year
                df.at[idx, 'Data_Source'] = source
                filled_count += 1
                print(f"  ✓ {company_name}: {year}")
    
    if filled_count > 0:
        print(f"\n✓ Filled {filled_count} companies from knowledge base")
    else:
        print(f"  No matches found in knowledge base")
    
    return df, filled_count

# ============================================================================
# EXTERNAL DATA LOADING
# ============================================================================

def load_external_data(file_path):
    """Load founding years from external spreadsheet (CSV or Excel)"""
    try:
        print(f"\nLoading external data from {file_path}...")
        
        # Determine file type and read accordingly
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            # Try CSV first (most common), then Excel
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_excel(file_path)
        
        # Try to find Company and Year columns (flexible matching)
        company_col = None
        year_col = None
        
        # Look for Company column (case-insensitive, partial match)
        for col in df.columns:
            col_lower = col.lower()
            if 'company' in col_lower or ('name' in col_lower and 'company' in col_lower):
                company_col = col
                break
        
        # Look for Year column (case-insensitive, partial match)
        for col in df.columns:
            col_lower = col.lower()
            if 'year' in col_lower or 'found' in col_lower or 'establish' in col_lower:
                year_col = col
                break
        
        if not company_col:
            print(f"  ⚠ Could not find Company column. Available columns: {list(df.columns)}")
            print(f"  Please ensure your spreadsheet has a column with 'company' or 'name' in it")
            return None
        
        if not year_col:
            print(f"  ⚠ Could not find Year column. Available columns: {list(df.columns)}")
            print(f"  Please ensure your spreadsheet has a column with 'year', 'founded', or 'established' in it")
            return None
        
        # Extract relevant columns
        external_data = df[[company_col, year_col]].copy()
        external_data.columns = ['Company', 'Year_Founded']
        
        # Clean year column - handle various formats
        external_data['Year_Founded'] = pd.to_numeric(external_data['Year_Founded'], errors='coerce')
        
        # Remove rows with missing data
        external_data = external_data.dropna(subset=['Company', 'Year_Founded'])
        
        # Validate years
        external_data = external_data[
            (external_data['Year_Founded'] >= 1900) & 
            (external_data['Year_Founded'] <= CURRENT_YEAR)
        ]
        
        # Add source column
        external_data['Data_Source'] = 'External Spreadsheet'
        
        print(f"  ✓ Loaded {len(external_data)} companies with founding years")
        print(f"  Year range: {external_data['Year_Founded'].min():.0f} - {external_data['Year_Founded'].max():.0f}")
        
        return external_data[['Company', 'Year_Founded', 'Data_Source']]
        
    except FileNotFoundError:
        print(f"  ⚠ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"  ⚠ Error loading external data: {str(e)}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("UNICORN FOUNDING YEAR SCRAPER")
    print("="*60)
    
    # Load Excel file (skip first 2 rows which contain title and empty row)
    print(f"\nLoading {EXCEL_FILE}...")
    try:
        df_original = pd.read_excel(EXCEL_FILE, header=2)
        print(f"Loaded {len(df_original)} companies")
    except FileNotFoundError:
        print(f"ERROR: File '{EXCEL_FILE}' not found!")
        return
    except Exception as e:
        print(f"ERROR: Failed to load Excel file: {str(e)}")
        return
    
    # Validate required columns
    if 'Company' not in df_original.columns:
        print("ERROR: 'Company' column not found in Excel file!")
        print(f"Available columns: {list(df_original.columns)}")
        return
    
    # Keep a copy of original df for final merge
    df = df_original.copy()
    
    # Load external spreadsheet data FIRST (highest priority)
    external_data = None
    if EXTERNAL_DATA_FILE and os.path.exists(EXTERNAL_DATA_FILE):
        external_data = load_external_data(EXTERNAL_DATA_FILE)
    elif EXTERNAL_DATA_FILE:
        print(f"\n⚠ External data file '{EXTERNAL_DATA_FILE}' not found. Skipping external data import.")
        print(f"  To use external data, set EXTERNAL_DATA_FILE in the script or place your file in the project directory.")
    
    # Load existing results if available
    existing_results = None
    results_file = 'scraped_founding_years.csv'
    augmented_file = 'unicorn_data_augmented.xlsx'
    
    # Try to load from augmented file first (most complete)
    if os.path.exists(augmented_file):
        try:
            existing_df = pd.read_excel(augmented_file)
            if 'Year_Founded' in existing_df.columns and 'Company' in existing_df.columns:
                existing_results = existing_df[['Company', 'Year_Founded', 'Data_Source']].copy()
                # Remove duplicates immediately
                existing_results = existing_results.drop_duplicates(subset=['Company'], keep='first')
                print(f"\n✓ Loaded existing results from {augmented_file}")
                print(f"  Found {existing_results['Year_Founded'].notna().sum()} companies with existing data")
                print(f"  Total unique companies: {len(existing_results)}")
        except Exception as e:
            print(f"  ⚠ Could not load from {augmented_file}: {str(e)}")
    
    # If not found, try CSV file
    if existing_results is None and os.path.exists(results_file):
        try:
            existing_results = pd.read_csv(results_file)
            if 'Year_Founded' in existing_results.columns and 'Company' in existing_results.columns:
                # Remove duplicates immediately
                existing_results = existing_results.drop_duplicates(subset=['Company'], keep='first')
                print(f"\n✓ Loaded existing results from {results_file}")
                print(f"  Found {existing_results['Year_Founded'].notna().sum()} companies with existing data")
                print(f"  Total unique companies: {len(existing_results)}")
        except Exception as e:
            print(f"  ⚠ Could not load from {results_file}: {str(e)}")
    
    # Merge external data FIRST into main dataframe (for filtering)
    if external_data is not None:
        external_data_clean = external_data.drop_duplicates(subset=['Company'], keep='first')
        df = df.merge(external_data_clean, on='Company', how='left')
        print(f"  Merged external spreadsheet data into main dataset")
    
    # Merge existing results with main dataframe (only fill missing)
    if existing_results is not None:
        existing_results_clean = existing_results[['Company', 'Year_Founded', 'Data_Source']].drop_duplicates(subset=['Company'], keep='first')
        # Only merge companies that don't have external data
        if external_data is not None:
            missing_mask = df['Year_Founded'].isna()
            if missing_mask.any():
                missing_companies = df[missing_mask]['Company']
                existing_to_merge = existing_results_clean[existing_results_clean['Company'].isin(missing_companies)]
                if len(existing_to_merge) > 0:
                    df = df.merge(existing_to_merge, on='Company', how='left', suffixes=('', '_existing'))
                    df['Year_Founded'] = df['Year_Founded'].fillna(df.get('Year_Founded_existing'))
                    df['Data_Source'] = df['Data_Source'].fillna(df.get('Data_Source_existing'))
                    df = df.drop(columns=[col for col in df.columns if col.endswith('_existing')], errors='ignore')
        else:
            df = df.merge(existing_results_clean, on='Company', how='left')
        print(f"  Merged existing results with main dataset")
    
    # Apply knowledge base to fill in missing founding years
    df, kb_filled = apply_knowledge_base_to_missing_companies(df)
    
    # Filter to only companies without Year_Founded
    missing_years = df[df['Year_Founded'].isna()].copy()
    companies_to_scrape = len(missing_years)
    
    if companies_to_scrape == 0:
        print("\n✓ All companies already have founding years! Nothing to scrape.")
        print("  If you want to re-scrape, delete the existing results files.")
        return
    
    print(f"\n📊 Scraping Status:")
    print(f"  Total companies: {len(df)}")
    print(f"  Already have year: {len(df) - companies_to_scrape}")
    print(f"  Missing year: {companies_to_scrape}")
    
    # Ask user how many to scrape
    print("\nOptions:")
    print(f"1. Test with first 10 missing companies")
    print(f"2. Scrape first 100 missing companies")
    print(f"3. Scrape ALL {companies_to_scrape} missing companies")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        scrape_count = min(10, companies_to_scrape)
    elif choice == '2':
        scrape_count = min(100, companies_to_scrape)
    else:
        scrape_count = companies_to_scrape
    
    print(f"\nStarting to scrape {scrape_count} companies (without founding years)...")
    print(f"Rate limit: {RATE_LIMIT} seconds between requests")
    print(f"Estimated time: {scrape_count * RATE_LIMIT / 60:.1f} minutes")
    
    # Results storage for new scrapes only
    new_results = []
    
    # Scrape only companies without years
    for idx in range(min(scrape_count, len(missing_years))):
        company = missing_years.iloc[idx]['Company']
        
        # Scrape
        year, source = scrape_company(company)
        
        # Store result
        new_results.append({
            'Company': company,
            'Year_Founded': year,
            'Data_Source': source
        })
        
        # Rate limiting
        time.sleep(RATE_LIMIT)
        
        # Progress update
        if (idx + 1) % 10 == 0:
            print(f"\n--- Progress: {idx + 1}/{scrape_count} ({(idx+1)/scrape_count*100:.1f}%) ---")
    
    # Create DataFrame from new results
    new_results_df = pd.DataFrame(new_results) if new_results else pd.DataFrame(columns=['Company', 'Year_Founded', 'Data_Source'])
    
    # Build complete results: start with all companies from original df
    # Start with all companies from original dataframe
    results_df = df_original[['Company']].copy()
    
    # Add external spreadsheet data FIRST (highest priority - won't be overwritten)
    if external_data is not None:
        external_data_clean = external_data.drop_duplicates(subset=['Company'], keep='first')
        results_df = results_df.merge(external_data_clean, on='Company', how='left')
        external_count = external_data_clean['Year_Founded'].notna().sum()
        print(f"  ✓ Merged external spreadsheet data: {external_count} companies with founding years")
    
    # Add existing results if available (deduplicate first)
    # Only merge companies that don't already have data from external spreadsheet
    if existing_results is not None:
        # Remove duplicates from existing results, keep first occurrence
        existing_results_clean = existing_results[['Company', 'Year_Founded', 'Data_Source']].drop_duplicates(subset=['Company'], keep='first')
        # Only fill in missing values (don't overwrite external data)
        if external_data is not None:
            # Only merge companies that don't have external data
            missing_mask = results_df['Year_Founded'].isna()
            if missing_mask.any():
                missing_companies = results_df[missing_mask]['Company']
                existing_to_merge = existing_results_clean[existing_results_clean['Company'].isin(missing_companies)]
                if len(existing_to_merge) > 0:
                    results_df = results_df.merge(existing_to_merge, on='Company', how='left', suffixes=('', '_existing'))
                    # Fill in missing values from existing results
                    results_df['Year_Founded'] = results_df['Year_Founded'].fillna(results_df.get('Year_Founded_existing'))
                    results_df['Data_Source'] = results_df['Data_Source'].fillna(results_df.get('Data_Source_existing'))
                    # Drop temporary columns
                    results_df = results_df.drop(columns=[col for col in results_df.columns if col.endswith('_existing')], errors='ignore')
        else:
            # No external data, merge all existing results
            results_df = results_df.merge(existing_results_clean, on='Company', how='left')
    
    # Update with new scraped results (but preserve manual entries)
    if len(new_results_df) > 0:
        # Update existing rows (don't add new ones - all companies should already be in df_original)
        for _, row in new_results_df.iterrows():
            mask = results_df['Company'] == row['Company']
            if mask.any():
                # Only update if:
                # 1. Current value is NaN (missing), OR
                # 2. Current source is "Not Found" (failed scrape), OR
                # 3. Current source is from scraping (not manual)
                current_source = results_df.loc[mask, 'Data_Source'].iloc[0]
                current_year = results_df.loc[mask, 'Year_Founded'].iloc[0]
                
                # Preserve manual entries and external spreadsheet data - don't overwrite
                # Check if it's a manual entry or from external spreadsheet
                is_protected = (pd.notna(current_year) and 
                               (current_source == 'External Spreadsheet' or
                                current_source == 'Manual Correction' or
                                (current_source not in ['Wikipedia', 'LinkedIn', 'Company Website', 'Google Search', 'Crunchbase', 'Not Found'])))
                
                if not is_protected:
                    # Safe to update - either missing, failed, or from scraping
                    results_df.loc[mask, 'Year_Founded'] = row['Year_Founded']
                    results_df.loc[mask, 'Data_Source'] = row['Data_Source']
                # If it's a manual entry, keep the existing data
            # Don't add new companies - they should all be in the original Excel file
    
    # Ensure no duplicates (safety check)
    results_df = results_df.drop_duplicates(subset=['Company'], keep='first')
    
    # Statistics for new scrapes
    new_total = len(new_results_df)
    new_found = new_results_df['Year_Founded'].notna().sum()
    new_wikipedia = (new_results_df['Data_Source'] == 'Wikipedia').sum()
    new_linkedin = (new_results_df['Data_Source'] == 'LinkedIn').sum()
    new_website = (new_results_df['Data_Source'] == 'Company Website').sum()
    new_google = (new_results_df['Data_Source'] == 'Google Search').sum()
    new_failed = (new_results_df['Data_Source'] == 'Not Found').sum()
    
    # Overall statistics
    total = len(results_df)
    found = results_df['Year_Founded'].notna().sum()
    wikipedia = (results_df['Data_Source'] == 'Wikipedia').sum()
    linkedin = (results_df['Data_Source'] == 'LinkedIn').sum()
    website = (results_df['Data_Source'] == 'Company Website').sum()
    google = (results_df['Data_Source'] == 'Google Search').sum()
    failed = (results_df['Data_Source'] == 'Not Found').sum()
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE!")
    print("="*60)
    print(f"\n📊 New Scrapes (This Session):")
    print(f"  Scraped: {new_total}")
    if new_total > 0:
        print(f"  Found: {new_found} ({new_found/new_total*100:.1f}%)")
        print(f"  Wikipedia: {new_wikipedia}")
        print(f"  LinkedIn: {new_linkedin}")
        print(f"  Company Website: {new_website}")
        print(f"  Google Search: {new_google}")
        print(f"  Failed: {new_failed}")
    
    print(f"\n📊 Overall Statistics (All Data):")
    print(f"  Total companies: {total}")
    if total > 0:
        print(f"  With year: {found} ({found/total*100:.1f}%)")
        print(f"  Missing year: {total - found} ({(total-found)/total*100:.1f}%)")
        print(f"  Wikipedia: {wikipedia}")
        print(f"  LinkedIn: {linkedin}")
        print(f"  Company Website: {website}")
        print(f"  Google Search: {google}")
        print(f"  Failed: {failed}")
    
    # Save results (ensure no duplicates)
    results_df = results_df.drop_duplicates(subset=['Company'], keep='first')
    results_df.to_csv('scraped_founding_years.csv', index=False)
    print(f"\n✓ Results saved to: scraped_founding_years.csv ({len(results_df)} companies)")
    
    # Merge with original data (df_original has all original columns)
    merged = df_original.merge(results_df[['Company', 'Year_Founded', 'Data_Source']], 
                                on='Company', how='left')
    
    # Calculate Years_to_Unicorn (if Date Joined column exists)
    if 'Date Joined' in merged.columns and 'Year_Founded' in merged.columns:
        # Date Joined is already a datetime/timestamp, just extract year
        merged['Date_Joined_Year'] = pd.to_datetime(merged['Date Joined'], errors='coerce').dt.year
        # Only calculate for rows where both values exist
        merged['Years_to_Unicorn'] = merged['Date_Joined_Year'] - merged['Year_Founded']
    else:
        if 'Date Joined' not in merged.columns:
            print("  ⚠ Warning: 'Date Joined' column not found. Skipping Years_to_Unicorn calculation.")
        if 'Year_Founded' not in merged.columns:
            print("  ⚠ Warning: 'Year_Founded' column not found. Skipping Years_to_Unicorn calculation.")
    
    # Save augmented data
    try:
        merged.to_excel('unicorn_data_augmented.xlsx', index=False)
        print(f"✓ Augmented data saved to: unicorn_data_augmented.xlsx")
    except PermissionError:
        print(f"  ⚠ Permission denied: 'unicorn_data_augmented.xlsx' is open in another program.")
        print(f"  Please close the file and run the script again, or the data has been saved to scraped_founding_years.csv")
    except Exception as e:
        print(f"  ⚠ Error saving Excel file: {str(e)}")
        print(f"  Data has been saved to scraped_founding_years.csv")
    
    print("\n" + "="*60)
    print("Done! Check the output files.")
    print("="*60)

if __name__ == "__main__":
    main()