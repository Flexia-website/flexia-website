import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
import logging

logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.visited_urls = set()
        self.timeout = 10
    
    def extract_images_from_page(self, url, max_images=30):
        """Extract image URLs from a web page"""
        images = []
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return images
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Regular image tags
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    full_url = urljoin(url, src)
                    images.append(full_url)
                    if len(images) >= max_images:
                        break
            
            # Lazy loading images (data-src)
            if len(images) < max_images:
                for img in soup.find_all(attrs={'data-src': True}):
                    src = img.get('data-src')
                    if src:
                        full_url = urljoin(url, src)
                        if full_url not in images:
                            images.append(full_url)
                            if len(images) >= max_images:
                                break
            
            logger.info(f"Extracted {len(images)} images from {url}")
            return images
            
        except requests.Timeout:
            logger.warning(f"Timeout extracting images from {url}")
            return images
        except Exception as e:
            logger.warning(f"Error extracting images from {url}: {e}")
            return images
    
    def search_websites(self, urls, searcher, max_pages=3):
        """Search multiple websites for faces"""
        all_results = []
        
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            logger.info(f"🌐 Crawling: {url}")
            pages_to_crawl = [url]
            crawled = 0
            
            while pages_to_crawl and crawled < max_pages:
                current_url = pages_to_crawl.pop(0)
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)
                crawled += 1
                
                try:
                    # Extract images
                    image_urls = self.extract_images_from_page(current_url)
                    
                    # Search each image
                    for img_url in image_urls:
                        try:
                            matches, error = searcher.search_enhanced(
                                img_url, top_k=3, use_landmarks=True
                            )
                            
                            if matches:
                                for match in matches:
                                    match['image_url'] = img_url
                                    match['source_page'] = current_url
                                    all_results.append(match)
                        except Exception as e:
                            logger.warning(f"Error searching image {img_url}: {e}")
                            continue
                    
                    # Find more pages (internal links)
                    if crawled < max_pages:
                        try:
                            response = self.session.get(current_url, timeout=self.timeout)
                            soup = BeautifulSoup(response.text, 'html.parser')
                            domain = urlparse(url).netloc
                            
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                full_url = urljoin(current_url, href)
                                parsed = urlparse(full_url)
                                
                                # Only follow internal links
                                if parsed.netloc == domain:
                                    if full_url not in self.visited_urls and full_url not in pages_to_crawl:
                                        pages_to_crawl.append(full_url)
                        except Exception as e:
                            logger.warning(f"Error finding links from {current_url}: {e}")
                    
                    # Be respectful
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error processing {current_url}: {e}")
                    continue
        
        logger.info(f"Web search found {len(all_results)} matches")
        return all_results
