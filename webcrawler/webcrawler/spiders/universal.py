import scrapy
import io
import fitz  # PyMuPDF
import tempfile
import os
from scrapy.crawler import CrawlerProcess
from twisted.internet import asyncioreactor
from scrapy.http import Request

# Install the asyncio reactor
try:
    asyncioreactor.install()
except Exception as e:
    print(f"Reactor installation failed: {e}")

class UniversalSpider(scrapy.Spider):
    name = 'universal_spider'
    custom_settings = {
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'timeout': 30 * 1000,
        },
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'CONCURRENT_REQUESTS': 8,
        "DUPEFILTER_CLASS": "scrapy.dupefilters.RFPDupeFilter",
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEED_EXPORT_FIELDS': ['url', 'title', 'content', 'depth', 'status', 'is_pdf'],
        'FEED_FORMAT': 'jsonl',
        'FEED_EXPORT_INDENT': None,  # Use compact format for better encoding compatibility
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'urls' in kwargs:
            self.start_urls = kwargs.get('urls').split(',')
        else:
            self.start_urls = ['https://google.com']

        self.allowed_domains = kwargs.get('allowed_domains', [])
        self.max_depth = kwargs.get('max_depth', 3)
        self.max_children_per_url = kwargs.get('max_children_per_url', 0)
        self.url_timeout = kwargs.get('url_timeout', 30)  # Timeout in seconds

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    'playwright': True,
                    'playwright_include_page': True,
                    'depth': 0,
                    'playwright_context': 'universal_context',
                    'playwright_context_options': {
                        'ignore_https_errors': True,
                    },
                    'download_timeout': self.url_timeout,
                    'playwright_page_init': self.page_init_handler,
                }
            )

    async def page_init_handler(self, page):
        """Configure page settings before navigation"""
        # Block loading of resource types we don't need
        await page.route('**/*', self.route_handler)

        # Optional: Set a user agent
        await page.set_extra_http_headers(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    @staticmethod
    async def route_handler(self, route):
        """Handle route interception to block unwanted resources"""
        resource_type = route.request.resource_type

        # Block these resource types to speed up crawling
        blocked_resources = ['stylesheet', 'image', 'media', 'font', 'script', 'other']

        if resource_type in blocked_resources:
            await route.abort()
        else:
            await route.continue_()

    def is_pdf(self, response):
        """Check if the response is a PDF file"""
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', 'ignore').lower()
        url = response.url.lower()

        return 'application/pdf' in content_type or url.endswith('.pdf')

    def _sanitize_text(self, text):
        """Clean text to ensure it can be encoded properly"""
        if text is None:
            return ""

        # Replace or remove problematic characters
        # Option 1: Replace with approximate ASCII
        try:
            # Normalize to NFKD form and remove non-ASCII
            import unicodedata
            text = unicodedata.normalize('NFKD', text)

            # Encode and decode with error handling
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        except Exception as e:
            self.logger.warning(f"Text sanitization error: {e}")
            # Fall back to basic ASCII
            text = str(text.encode('ascii', errors='ignore').decode('ascii', errors='ignore'))

        return text

    async def parse(self, response):
        depth = response.meta.get('depth', 3)
        if depth > self.max_depth:
            return

        # Check if this is a PDF
        if self.is_pdf(response):
            # Handle PDF content - can't use yield from in async function
            for item in self.parse_pdf(response):
                yield item
            return

        # Handle HTML content as before
        page = response.meta.get('playwright_page')
        try:
            content = (await page.evaluate('document.body.innerText'))[:300000] if page else ''
            yield {
                'url': response.url,
                'title': response.css('title::text').get(),
                'content': content,
                'depth': depth,
                'status': response.status,
                'is_pdf': False
            }
        finally:
            if page:
                await page.close()

    def parse_pdf(self, response):
        """Parse PDF content from the response body"""
        depth = response.meta.get('depth', 3)

        try:
            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.body)
                temp_path = temp_file.name

            # Extract text from PDF using PyMuPDF with encoding error handling
            pdf_content = ""
            with fitz.open(temp_path) as pdf_document:
                title = pdf_document.metadata.get('title', os.path.basename(response.url))
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    # Get text and handle potential encoding issues
                    try:
                        page_text = page.get_text()
                        # Clean or normalize text if needed
                        pdf_content += page_text
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num}: {e}")
                        # Continue with what we have so far

            # Limit content size and handle encoding
            pdf_content = self._sanitize_text(pdf_content[:50000])

            # Clean up the temporary file
            os.unlink(temp_path)

            yield {
                'url': response.url,
                'title': self._sanitize_text(title),
                'content': pdf_content,
                'depth': depth,
                'status': response.status,
                'is_pdf': True
            }
        except Exception as e:
            self.logger.error(f"Error processing PDF {response.url}: {e}")
            yield {
                'url': response.url,
                'title': "Error processing PDF",
                'content': f"Error: {str(e)}",
                'depth': depth,
                'status': response.status,
                'is_pdf': True
            }

    def _is_valid_url(self, url):
        if not self.allowed_domains:
            return True
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(d in domain for d in self.allowed_domains)