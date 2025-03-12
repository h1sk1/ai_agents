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
                }
            )

    def is_pdf(self, response):
        """Check if the response is a PDF file"""
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', 'ignore').lower()
        url = response.url.lower()

        return 'application/pdf' in content_type or url.endswith('.pdf')

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

            # Extract text from PDF using PyMuPDF
            pdf_content = ""
            with fitz.open(temp_path) as pdf_document:
                title = pdf_document.metadata.get('title', os.path.basename(response.url))
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    pdf_content += page.get_text()

            # Limit content size
            pdf_content = pdf_content[:300000]

            # Clean up the temporary file
            os.unlink(temp_path)

            yield {
                'url': response.url,
                'title': title,
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