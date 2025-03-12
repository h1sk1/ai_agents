import os
import csv
import json
from itemadapter import ItemAdapter


class WebcrawlerPipeline:
    def process_item(self, item, spider):
        return item


class FileWriterPipeline:
    def __init__(self, output_path=None):
        self.output_path = output_path
        self.file = None
        self.writer = None

    @classmethod
    def from_crawler(cls, crawler):
        # Get output path from settings or use default
        output_path = crawler.spider.output_path if hasattr(crawler.spider, 'output_path') else 'output.csv'
        return cls(output_path)

    def open_spider(self, spider):
        # Create directory if it doesn't exist
        directory = os.path.dirname(self.output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Open the output file
        self.file = open(self.output_path, 'w', newline='', encoding='utf-8')

        # Determine file format from extension
        if self.output_path.endswith('.csv'):
            self.writer = csv.writer(self.file)
            # Write header
            self.writer.writerow(['url', 'title', 'content', 'depth', 'status'])

    def close_spider(self, spider):
        if self.file:
            self.file.close()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        if self.output_path.endswith('.csv'):
            self.writer.writerow([
                adapter.get('url', ''),
                adapter.get('title', ''),
                adapter.get('content', ''),
                adapter.get('depth', 0),
                adapter.get('status', 0)
            ])
        elif self.output_path.endswith('.jsonl'):
            # Write JSON line
            self.file.write(json.dumps(dict(item)) + '\n')

        return item