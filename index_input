from elasticsearch import Elasticsearch
import json
import argparse

def create_index(es, index_name):
    """Create an index in Elasticsearch."""
    es.indices.create(index=index_name, ignore=400)

def ingest_data_into_index(es, index_name, file_path):
    """Ingest JSON data into the specified Elasticsearch index."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        for i, doc in enumerate(data):
            es.index(index=index_name, id=i, body=doc)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create Elasticsearch index and ingest data.')
    parser.add_argument('--indexname', required=True, help='Name of the Elasticsearch index.')
    parser.add_argument('--filepath', required=True, help='Path to the JSON file containing the Q&A data.')
    args = parser.parse_args()

    # Initialize Elasticsearch client
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    # Create index and ingest data
    create_index(es, args.indexname)
    ingest_data_into_index(es, args.indexname, args.filepath)
