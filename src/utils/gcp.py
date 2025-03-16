from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="GCP-UTILS: %(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def create_big_query_dataset(
    bq_client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    exists_ok: bool = False,
) -> None:
    """Create a BigQuery dataset"""

    # Construct a full Dataset ID
    full_dataset_id = f"{project_id}.{dataset_id}"

    # Create a DatasetReference
    dataset = bigquery.Dataset(full_dataset_id)

    # Specify the location (optional)
    dataset.location = "US"

    # Create the dataset
    dataset = bq_client.create_dataset(dataset, exists_ok=exists_ok, timeout=30)

    LOGGER.info(f"Created dataset {full_dataset_id}")
