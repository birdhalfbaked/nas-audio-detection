"""Map ICAO 3-letter airline codes to spoken airline names (e.g. DAL -> Delta)."""

# Minimal map; extend as needed for operations.
ICAO_AIRLINE_NAME: dict[str, str] = {
    "DAL": "Delta",
    "UAL": "United",
    "AAL": "American",
    "SWA": "Southwest",
    "JBU": "JetBlue",
    "ASA": "Alaska",
}
