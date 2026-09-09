from typing import Protocol, Dict, Any

class InvoiceExtractor(Protocol):
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract structured invoice data from a file."""
        ...
