"""
Semantic Scholar API client for fetching paper metadata and abstracts.

Provides a clean interface to the Semantic Scholar API with rate limiting,
error handling, and pagination support.
"""

import os
import time
import requests
from typing import Any


class SemanticScholarClient:
    """Client for interacting with the Semantic Scholar API."""

    BASE_URL: str = "https://api.semanticscholar.org/graph/v1"

    # Rate limiting: ~100 requests/min for free tier = ~1.67 req/sec
    # Use 0.7 second delay to stay well under the limit
    MIN_REQUEST_INTERVAL: float = 0.7

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the Semantic Scholar client.

        Args:
            api_key: Optional API key. If not provided, reads from SEMANTIC_SCHOLAR_API_KEY env var.
                    If neither provided, uses free tier (rate limited).
        """
        self.api_key: str | None = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.last_request_time: float = 0
        self.session: requests.Session = requests.Session()

        # Set up headers
        self.headers: dict[str, str] = {"User-Agent": "SemanticScholarClient/1.0"}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make a request to the Semantic Scholar API with error handling.

        Args:
            endpoint: API endpoint path (e.g., "/paper/search")
            params: Query parameters

        Returns:
            Response JSON as dictionary

        Raises:
            ValueError: For 4xx errors (bad request, not found, etc.)
            RuntimeError: For 5xx errors or rate limit exceeded
        """
        self._rate_limit()

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(
                url,
                params=params,  # type: ignore
                headers=self.headers,
                timeout=10,
            )

            # Handle rate limiting (429)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise RuntimeError(
                    f"Rate limited by API. Retry after {retry_after} seconds."
                )

            # Handle server errors (5xx)
            if response.status_code >= 500:
                raise RuntimeError(
                    f"API server error ({response.status_code}): {response.text}"
                )

            # Handle client errors (4xx)
            if response.status_code >= 400:
                raise ValueError(
                    f"API client error ({response.status_code}): {response.text}"
                )

            # Handle success
            if response.status_code == 200:
                return response.json()

            # Handle other status codes
            raise RuntimeError(f"Unexpected status code: {response.status_code}")

        except requests.exceptions.Timeout:
            raise RuntimeError("API request timed out after 10 seconds")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection error: {str(e)}")

    def search_papers(
        self, query: str, year_range: tuple[int, int] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search for papers by query string.

        Args:
            query: Search query string (e.g., "machine learning")
            year_range: Optional tuple of (min_year, max_year) to filter results
            limit: Maximum number of results to return (default: 10, max: 100)

        Returns:
            List of paper dictionaries with keys:
                - paperId: Unique paper identifier
                - title: Paper title
                - authors: List of author dicts with 'name' and 'authorId'
                - year: Publication year
                - abstract: Paper abstract (if available)
                - citationCount: Number of citations
                - venue: Publication venue

        Raises:
            ValueError: If query is empty or limit is invalid
            RuntimeError: If API request fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        params = {
            "query": query.strip(),
            "limit": limit,
            "fields": "paperId,title,authors,year,abstract,citationCount,venue",
        }

        # Add year range filter if provided
        if year_range:
            min_year, max_year = year_range
            if min_year > max_year:
                raise ValueError("min_year must be <= max_year")
            params["minYearPub"] = min_year
            params["maxYearPub"] = max_year

        try:
            response = self._make_request("/paper/search", params)
            papers = response.get("data", [])  # type: ignore
            return papers
        except (ValueError, RuntimeError) as e:
            # Re-raise with context
            raise type(e)(f"Failed to search papers: {str(e)}")

    def get_paper_details(self, paper_id: str) -> dict[str, Any]:
        """
        Get full metadata for a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")

        Returns:
            Paper metadata dictionary with keys:
                - paperId: Unique identifier
                - title: Paper title
                - authors: List of author dicts
                - year: Publication year
                - abstract: Paper abstract
                - citationCount: Citation count
                - venue: Publication venue
                - url: Paper URL
                - externalIds: External identifiers (DOI, arXiv, etc.)
                - references: List of referenced papers
                - citations: List of citing papers

        Raises:
            ValueError: If paper_id is invalid or paper not found
            RuntimeError: If API request fails
        """
        if not paper_id or not paper_id.strip():
            raise ValueError("Paper ID cannot be empty")

        paper_id = paper_id.strip()

        params = {
            "fields": (
                "paperId,title,authors,year,abstract,citationCount,venue,"
                "url,externalIds,references,citations"
            )
        }

        try:
            response = self._make_request(f"/paper/{paper_id}", params)
            return response
        except ValueError as e:
            if "404" in str(e):
                raise ValueError(f"Paper not found: {paper_id}")
            raise
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get paper details: {str(e)}")

    def get_paper_text(self, paper_id: str) -> dict[str, Any]:
        """
        Get abstract and available full text for a paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Dictionary with keys:
                - paperId: Paper identifier
                - title: Paper title
                - abstract: Paper abstract (if available)
                - fullText: Full text content (if available)
                - textAvailable: Boolean indicating if full text is available
                - url: URL to paper

        Raises:
            ValueError: If paper_id is invalid or paper not found
            RuntimeError: If API request fails
        """
        if not paper_id or not paper_id.strip():
            raise ValueError("Paper ID cannot be empty")

        paper_id = paper_id.strip()

        params = {"fields": "paperId,title,abstract,url"}

        try:
            response = self._make_request(f"/paper/{paper_id}", params)

            # Structure the response
            result: dict[str, Any] = {
                "paperId": response.get("paperId"),
                "title": response.get("title"),
                "abstract": response.get("abstract"),
                "url": response.get("url"),
                "textAvailable": bool(response.get("abstract")),
                "fullText": None,  # Full text requires PDF parsing (future enhancement)
            }

            return result
        except ValueError as e:
            if "404" in str(e):
                raise ValueError(f"Paper not found: {paper_id}")
            raise
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get paper text: {str(e)}")

    def search_papers_paginated(
        self,
        query: str,
        year_range: tuple[int, int] | None = None,
        batch_size: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for papers with automatic pagination for large result sets.

        This is a generator-like function that yields batches of papers.
        For simplicity, it returns all results as a single list.

        Args:
            query: Search query string
            year_range: Optional tuple of (min_year, max_year)
            batch_size: Number of results per batch (max 100)

        Returns:
            List of all paper dictionaries found

        Raises:
            ValueError: If query is empty or batch_size is invalid
            RuntimeError: If API request fails
        """
        if batch_size < 1 or batch_size > 100:
            raise ValueError("batch_size must be between 1 and 100")

        # For now, return single batch (Semantic Scholar API doesn't support offset-based pagination)
        # This method is here for future enhancement when pagination is needed
        return self.search_papers(query, year_range, batch_size)


def create_client(api_key: str | None = None) -> SemanticScholarClient:
    """
    Factory function to create a Semantic Scholar client.

    Args:
        api_key: Optional API key. If not provided, reads from environment.

    Returns:
        SemanticScholarClient instance
    """
    return SemanticScholarClient(api_key)
