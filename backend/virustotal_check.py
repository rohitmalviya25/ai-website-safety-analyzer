
import os

import requests
from dotenv import load_dotenv

# Load the .env file from the project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
BASE_URL = "https://www.virustotal.com/api/v3"


def check_virustotal(url: str) -> dict:
    """Look up a URL's existing VirusTotal report."""

    if not API_KEY:
        return {
            "status": "unavailable",
            "message": "VirusTotal API key is not configured."
        }

    try:
        # URL identifier format used by VirusTotal API v3.
        import base64
        import hashlib

        url_id = base64.urlsafe_b64encode(
            url.encode("utf-8")
        ).decode("utf-8").rstrip("=")

        response = requests.get(
            f"{BASE_URL}/urls/{url_id}",
            headers={"x-apikey": API_KEY},
            timeout=15,
        )

        if response.status_code == 404:
            return {
                "status": "not_found",
                "message": "No existing VirusTotal report was found."
            }

        if response.status_code == 429:
            return {
                "status": "unavailable",
                "message": "VirusTotal API quota or rate limit reached."
            }

        response.raise_for_status()
        data = response.json()["data"]["attributes"]
        stats = data.get("last_analysis_stats", {})

        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless = stats.get("harmless", 0)
        undetected = stats.get("undetected", 0)

        return {
            "status": "reported",
            "malicious": malicious,
            "suspicious": suspicious,
            "harmless": harmless,
            "undetected": undetected,
            "total_engines": sum(stats.values()),
            "message": "Existing VirusTotal report retrieved."
        }

    except requests.RequestException:
        return {
            "status": "unavailable",
            "message": "Could not connect to VirusTotal."
        }
    except (ValueError, KeyError, TypeError):
        return {
            "status": "unavailable",
            "message": "VirusTotal returned an unexpected response."
        }
