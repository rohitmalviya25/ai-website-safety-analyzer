
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse


# Keep this order unchanged when training the ML model.
FEATURE_NAMES = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]


SUSPICIOUS_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".msi",
    ".apk", ".vbs", ".ps1", ".jar"
}


def shannon_entropy(text):
    """Measure how varied the characters in a string are."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def extract_features(url):
    """
    Convert a URL into numerical features.

    This function never visits the URL.
    It only analyzes the supplied text.
    """

    if not isinstance(url, str):
        raise ValueError("URL must be a string.")

    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    if len(url) > 2048:
        raise ValueError("URL is too long.")

    # Accept only HTTP/HTTPS URLs or bare hostnames.
    # Never fetch the submitted URL.
    if "://" not in url:
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    parsed = urlparse(url_to_parse)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    try:
        hostname = parsed.hostname or ""
        # Accessing port can raise ValueError for malformed URLs.
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Invalid URL format.") from error

    if not hostname:
        raise ValueError("A valid hostname is required.")

    # Normalize hostname for consistent feature extraction.
    hostname = hostname.lower().rstrip(".")

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        has_ip_address = 1
    except ValueError:
        has_ip_address = 0

    # Approximation: labels before the last two are subdomains.
    # This is not fully accurate for domains such as example.co.uk.
    domain_parts = hostname.split(".")
    subdomain_count = max(0, len(domain_parts) - 2)

    # Count digits in the entire URL.
    number_of_digits = sum(character.isdigit() for character in url)

    percentage_numeric_chars = (
        number_of_digits / len(url) * 100
    )

    # Detect suspicious executable/script-like path endings.
    path_lower = parsed.path.lower()
    suspicious_file_extension = int(
        any(
            path_lower.endswith(extension)
            for extension in SUSPICIOUS_EXTENSIONS
        )
    )

    # Count query parameters without visiting the website.
    if parsed.query:
        query_param_count = len(parsed.query.split("&"))
    else:
        query_param_count = 0

    # Count dot characters in the complete URL.
    dot_count = url.count(".")

    # Count tokens separated by common URL delimiters.
    tokens = [
        token for token in re.split(r"[/.:?&=_\-#]+", url)
        if token
    ]
    token_count = len(tokens)

    # Last hostname label is used as an approximate TLD.
    tld = domain_parts[-1] if domain_parts else ""

    features = {
        "url_length": len(url),
        "has_ip_address": has_ip_address,
        "dot_count": dot_count,
        "https_flag": int(parsed.scheme.lower() == "https"),
        "url_entropy": shannon_entropy(url),
        "token_count": token_count,
        "subdomain_count": subdomain_count,
        "query_param_count": query_param_count,
        "tld_length": len(tld),
        "path_length": len(parsed.path),
        "has_hyphen_in_domain": int("-" in hostname),
        "number_of_digits": number_of_digits,
        "suspicious_file_extension": suspicious_file_extension,
        "domain_name_length": len(hostname),
        "percentage_numeric_chars": percentage_numeric_chars,
    }

    # Guarantee the model always receives features in the same order.
    return {name: features[name] for name in FEATURE_NAMES}
