import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlparse

import tldextract
import whois


FEATURE_NAMES = [
    "url_length",
    "has_https",
    "has_ip",
    "has_at_symbol",
    "special_char_count",
    "subdomain_count",
    "domain_age_days",
    "path_length",
]


def normalize_url(url: str) -> str:
    url = str(url).strip()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url


def get_domain(url: str) -> str:
    normalized = normalize_url(url)

    try:
        parsed = urlparse(normalized)
        return parsed.hostname or ""
    except Exception:
        return ""


def is_ip_address(hostname: str) -> int:
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def get_subdomain_count(hostname: str) -> int:
    if not hostname:
        return 0

    extracted = tldextract.extract(hostname)

    if not extracted.subdomain:
        return 0

    return len(
        [
            part
            for part in extracted.subdomain.split(".")
            if part
        ]
    )


def get_domain_age_days(domain: str) -> int:
    """
    WHOIS lookup used only when a dataset doesn't
    already provide domain age.
    """

    if not domain or is_ip_address(domain):
        return 0

    try:
        record = whois.whois(domain)

        creation_date = record.creation_date

        if not creation_date:
            return 0

        if isinstance(creation_date, list):
            creation_date = min(
                date for date in creation_date if date
            )

        if creation_date.tzinfo is None:
            creation_date = creation_date.replace(
                tzinfo=timezone.utc
            )

        now = datetime.now(timezone.utc)

        age = (now - creation_date).days

        return max(age, 0)

    except Exception:
        return 0


def extract_features(
    url: str,
    domain_age_override=None
) -> dict:

    normalized = normalize_url(url)

    parsed = urlparse(normalized)

    hostname = parsed.hostname or ""

    if domain_age_override is not None:
        try:
            domain_age = max(
                int(float(domain_age_override)),
                0
            )
        except (TypeError, ValueError):
            domain_age = 0
    else:
        domain_age = get_domain_age_days(
            hostname
        )

    special_characters = set(
        "@?=&%-_~;!$*+,"
    )

    features = {
        "url_length": len(normalized),

        "has_https": int(
            parsed.scheme.lower() == "https"
        ),

        "has_ip": is_ip_address(hostname),

        "has_at_symbol": int(
            "@" in normalized
        ),

        "special_char_count": sum(
            1
            for char in normalized
            if char in special_characters
        ),

        "subdomain_count": get_subdomain_count(
            hostname
        ),

        "domain_age_days": domain_age,

        "path_length": len(
            parsed.path or ""
        )
    }

    return features