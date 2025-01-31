from __future__ import annotations

import logging
import re
from itertools import cycle

import numpy as np
import requests
import tls_client
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter, Retry

from ..jobs import CompensationInterval, JobType


def create_logger(name: str):
    logger = logging.getLogger(f"JobSpy:{name}")
    logger.propagate = False
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        log_fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


class RotatingProxySession:
    def __init__(self, proxies=None):
        if isinstance(proxies, str):
            self.proxy_cycle = cycle([self.format_proxy(proxies)])
        elif isinstance(proxies, list):
            self.proxy_cycle = cycle([self.format_proxy(proxy) for proxy in proxies]) if proxies else None
        else:
            self.proxy_cycle = None

    @staticmethod
    def format_proxy(proxy):
        """Utility method to format a proxy string into a dictionary."""
        if proxy.startswith("http://") or proxy.startswith("https://"):
            return {"http": proxy, "https": proxy}
        return {"http": f"http://{proxy}", "https": f"http://{proxy}"}


class RequestsRotating(RotatingProxySession, requests.Session):
    def __init__(self, proxies=None, has_retry=False, delay=1, clear_cookies=False):
        RotatingProxySession.__init__(self, proxies=proxies)
        requests.Session.__init__(self)
        self.clear_cookies = clear_cookies
        self.allow_redirects = True
        self.setup_session(has_retry, delay)

    def setup_session(self, has_retry, delay):
        if has_retry:
            retries = Retry(
                total=3,
                connect=3,
                status=3,
                status_forcelist=[500, 502, 503, 504, 429],
                backoff_factor=delay,
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.mount("http://", adapter)
            self.mount("https://", adapter)

    def request(self, method, url, **kwargs):
        if self.clear_cookies:
            self.cookies.clear()

        if self.proxy_cycle:
            next_proxy = next(self.proxy_cycle)
            if next_proxy["http"] != "http://localhost":
                self.proxies = next_proxy
            else:
                self.proxies = {}
        return requests.Session.request(self, method, url, **kwargs)


class TLSRotating(RotatingProxySession, tls_client.Session):
    def __init__(self, proxies=None):
        RotatingProxySession.__init__(self, proxies=proxies)
        tls_client.Session.__init__(self, random_tls_extension_order=True)

    def execute_request(self, *args, **kwargs):
        if self.proxy_cycle:
            next_proxy = next(self.proxy_cycle)
            if next_proxy["http"] != "http://localhost":
                self.proxies = next_proxy
            else:
                self.proxies = {}
        response = tls_client.Session.execute_request(self, *args, **kwargs)
        response.ok = response.status_code in range(200, 400)
        return response


def create_session(
    *,
    proxies: dict | str | None = None,
    ca_cert: str | None = None,
    is_tls: bool = True,
    has_retry: bool = False,
    delay: int = 1,
    clear_cookies: bool = False,
) -> requests.Session:
    """
    Creates a requests session with optional tls, proxy, and retry settings.
    :return: A session object
    """
    if is_tls:
        session = TLSRotating(proxies=proxies)
    else:
        session = RequestsRotating(
            proxies=proxies,
            has_retry=has_retry,
            delay=delay,
            clear_cookies=clear_cookies,
        )

    if ca_cert:
        session.verify = ca_cert

    return session


class LoggingError(Exception):
    """Raised when there are logging-related errors."""

    def __init__(self, level_name: str):
        self.message = f"Invalid log level: {level_name!r}"
        super().__init__(self.message)


def set_logger_level(verbose: int = 2):
    """
    Adjusts the logger's level. This function allows the logging level to be changed at runtime.

    Parameters:
    - verbose: int {0, 1, 2} (default=2, all logs)
    """
    if verbose is None:
        return
    level_name = {2: "INFO", 1: "WARNING", 0: "ERROR"}.get(verbose, "INFO")
    level = getattr(logging, level_name.upper(), None)
    if level is not None:
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("JobSpy:"):
                logging.getLogger(logger_name).setLevel(level)
    else:
        raise LoggingError(level_name)


def markdown_converter(description_html: str):
    if description_html is None:
        return None
    markdown = md(description_html)
    return markdown.strip()


def extract_emails_from_text(text: str) -> list[str] | None:
    if not text:
        return None
    email_regex = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return email_regex.findall(text)


def get_enum_from_job_type(job_type_str: str) -> JobType | None:
    """
    Given a string, returns the corresponding JobType enum member if a match is found.
    """
    res = None
    for job_type in JobType:
        if job_type_str in job_type.value:
            res = job_type
    return res


def currency_parser(cur_str):
    # Remove any non-numerical characters
    # except for ',' '.' or '-' (e.g. EUR)
    cur_str = re.sub("[^-0-9.,]", "", cur_str)
    # Remove any 000s separators (either , or .)
    cur_str = re.sub("[.,]", "", cur_str[:-3]) + cur_str[-3:]

    if "." in list(cur_str[-3:]):
        num = float(cur_str)
    elif "," in list(cur_str[-3:]):
        num = float(cur_str.replace(",", "."))
    else:
        num = float(cur_str)

    return np.round(num, 2)


def remove_attributes(tag):
    for attr in list(tag.attrs):
        del tag[attr]
    return tag


def extract_salary(
    salary_str,
    lower_limit=1000,
    upper_limit=700000,
    hourly_threshold=350,
    monthly_threshold=30000,
    enforce_annual_salary=False,
):
    """
    Extracts salary information from a string and returns the salary interval, min and max salary values, and currency.
    (TODO: Needs test cases as the regex is complicated and may not cover all edge cases)
    """
    if not salary_str:
        return None, None, None, None

    parsed_values = parse_salary_string(salary_str)
    if not parsed_values:
        return None, None, None, None

    min_salary, max_salary = parsed_values
    return calculate_salary_range(
        min_salary,
        max_salary,
        lower_limit,
        upper_limit,
        hourly_threshold,
        monthly_threshold,
        enforce_annual_salary,
    )


def parse_salary_string(salary_str: str) -> tuple[int, int] | None:
    min_max_pattern = r"\$(\d+(?:,\d+)?(?:\.\d+)?)([kK]?)\s*[-—–]\s*(?:\$)?(\d+(?:,\d+)?(?:\.\d+)?)([kK]?)"  # noqa: RUF001
    match = re.search(min_max_pattern, salary_str)
    if not match:
        return None

    def to_int(s: str) -> int:
        return int(float(s.replace(",", "")))

    min_salary = to_int(match.group(1))
    max_salary = to_int(match.group(3))

    # Handle 'k' suffix for min and max salaries independently
    if "k" in match.group(2).lower() or "k" in match.group(4).lower():
        min_salary *= 1000
        max_salary *= 1000

    return min_salary, max_salary


def calculate_salary_range(
    min_salary: int,
    max_salary: int,
    lower_limit: int,
    upper_limit: int,
    hourly_threshold: int,
    monthly_threshold: int,
    enforce_annual_salary: bool,
) -> tuple[str, float, float, str] | tuple[None, None, None, None]:
    interval, annual_min, annual_max = determine_interval_and_annual_values(
        min_salary, max_salary, hourly_threshold, monthly_threshold
    )

    if not annual_max:
        return None, None, None, None

    if not is_valid_salary_range(annual_min, annual_max, lower_limit, upper_limit):
        return None, None, None, None

    if enforce_annual_salary:
        return interval, annual_min, annual_max, "USD"
    return interval, min_salary, max_salary, "USD"


def determine_interval_and_annual_values(
    min_salary: int, max_salary: int, hourly_threshold: int, monthly_threshold: int
) -> tuple[str, float, float | None]:
    if min_salary < hourly_threshold:
        return (
            CompensationInterval.HOURLY.value,
            min_salary * 2080,
            max_salary * 2080 if max_salary < hourly_threshold else None,
        )
    elif min_salary < monthly_threshold:
        return (
            CompensationInterval.MONTHLY.value,
            min_salary * 12,
            max_salary * 12 if max_salary < monthly_threshold else None,
        )
    return CompensationInterval.YEARLY.value, min_salary, max_salary


def is_valid_salary_range(annual_min: float, annual_max: float, lower_limit: int, upper_limit: int) -> bool:
    return (
        lower_limit <= annual_min <= upper_limit
        and lower_limit <= annual_max <= upper_limit
        and annual_min < annual_max
    )


def extract_job_type(description: str):
    if not description:
        return []

    keywords = {
        JobType.FULL_TIME: r"full\s?time",
        JobType.PART_TIME: r"part\s?time",
        JobType.INTERNSHIP: r"internship",
        JobType.CONTRACT: r"contract",
    }

    listing_types = []
    for key, pattern in keywords.items():
        if re.search(pattern, description, re.IGNORECASE):
            listing_types.append(key)

    return listing_types if listing_types else None


def setup_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


class InvalidLogLevelError(Exception):
    """Raised when an invalid logging level is provided."""

    def __init__(self, level_name: str):
        self.message = f"Invalid logging level: {level_name!r}"
        super().__init__(self.message)


def set_log_level(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), None)
    if level is not None:
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("jobspy2"):
                logging.getLogger(logger_name).setLevel(level)
    else:
        raise InvalidLogLevelError(level_name)
