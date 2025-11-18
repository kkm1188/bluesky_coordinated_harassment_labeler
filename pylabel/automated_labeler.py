"""
Implementation of automated moderator
"""

from typing import List, Tuple
from pathlib import Path
import re
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse

from atproto import Client

# Provided constant
T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 0.3  # They gave 0.3 but we will treat as Hamming threshold later

# Import the helper that fetches a post from a URL
from pylabel.label import post_from_url


# ---------------------------
# Perceptual Hash Utilities
# ---------------------------

def phash_int(image: Image.Image, hash_size: int = 8) -> int:
    """
    Simple perceptual average hash → integer bitstring.
    64-bit for hash_size=8.
    """
    img = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = [1 if px > avg else 0 for px in pixels]

    h = 0
    for b in bits:
        h = (h << 1) | b
    return h


def hamming(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


# ---------------------------
# Automated Labeler
# ---------------------------

class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client

        # Base directory: parent of pylabel/
        base = Path(__file__).resolve().parent.parent
        test_data = base / "test-data"

        # T&S words + domains
        self.t_and_s_words = self._load_single_column_csv(test_data / "t-and-s-words.csv")
        self.t_and_s_domains = self._load_single_column_csv(test_data / "t-and-s-domains.csv")

        # News domains → label
        self.news_map = self._load_news_domains(test_data / "news-domains.csv")

        # Dog images
        self.dog_hashes = self._load_dog_hashes(test_data / "dog-list-images")

        # Regex to catch URLs in post text
        self.url_regex = re.compile(r"https?://\S+")

    # ---------------------------
    # Main entrypoint
    # ---------------------------

    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation rules to the post specified by the given url.
        Returns list of label strings.
        """

        # 1. Load post using their helper
        post = post_from_url(self.client, url)

        # 2. Extract content
        text, links, images = self._extract_content(post)

        labels = []

        # 3. T&S
        if self._match_t_and_s(text, links):
            labels.append(T_AND_S_LABEL)

        # 4. News sources
        labels.extend(self._news_labels(links))

        # 5. Dog images
        if self._has_dog_image(images):
            labels.append(DOG_LABEL)

        return list(set(labels))

    # ---------------------------
    # CSV Loaders
    # ---------------------------

    @staticmethod
    def _load_single_column_csv(path: Path) -> set:
        df = pd.read_csv(path, header=None)
        return set(df[0].astype(str).str.strip().str.lower())

    @staticmethod
    def _load_news_domains(path: Path) -> dict:
        df = pd.read_csv(path, header=None)
        df[0] = df[0].astype(str).str.strip().str.lower()
        df[1] = df[1].astype(str).str.strip()
        return dict(zip(df[0], df[1]))

    # ---------------------------
    # Content Extraction
    # ---------------------------

    def _extract_content(self, post) -> Tuple[str, List[str], List[str]]:
        """
        Extract text, links, image URLs from the post structure returned by post_from_url.
        The structure normally looks like: post.record.text, post.embed.images, etc.
        """

        # Extract text
        text = getattr(post.record, "text", "") or ""

        # Extract URLs in text
        links = self.url_regex.findall(text)

        # Extract image URLs from embed
        image_urls = []
        embed = getattr(post, "embed", None)
        if embed and hasattr(embed, "images"):
            for img in embed.images:
                # Fullsize URLs are what you want
                if hasattr(img, "fullsize"):
                    image_urls.append(img.fullsize)

        return text, links, image_urls

    # ---------------------------
    # T&S Word + Domain Matching
    # ---------------------------

    def _match_t_and_s(self, text: str, links: List[str]) -> bool:
        lower_text = text.lower()

        # Word match
        for w in self.t_and_s_words:
            if w in lower_text and w.strip() != "":
                return True

        # Domain match
        for u in links:
            domain = self._domain(u)
            if domain in self.t_and_s_domains:
                return True

        return False

    @staticmethod
    def _domain(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host
        except:
            return ""

    # ---------------------------
    # News Labels
    # ---------------------------

    def _news_labels(self, links: List[str]) -> List[str]:
        labels = set()

        for u in links:
            domain = self._domain(u)
            # exact match
            if domain in self.news_map:
                labels.add(self.news_map[domain])
                continue

            # subdomain match (edition.cnn.com → cnn.com)
            for known, label in self.news_map.items():
                if domain.endswith("." + known):
                    labels.add(label)

        return list(labels)

    # ---------------------------
    # Dog Labeler (Perceptual Hash)
    # ---------------------------

    def _load_dog_hashes(self, directory: Path) -> List[int]:
        hashes = []
        if not directory.exists():
            return hashes

        for f in directory.iterdir():
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                img = Image.open(f)
                h = phash_int(img)
                hashes.append(h)
            except:
                pass

        return hashes

    def _has_dog_image(self, urls: List[str]) -> bool:
        if not urls or not self.dog_hashes:
            return False

        for url in urls:
            try:
                r = requests.get(url, timeout=5)
                r.raise_for_status()
                img = Image.open(BytesIO(r.content))
                h = phash_int(img)

                for dog_h in self.dog_hashes:
                    dist = hamming(h, dog_h)
                    if dist <= 10:  # typical threshold for 64-bit hash
                        return True
            except:
                continue

        return False
