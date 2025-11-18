"""
Coordinated Harassment/Bridgading Labeler Implementation
This labeler detects attempts of coordinated harassment using the following algorithms: 
    1. Temporal Signals - Detects posts that mentioon the same target within a short time window
    2. Content Similarity - Detects similarities between posts, i.e common phrases, copy paste, etc
    3. Behavioral Signals - Detects new accounts participating in the attack, usually created within 30 days
"""

from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from bisect import bisect_left, bisect_right
import re
from difflib import SequenceMatcher
from atproto import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# These are the label thresholds that we will use to determine the classification of threat
LABEL_POTENTIAL_COORDINATION = "potential-coordination"
LABEL_LIKELY_COORDINATION = "likely-coordination"
LABEL_CONFIRMED_COORDINATION = "confirmed-coordination-high-risk"

# These are the thresholds that will allow us to decide which classification/taxonomy they will be assigned
THRESHOLD_POTENTIAL = 0.40
THRESHOLD_LIKELY = 0.60
THRESHOLD_CONFIRMED = 0.75

# These are the parameters we will use to detect harassment, as chosen by trial and error
TEMPORAL_WINDOW_MINUTES = 10  # Time window in minutes for temporal signals
NEW_ACCOUNT_DAYS = 30  # New accounts created within this number of days for behavioral signals
SIMILARITY_THRESHOLD = 0.65  # This is the threshold for computed similarity 
MIN_POSTS_FOR_COORDINATION = 2  # This is the minimum number of posts to be considered for coordination


class CoordinatedHarassmentLabeler:
    """
    This is the class for the automated label used for detecting coordinated harassment campaigns
    
    It will combine the signal strength of each of the three algorithms, temporal signals, content
    similarity, and behavioral signals, into a coordination score between 0-1 and assign labels according
    to the thresholds defined above
    """

    def __init__(self, client: Client):
        """
        Initializing the labeler

        Args:
            client: Client for fetching posts
        """
        self.client = client

        # Dict for storing recent posts for temporal analysis
        # Key: user that was mentioned in the post, Value: list of posts that mention that person, including timestamps and post_data
        self.post_dict: Dict[str, List[Tuple[datetime, dict]]] = defaultdict(list)

        # Dict for account metadata for behavior analysis
        self.account_dict: Dict[str, dict] = {}

        # Using regex patterns for extracting mentions and normalizing post data
        self.mention_regex = re.compile(r'@(\w+)') # looking for @ mentions for the same person
        self.url_regex = re.compile(r'https?://\S+') # removes urls for comparing posts for normalization

    def moderate_posts_batch(self, posts_data: List[dict]) -> Dict[str, List[str]]:
        """
        Moderate a batch of posts for synthetic data with .csv file

        This method uses two-pass processing:
        1. First pass: Cache all posts by target
        2. Second pass: Evaluate each post with full context

        Args:
            posts_data: List of post dictionaries with keys:
                - post_id: unique identifier
                - author_id: author identifier
                - author_created_at: when author account was created
                - timestamp: datetime or ISO string
                - text: post content
                - target_user: (optional) explicitly identified target

        Returns:
            Dictionary mapping post_id to list of labels
        """
        # Parse all timestamps once and cache them (this helps mitigate redundant time stamping)
        timestamp_cache = {}
        for post in posts_data:
            post_id = post['post_id']
            timestamp_cache[post_id] = self._parse_timestamp(post['timestamp'])

        # Sort posts using cached timestamps
        sorted_posts = sorted(posts_data, key=lambda p: timestamp_cache[p['post_id']])

        # Build complete context dictionary for all posts in CSV file
        target_posts = defaultdict(list)
        for post_dict in sorted_posts:
            targets = self._identify_targets(post_dict)
            for target in targets:
                target_posts[target].append(post_dict)

        # Create sorted timestamp-post pairs for a binary search of temporal window (more efficient than checking all posts for timestamps)
        target_posts_sorted = {}
        for target, posts in target_posts.items():
            # Create list of (timestamp, post) tuples for binary search
            posts_with_times = [(timestamp_cache[p['post_id']], p) for p in posts]
            target_posts_sorted[target] = posts_with_times

        # Evaluate each post with full context
        ## MIGHT CHANGE THIS TO ONLY POSTS WITH NEGATIVE/ABUSIVE SENTIMENT ANALYSIS VIA LLM API
        results = {}
        for post_dict in sorted_posts:
            post_id = post_dict['post_id']
            targets = self._identify_targets(post_dict)

            if not targets:
                results[post_id] = []
                continue

            labels = set()
            for target in targets:
                # Get all posts about this target within temporal window
                # temporal window = current_time - temporal time window
                current_time = timestamp_cache[post_dict['post_id']]
                window_start = current_time - timedelta(minutes=TEMPORAL_WINDOW_MINUTES)

                # Use binary search to find posts within time window
                posts_with_times = target_posts_sorted[target]
                timestamps_only = [t for t, p in posts_with_times]

                # Binary search for range boundaries using python's bisect 
                left_idx = bisect_left(timestamps_only, window_start)
                right_idx = bisect_right(timestamps_only, current_time)

                # Extract posts in the time window
                context_posts = [p for t, p in posts_with_times[left_idx:right_idx]]

                # Ensure theirs enough posts for coordinated harassment
                if len(context_posts) < MIN_POSTS_FOR_COORDINATION:
                    continue

                # Compute score & assign labels
                score = self.compute_coordination_score(post_dict, context_posts, timestamp_cache)
                post_labels = self._labels_from_score(score)
                labels.update(post_labels)

            results[post_id] = list(labels)

        return results

    # Coordination Score Computation
    def compute_coordination_score(self, post: dict, group_context: List[dict], timestamp_cache: dict = None) -> float:
        """
        Computes the coordination score for a post

        This algorithm combines three scores:
        1. Temporal clustering (0-1): How concentrated are posts in time?
        2. Content similarity (0-1): How similar are the posts in content?
        3. Behavioral anomalies (0-1): Are there suspicious account patterns?

        Args:
            post: The post being analyzed
            group_context: List of all posts that mentions the target
            timestamp_cache: Dict mapping post_id to parsed datetime for temporal score analysis
        Returns:
            Coordination score between 0 and 1
        """

        # Extract the three signal scores from the helper methods
        temporal_score = self._compute_temporal_signal(group_context, timestamp_cache)
        similarity_score = self._compute_content_similarity_signal(group_context)
        behavioral_score = self._compute_behavioral_signal(group_context)

        # Combine computations through a weighted average
        # Temporal and similarity signals are stronger indicators than behavioral signals, so we've added more weight to them
        weights = {
            'temporal': 0.30,
            'similarity': 0.50,  # Highest weight - copy-paste is strongest indicator!
            'behavioral': 0.20
        }

        coordination_score = (
            weights['temporal'] * temporal_score + weights['similarity'] * similarity_score + weights['behavioral'] * behavioral_score)

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, coordination_score))

    # Compute Temporal Signal Analysis
    def _compute_temporal_signal(self, posts: List[dict], timestamp_cache: dict = None) -> float:
        """
        Computes the temporal clustering signal

        Detects posts that mention the same target within a specific time window.
        A higher concentration of posts in shorter time = higher score
        Args:
            posts: List of post dicts
            timestamp_cache: Dict mapping post_id to parsed datetime
        Returns:
            Temporal signal score (0-1)
        """
        # Ensures there are multiple posts
        if len(posts) < 2:
            return 0.0

        # Collecting timestamps from all posts, turning it to datetime datatype, & sorting
        if timestamp_cache:
            timestamps = [timestamp_cache[p['post_id']] for p in posts]
        else:
            # Fallback for compatibility
            ## DO WE NEED THIS? 
            timestamps = [self._parse_timestamp(p['timestamp']) for p in posts]
        timestamps.sort()

        # Calculating the timespan in minutes via last timestamp - first timestamp
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 60.0 

        # The shorter the window, the higher the score
        # Score decays as time span increases
        if time_span < 1:  # Less than 1 minute
            return 1.0
        elif time_span < 5:  # Less than 5 minutes
            return 0.8
        elif time_span < TEMPORAL_WINDOW_MINUTES:  # Within window
            # Linear decay from 0.8 to 0.4 for gradual decrease
            return 0.8 - (0.4 * (time_span - 5) / (TEMPORAL_WINDOW_MINUTES - 5))
        else:  # Outside of temporal window
            return 0.2  # Low but non-zero (could still be suspicious)

    def _compute_content_similarity_signal(self, posts: List[dict]) -> float:
        """
        Computes content similarity signal using ML library scikit Learn: TF-IDF + cosine similarity. This vectorizes ngrams 
        and compares similarity via matrix multiplication for a faster computation than brute force.
        Looks for similar text styles or copy/paste
        Args:
            posts: List of post dicts
        Returns:
            Content similarity score (0-1)
        """
        # Ensures there are multiple posts
        if len(posts) < 2:
            return 0.0

        # Grabs the text from the posts and normalize
        texts = [self._normalize_text(p.get('text', '')) for p in posts]

        # Filter out empty texts
        texts = [t for t in texts if len(t) > 0]
        if len(texts) < 2:
            return 0.0

        # Vectorize using TF-IDF with character n-grams 
        vectorizer = TfidfVectorizer(
            analyzer='char',      # Character-level
            ngram_range=(3, 5),   # Looking at 3-5 character n-grams
            min_df=1              # Include all n-grams
        )

        try:
            # Convert text to vectors
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Compute all pairwise cosine similarities at once (vectorized = fast!)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Extract similarities while avoiding redundancies
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarities.append(similarity_matrix[i, j])

            if len(similarities) == 0:
                return 0.0

            avg_similarity = np.mean(similarities)

        except (ValueError, AttributeError):
            # Fallback to original brute force method if TF-IDF fails (i.e all texts too short or identical)
            return self._compute_content_similarity_fallback(posts)

        # Check for word for word repeated phrases (n-grams) with helper function
        repeated_phrases_score = self._detect_repeated_phrases(texts)

        # Combine the 2 scores with a slightly heavier weight to copy/paste & repeated phrases
        return (0.4 * avg_similarity) + (0.6 * repeated_phrases_score)

    def _compute_content_similarity_fallback(self, posts: List[dict]) -> float:
        """
        Fallback to original SequenceMatcher approach if TF-IDF fails
        """
        texts = [p.get('text', '') for p in posts]

        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self._text_similarity(texts[i], texts[j])
                similarities.append(sim)

        if len(similarities) == 0:
            return 0.0

        avg_similarity = sum(similarities) / len(similarities)
        repeated_phrases_score = self._detect_repeated_phrases(texts)

        return (0.4 * avg_similarity) + (0.6 * repeated_phrases_score)

    def _compute_behavioral_signal(self, posts: List[dict]) -> float:
        """
        Compute behavioral anomaly signal through account metadata

        Detects suspicious account patterns:
        - New accounts (<30 days) participating in pile-ons
        - Accounts that suddenly spike in activity toward a single target
        - Similar account creation times
        Args:
            posts: List of post dictionaries
        Returns:
            Behavioral anomaly score (0-1)
        """
        if len(posts) < 2:
            return 0.0

        # Extract unique authors from all posts in the temporal window
        authors = [p.get('author_id', '') for p in posts]
        unique_authors = list(set(authors))

        # Ensure there's more than 1 author
        if len(unique_authors) < 2:
            return 0.0  

        author_metadata = self._get_author_metadata(posts)

        signals = []
        # computing new account -> unique authors ratio - seeing if its high
        new_account_count = sum(1 for author in unique_authors if self._is_new_account(author, author_metadata.get(author)))
        new_account_ratio = new_account_count /len(unique_authors)
        signals.append(new_account_ratio)

        # Checking account activity bursts - i.e check if these accounts are suddenly very active (multiple posts in group)
        author_post_counts = defaultdict(int)
        for author in authors:
            author_post_counts[author] += 1

        avg_posts_per_author = sum(author_post_counts.values()) / len(author_post_counts)
        
        # If average posts per author is high, it could suggest concentrated activity
        burst_score = min(1.0, (avg_posts_per_author - 1) / 3.0)  # Normalize
        signals.append(burst_score)

        # Diversity of participants: many different accounts posting = higher coordination likelihood
        participation_diversity = min(1.0, len(unique_authors) / 10.0)
        signals.append(participation_diversity)

        # Average the behavioral signals
        return sum(signals) / len(signals) if signals else 0.0
    
    # Extract author metadata for behavioral analysis
    def _get_author_metadata(self, posts: List[dict]) -> Dict[str, dict]: 
        """_summary_

        Args:
            posts (List[dict]): _description_

        Returns:
            Dict[str, dict]: _description_
        """
        author_metadata = {}
        for post in posts:
            author_id = post.get('author_id')
            if author_id not in author_metadata:
                if 'author_created_at' in post:
                    author_metadata[author_id] = {'created_at': post['author_created_at']}
        return author_metadata
    
    # Fallback method for checking text similarity 
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the similarity between two text strings by first normalizing the the text
        and utilizing sequence matching

        Args:
            text1: First text
            text2: Second text
        Returns:
            Similarity score (0-1)
        """
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        # Check that niether are blank
        if len(norm1) == 0 or len(norm2) == 0:
            return 0.0

        # Use SequenceMatcher for similarity ratio
        similarity = SequenceMatcher(None, norm1, norm2).ratio()

        # returns a ratio between 0-1 for similarity score
        return similarity

    # normalize texts within posts for checking similarity
    def _normalize_text(self, text: str) -> str:
        """
        Normalizes text for comparison:

        - Makes everything lowercase
        - Removes URLs
        - Removes extra whitespace
        - Removes punctuation

        Args:
            text: Input text
        Returns:
            Normalized text
        """
        # Makes everything lowercase
        text = text.lower()
        
        # Removes urls (unncessary in searching for similarities)
        text = self.url_regex.sub('', text)

        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s@]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    # Checking copy and pasted words
    def _detect_repeated_phrases(self, texts: List[str]) -> float:
        """
        Detects repeated phrases across multiple posts
        Looks for 3-grams and 4-grams that appear in multiple posts which is a strong indicator of copy-paste
        Args:
            texts: List of text strings
        Returns:
            Score (0-1) indicating prevalence of repeated phrases
        """
        if len(texts) < 2:
            return 0.0

        # Extract n-grams from each text
        # all_ngrams = all n-grams from every post - i.e a list of sets
        # ngrams = the set of n-grams in each post
        all_ngrams = []
        for text in texts:
            words = self._normalize_text(text).split()
            # Get 3-grams and 4-grams
            ngrams_single_post = set()
            for n in [3, 4]:
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    if len(ngram) > 10:  # Only meaningful phrases longer than 10 chars
                        ngrams_single_post.add(ngram)
            all_ngrams.append(ngrams_single_post)

        # Count how many posts share each n-gram
        ngram_counts = defaultdict(int)
        # for each post 
        for ngrams_single_post in all_ngrams:
            # for each n-gram in each post
            for ngram in ngrams_single_post:
                ngram_counts[ngram] += 1

        # Grab all n-grams that appeared more than once
        shared_ngrams = [count for count in ngram_counts.values() if count >= 2]

        # If no copied n-grams, return 0
        if len(shared_ngrams) == 0:
            return 0.0

        # ratio of highest number of shared n-grams to number of posts - i.e more shared n-grams = higher score
        max_sharing = max(shared_ngrams)
        ratio = max_sharing / len(texts)

        # capping ratio at 1
        return min(1.0, ratio)

    # identifying targets of @ mentions
    def _identify_targets(self, post_data: dict) -> Set[str]:
        """
        Gathering all potential targets of harassment in a post - i.e any one that's mentioned

        Looking for @ mentions to identify possible targets
        """
        # Using a set for automatic deduplication (multiple @ mentions are only tracked once)
        targets = set()

        # Extracting the @ mentions from the post and returning them in a set
        text = post_data.get('text', '')
        # using regex to find everything that looks like @*username*
        mentions = self.mention_regex.findall(text)
        targets.update(mentions)

        return targets

    # Helper function to find new accounts created within the threshold
    def _is_new_account(self, author_id: str, metadata: dict = None) -> bool:
        """
        Check if an account is within the new account threshold via metadata
        Args:
            author_id: Author identifier
            metadata: 
        Returns:
            True if account appears to be new
        """
        # Check cache
        if author_id in self.account_dict:
           return self.account_dict[author_id].get('is_new', False)

        isNew = False
        
        created_at = self._parse_timestamp(metadata['created_at'])
        account_age_days = (datetime.now() - created_at).days
        is_new = account_age_days < NEW_ACCOUNT_DAYS

        self.account_dict[author_id] = {'is_new': is_new}

        return is_new

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        Parse timestamp into datetime object.

        Args:
            timestamp: Datetime object or ISO string
        Returns:
            Datetime object
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                # Try other common formats
                try:
                    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except:
                    return datetime.now()
        else:
            return datetime.now()

    # adds the labels based on the thresholds
    def _labels_from_score(self, score: float) -> List[str]:
        """
        Convert coordination score to appropriate labels.

        Thresholds (from Tianyin's design):
        - 0.75-1.00: confirmed-coordination-high-risk
        - 0.60-0.75: likely-coordination
        - 0.40-0.60: potential-coordination
        - 0.00-0.40: no label

        Args:
            score: Coordination score (0-1)

        Returns:
            List of label strings
        """
        if score >= THRESHOLD_CONFIRMED:
            return [LABEL_CONFIRMED_COORDINATION]
        elif score >= THRESHOLD_LIKELY:
            return [LABEL_LIKELY_COORDINATION]
        elif score >= THRESHOLD_POTENTIAL:
            return [LABEL_POTENTIAL_COORDINATION]
        else:
            return []

# Convenience alias for compatibility with testing harness
AutomatedLabeler = CoordinatedHarassmentLabeler
