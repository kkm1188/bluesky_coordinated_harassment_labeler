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
import re
from difflib import SequenceMatcher
from atproto import Client

# Import the label provided by the assignment GitHub repo
# from pylabel.label import post_from_url

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
        Moderate a batch of posts (for use with data.csv).

        This is useful for testing and evaluation where we have a dataset of posts
        rather than fetching from URLs one at a time.

        This method uses two-pass processing:
        1. First pass: Cache all posts by target
        2. Second pass: Evaluate each post with full context

        Args:
            posts_data: List of post dictionaries with keys:
                - post_id: unique identifier
                - author_id: author identifier
                - timestamp: datetime or ISO string
                - text: post content
                - target_user: (optional) explicitly identified target

        Returns:
            Dictionary mapping post_id to list of labels
        """
        # Sort posts by timestamp
        sorted_posts = sorted(posts_data, key=lambda p: self._parse_timestamp(p['timestamp']))

        # First pass: Build complete context for all targets
        target_posts = defaultdict(list)
        for post_dict in sorted_posts:
            targets = self._identify_targets(post_dict)
            for target in targets:
                target_posts[target].append(post_dict)

        # Second pass: Evaluate each post with full context
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
                current_time = self._parse_timestamp(post_dict['timestamp'])
                window_start = current_time - timedelta(minutes=TEMPORAL_WINDOW_MINUTES)

                # Filter posts within window
                context_posts = [
                    p for p in target_posts[target]
                    if window_start <= self._parse_timestamp(p['timestamp']) <= current_time
                ]

                if len(context_posts) < MIN_POSTS_FOR_COORDINATION:
                    continue

                # Compute score and assign labels
                score = self.compute_coordination_score(post_dict, context_posts)
                post_labels = self._labels_from_score(score)
                labels.update(post_labels)

            results[post_id] = list(labels)

        return results

    # Coordination Score Computation
    def compute_coordination_score(self, post: dict, group_context: List[dict]) -> float:
        """
        Computes the coordination score for a post 

        This algorithm combines three scores: 
        1. Temporal clustering (0-1): How concentrated are posts in time?
        2. Content similarity (0-1): How similar are the posts in content?
        3. Behavioral anomalies (0-1): Are there suspicious account patterns?

        Args:
            post: The post being analyzed
            group_context: List of all posts that mentions the target

        Returns:
            Coordination score between 0 and 1
        """
        
        # Extract the three signal scores from the helper methods
        temporal_score = self._compute_temporal_signal(group_context)
        similarity_score = self._compute_content_similarity_signal(group_context)
        behavioral_score = self._compute_behavioral_signal(group_context)

        # Combine computations through a weighted average
        # Temporal and similarity signals are stronger indicators than behavioral signals
        # High similarity + temporal clustering is the strongest signal for coordination
        weights = {
            'temporal': 0.30,
            'similarity': 0.50,  # Highest weight - copy-paste is strongest indicator
            'behavioral': 0.20
        }

        coordination_score = (
            weights['temporal'] * temporal_score +
            weights['similarity'] * similarity_score +
            weights['behavioral'] * behavioral_score
        )

        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, coordination_score))

    # Compute Temporal Signal Analysis
    def _compute_temporal_signal(self, posts: List[dict]) -> float:
        """
        Computes the temporal clustering signal
        
        Detects posts that mention the same target within a specific time window.
        A higher concentration of posts in shorter time = higher score
        Args:
            posts: List of post dicts
        Returns:
            Temporal signal score (0-1)
        """
        # Ensures there are multiple posts
        if len(posts) < 2:
            return 0.0

        # Collecting timestamps from all posts, turning it to datetime datatype, & sorting
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
        Computes content similarity signal
        Looks for similar text styles or copy/paste
        Args:
            posts: List of post dicts
        Returns:
            Content similarity score (0-1)
        """
        # Ensures there are multiple posts
        if len(posts) < 2:
            return 0.0

        # Grabs the text from the posts
        texts = [p.get('text', '') for p in posts]

        # Calculates pairwise similarities of the texts
        similarities = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Test similarities between 2 phrases
                sim = self._text_similarity(texts[i], texts[j])
                similarities.append(sim)
                
        # Making sure there is some smiliarity
        if len(similarities) == 0:
            return 0.0

        # Average similarity
        avg_similarity = sum(similarities) / len(similarities)

        # Check for repeated phrases (n-grams) with helper function
        repeated_phrases_score = self._detect_repeated_phrases(texts)

        # Combine the 2 scores with a slightly heavier weight to repeated phrases
        return (0.4 * avg_similarity) + (0.6 * repeated_phrases_score)

    def _compute_behavioral_signal(self, posts: List[dict]) -> float:
        """
        Compute behavioral anomaly signal through account metadata

        Detects suspicious account patterns:
        - New accounts (<30 days) participating in pile-ons
        - Accounts that suddenly spike in activity toward a single target
        - Similar account creation times (potential sock puppets)
        Args:
            posts: List of post dictionaries
        Returns:
            Behavioral anomaly score (0-1)
        """
        if len(posts) < 2:
            return 0.0

        # Extract unique authors
        authors = [p.get('author_id', '') for p in posts]
        unique_authors = list(set(authors))

        # Ensure there's more than 1 author
        if len(unique_authors) < 2:
            return 0.0  

        author_metadata = self._get_author_metadata(posts)

        signals = []
        # 1: Proportion of new accounts
        new_account_count = sum(1 for author in unique_authors if self._is_new_account(author, author_metadata.get(author)))
        new_account_ratio = new_account_count /len(unique_authors)
        signals.append(new_account_ratio)

        # 2: Account activity burst
        # Check if these accounts are suddenly very active (multiple posts in group)
        author_post_counts = defaultdict(int)
        for author in authors:
            author_post_counts[author] += 1

        avg_posts_per_author = sum(author_post_counts.values()) / len(author_post_counts)
        # If average posts per author is high, it could suggest concentrated activity
        burst_score = min(1.0, (avg_posts_per_author - 1) / 3.0)  # Normalize
        signals.append(burst_score)

        # 3: Diversity of participants
        # Many different accounts posting = higher coordination likelihood
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
    # 
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
        
        #** KENDALL LOOK AT THIS AND FIGURE IT OUT!!!
        # Removes urls (unncessary in searching for similarities)
        text = self.url_regex.sub('', text)

        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s@]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
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

    # ---------------------------
    # Label Assignment
    # ---------------------------

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
       
       

    # ---------------------------
    # UNECCESSARY API CODE!
    # ---------------------------

        
    # # Entry point for moderation via Bluesky API
    # def moderate_post(self, url: str) -> List[str]:
    #     """
    #     Apply coordinated harassment detection to the post specified by the given url
    #     The function will: 
    #         1. Fetch post and extract content
    #         2. Find potential targets
    #         3. Gather other posts about the target
    #         4. compute computation scores based on each of the algorithsm
    #         5. return the appropriate labels based on the score thresholds
    #     """
    #     # Loading post with helper function from label.py
    #     post = post_from_url(self.client, url)

    #     # Extract post content/metadata in the form of a dict
    #     post_data = self._extract_post_data(post)

    #     # Identify potential targets mentioned in the post
    #     targets = self._identify_targets(post_data)

    #     # If no targets are identified there is no harassment deteected
    #     if len(targets) == 0:
    #         return []

    #     # Loop through targets to search for areas of potential harassment 
    #     labels = set() # Using set for automatic deduplication of labels
    #     for target in targets:
    #         # Grab all of the recent posts about this target that are within the temporal window
    #         context_posts = self._get_context_posts(target, post_data['timestamp'])

    #         # Adding current post to post data for analysis
    #         all_posts = context_posts + [post_data]

    #         # only analyze all_posts if their are enough mentions to meet the threshold
    #         if len(all_posts) < MIN_POSTS_FOR_COORDINATION:
    #             continue

    #             # YOU FINISHED HERE!!!!
    #         # Compute coordination score
    #         score = self.compute_coordination_score(post_data, all_posts)

    #         # 6. Assign labels based on score
    #         post_labels = self._labels_from_score(score)
    #         labels.update(post_labels)

    #         # 7. Cache this post for future temporal analysis
    #         self._cache_post(target, post_data)

    #     return list(labels)
    
    
    #     # Grabbing necessary metadata
    # def _extract_post_data(self, post) -> dict:
    #     """
    #     Extract relevant data from a post object.

    #     Args:
    #         post: ATProto post object

    #     Returns:
    #         Dictionary with post data
    #     """
    #     # Extract text
    #     text = post.record.text

    #     # Extract timestamp
    #     timestamp = post.record.created_at
    #     if timestamp:
    #         timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    #     else:
    #         timestamp = datetime.now()

    #     # Extract author
    #     author_id = post.author
    #     if author_id:
    #         author_id = author_id.did or author_id.handle
    #     else:
    #         author_id = "unknown"

    #     # Extract post ID
    #     post_id = post.uri

    #     return {
    #         'post_id': post_id,
    #         'author_id': author_id,
    #         'timestamp': timestamp,
    #         'text': text
    #     }
    # def _get_context_posts(self, target: str, timestamp) -> List[dict]:
    #     """
    #     Finding the recent posts about the same target with the temporal window
    #     Args:
    #         target: target identifier
    #         timestamp: timestamp of post (so we can calculate temporal window)
    #     Return a list of posts within the window
    #     """
    #     # Double checking the target has been mentioned previously
    #     if target not in self.post_dict:
    #         return []

    #     # Calculating the temporal window starttime using datetime import
    #     # temporal window = timestamp - # min in temporal window
    #     temporal_window_end = self._parse_timestamp(timestamp) # turning timestamp to datetime object
    #     # using timedelta to indicate the unit of time measurement 
    #     temporal_window_start = temporal_window_end - timedelta(minutes = TEMPORAL_WINDOW_MINUTES)

    #     # Find all posts within temporal_window_start to temporal_window_end that specifically mention the target
    #     context = []
    #     for timestamp, post_data in self.post_dict[target]:
    #         # if the post is within the alloted timeframe
    #         if temporal_window_start <= timestamp <= temporal_window_end:
    #             context.append(post_data)

    #     return context
        
    # def _cache_post(self, target: str, post_data: dict):
    #     """
    #     Cache a post for future temporal analysis.

    #     Args:
    #         target: Target identifier
    #         post_data: Post dictionary
    #     """
    #     timestamp = self._parse_timestamp(post_data['timestamp'])
    #     self.post_dict[target].append((timestamp, post_data))

    #     # Clean old posts (beyond 2x temporal window)
    #     cutoff = datetime.now() - timedelta(minutes=2 * TEMPORAL_WINDOW_MINUTES)
    #     self.post_dict[target] = [
    #         (ts, data) for ts, data in self.post_dict[target]
    #         if ts >= cutoff
    #     ]



# Convenience alias for compatibility with testing harness
AutomatedLabeler = CoordinatedHarassmentLabeler
