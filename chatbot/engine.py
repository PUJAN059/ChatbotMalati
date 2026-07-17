"""
Chatbot NLP Engine
Handles intent classification, response generation, and conversation context.
Uses TF-IDF vectorization + cosine similarity for pattern matching.
Multi-API fallback: Groq → OpenRouter → local fallback.
Includes Bhagavad Gita and Bible knowledge bases for religious/spiritual questions.
"""

import json
import os
import random
import re
import string
import html
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class ChatbotEngine:
    """
    A lightweight NLP chatbot engine using TF-IDF and cosine similarity
    for intent classification. Supports short-term conversation memory
    and multi-API fallback for knowledge questions.
    """

    # Minimum similarity score to match an intent (0.0 - 1.0)
    CONFIDENCE_THRESHOLD = 0.5
    # Above this, always trust local intent (greetings, jokes, etc.)
    HIGH_CONFIDENCE = 0.7
    # How many recent messages to keep in context
    MAX_CONTEXT_LENGTH = 6

    def __init__(self, intents_path: str = None):
        if intents_path is None:
            intents_path = Path(__file__).parent.parent / "data" / "intents.json"

        self.intents_path = Path(intents_path)
        self.intents: list[dict] = []
        self.conversation_history: list[dict] = []
        # Track last used response per intent tag to avoid repeats
        self._last_response_by_tag: dict[str, str] = {}
        # Track recently used responses per intent for broader variety
        self._recent_responses_by_tag: dict[str, list[str]] = {}

        # Knowledge bases for religious/spiritual questions
        self._bhagavad_gita: dict = {}
        self._bible: dict = {}

        # TF-IDF structures
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self._all_patterns: list[str] = []
        self._pattern_tags: list[str] = []
        self._tfidf_matrix = None
        self._is_trained = False

        # Multi-API clients (in priority order)
        self._providers: list[dict] = []

        load_dotenv(Path(__file__).parent.parent / ".env")
        self._setup_providers()

        self._load_intents()
        self._load_religious_knowledge_bases()
        self._train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_intents(self) -> None:
        """Load intents from the JSON file."""
        if not self.intents_path.exists():
            raise FileNotFoundError(
                f"Intents file not found at: {self.intents_path}"
            )
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.intents = data.get("intents", [])

    def _load_religious_knowledge_bases(self) -> None:
        """Load Bhagavad Gita and Bible knowledge bases."""
        data_dir = Path(__file__).parent.parent / "data"

        # Load Bhagavad Gita
        gita_path = data_dir / "bhagavad_gita.json"
        if gita_path.exists():
            try:
                with open(gita_path, "r", encoding="utf-8") as f:
                    self._bhagavad_gita = json.load(f)
                print("[Malati] Bhagavad Gita knowledge base loaded.")
            except Exception as exc:
                print(f"[Malati] Error loading Bhagavad Gita: {exc}")

        # Load Bible
        bible_path = data_dir / "bible.json"
        if bible_path.exists():
            try:
                with open(bible_path, "r", encoding="utf-8") as f:
                    self._bible = json.load(f)
                print("[Malati] Bible knowledge base loaded.")
            except Exception as exc:
                print(f"[Malati] Error loading Bible: {exc}")

    def _clean_text(self, text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_answer_quantity(self, user_message: str) -> str:
        """
        Detect if the user wants a short (one), medium (few), or long (multiple/all) answer.
        Returns: 'short', 'medium', 'long'
        """
        lower = user_message.lower().strip()

        # Short answer patterns
        short_patterns = [
            "one line", "one verse", "one quote", "one shloka", "one shlok",
            "single line", "single verse", "single quote", "single shloka",
            "1 line", "1 verse", "1 quote", "1 shloka", "1 shlok",
            "just one", "only one", "just 1", "only 1",
            "short", "brief", "quick", "concise",
            "one liner", "one-liner", "1 liner",
            "one shloka", "ek shloka", "ek line",
            "best one", "most famous", "most popular",
            "top one", "top 1", "best verse", "best quote",
            "popular one", "famous one", "best line",
            "best verse", "best quote", "best shloka",
        ]

        # Long answer patterns
        long_patterns = [
            "multiple", "all verses", "all quotes", "all shlokas",
            "many", "several", "list", "give me all",
            "tell me all", "show me all", "explain all",
            "multiple lines", "multiple verses", "multiple quotes",
            "every verse", "every quote", "all of them",
            "full", "complete", "entire", "detailed",
            "elaborate", "comprehensive", "in detail",
            "all", "everything", "whole",
        ]

        # Check for short patterns first
        for pattern in short_patterns:
            if pattern in lower:
                return "short"

        # Check for long patterns
        for pattern in long_patterns:
            if pattern in lower:
                return "long"

        # Default to medium
        return "medium"

    def _is_religious_question(self, user_message: str) -> bool:
        """Detect if a question is about religious/spiritual topics."""
        lower = user_message.lower().strip()

        # Keywords for religious/spiritual topics
        religious_keywords = [
            "bhagavad gita", "bhagavad gita", "bhagwat gita", "bhagvat gita",
            "geeta", "gita", "krishna", "arjuna", "kurukshetra",
            "bible", "jesus", "christ", "christian", "gospel",
            "holy spirit", "lord", "divine",
            "prayer", "meditate", "meditation", "dhyana",
            "soul", "atman", "spirit",
            "commandment", "commandments", "ten commandments",
            "parable", "parables", "sermon", "beatitudes",
            "karma", "dharma", "moksha", "reincarnation",
            "rebirth", "samsara", "enlightenment", "salvation",
            "forgiveness", "repentance", "grace",
            "old testament", "new testament", "psalm", "psalms",
            "proverb", "proverbs", "apostle", "apostles",
            "scripture", "sacred", "worship", "devotion", "faith",
            "bhakti", "yoga philosophy", "yoga path",
            "karma yoga", "jnana yoga", "bhakti yoga", "raja yoga",
            "lord's prayer", "golden rule", "good samaritan",
            "prodigal son", "lost sheep", "mustard seed",
            "who is god", "what is god", "god existence",
            "after death", "life after death", "purpose of life",
            "meaning of life", "why are we here", "nature of god",
            "divine love", "unconditional love", "love god",
            "love neighbor", "love enemies", "surrender to god",
            "trust in god", "faith in god", "believe in god"
        ]

        # Strip punctuation from words for better matching
        import string
        clean_words = [w.strip(string.punctuation) for w in lower.split()]

        for keyword in religious_keywords:
            # For multi-word keywords, check if the full phrase is present
            if " " in keyword:
                if keyword in lower:
                    return True
            else:
                # For single words, check exact word match
                if keyword in clean_words:
                    return True

        return False

    def _search_bhagavad_gita(self, query: str, quantity: str = "medium") -> str:
        """Search the Bhagavad Gita knowledge base for relevant information."""
        if not self._bhagavad_gita:
            return ""

        lower = query.lower()
        results = []

        # Search through chapters
        for chapter in self._bhagavad_gita.get("chapters", []):
            chapter_text = f"{chapter.get('title', '')} {chapter.get('subtitle', '')}".lower()
            summary_text = chapter.get("summary", "").lower()
            teachings_text = " ".join(chapter.get("key_teachings", [])).lower()

            # Check if query matches chapter content
            if any(word in chapter_text or word in summary_text or word in teachings_text
                   for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "chapter",
                    "chapter": chapter.get("chapter"),
                    "title": chapter.get("title"),
                    "subtitle": chapter.get("subtitle"),
                    "summary": chapter.get("summary"),
                    "teachings": chapter.get("key_teachings", []),
                    "key_verses": chapter.get("key_verses", [])
                })

        # Search famous verses
        for verse in self._bhagavad_gita.get("famous_verses", []):
            verse_text = f"{verse.get('text', '')} {verse.get('translation', '')}".lower()
            if any(word in verse_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "famous_verse",
                    "verse": verse.get("verse"),
                    "text": verse.get("text"),
                    "translation": verse.get("translation"),
                    "significance": verse.get("significance")
                })

        # Search core teachings
        for key, teaching in self._bhagavad_gita.get("core_teachings", {}).items():
            teaching_text = f"{teaching.get('title', '')} {teaching.get('description', '')}".lower()
            if any(word in teaching_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "core_teaching",
                    "title": teaching.get("title"),
                    "description": teaching.get("description"),
                    "key_principle": teaching.get("key_principle"),
                    "reference": teaching.get("reference")
                })

        return self._format_gita_results(results, query, quantity)

    def _search_bible(self, query: str, quantity: str = "medium") -> str:
        """Search the Bible knowledge base for relevant information."""
        if not self._bible:
            return ""

        lower = query.lower()
        results = []

        # Search parables
        for parable in self._bible.get("parables", []):
            parable_text = f"{parable.get('title', '')} {parable.get('teaching', '')} {parable.get('story', '')}".lower()
            if any(word in parable_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "parable",
                    "title": parable.get("title"),
                    "reference": parable.get("reference"),
                    "story": parable.get("story"),
                    "teaching": parable.get("teaching"),
                    "lesson": parable.get("lesson")
                })

        # Search Beatitudes
        for beatitude in self._bible.get("beatitudes", []):
            beatitude_text = f"{beatitude.get('text', '')} {beatitude.get('meaning', '')}".lower()
            if any(word in beatitude_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "beatitude",
                    "verse": beatitude.get("verse"),
                    "text": beatitude.get("text"),
                    "meaning": beatitude.get("meaning")
                })

        # Search Jesus teachings
        for teaching_topic in self._bible.get("jesus_teachings", []):
            topic_text = f"{teaching_topic.get('topic', '')} {' '.join(teaching_topic.get('key_teachings', []))}".lower()
            if any(word in topic_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "jesus_teaching",
                    "topic": teaching_topic.get("topic"),
                    "reference": teaching_topic.get("reference"),
                    "key_teachings": teaching_topic.get("key_teachings"),
                    "significance": teaching_topic.get("significance")
                })

        # Search Old Testament stories
        for story in self._bible.get("old_testament_stories", []):
            story_text = f"{story.get('title', '')} {story.get('summary', '')} {story.get('teaching', '')}".lower()
            if any(word in story_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "ot_story",
                    "title": story.get("title"),
                    "reference": story.get("reference"),
                    "summary": story.get("summary"),
                    "teaching": story.get("teaching")
                })

        # Search Psalms and Proverbs
        for psalm in self._bible.get("psalms_and_proverbs", {}).get("selected_psalms", []):
            psalm_text = f"{psalm.get('text', '')} {psalm.get('meaning', '')}".lower()
            if any(word in psalm_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "psalm",
                    "reference": psalm.get("reference"),
                    "text": psalm.get("text"),
                    "meaning": psalm.get("meaning")
                })

        for proverb in self._bible.get("psalms_and_proverbs", {}).get("selected_proverbs", []):
            proverb_text = f"{proverb.get('text', '')} {proverb.get('meaning', '')}".lower()
            if any(word in proverb_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "proverb",
                    "reference": proverb.get("reference"),
                    "text": proverb.get("text"),
                    "meaning": proverb.get("meaning")
                })

        # Search Ten Commandments
        for commandment in self._bible.get("ten_commandments", []):
            cmd_text = f"{commandment.get('command', '')} {commandment.get('meaning', '')}".lower()
            if any(word in cmd_text for word in lower.split() if len(word) > 3):
                results.append({
                    "type": "commandment",
                    "number": commandment.get("number"),
                    "command": commandment.get("command"),
                    "reference": commandment.get("reference"),
                    "meaning": commandment.get("meaning")
                })

        # Search Apostle teachings
        for apostle_group in self._bible.get("apostle_teachings", []):
            for teaching in apostle_group.get("key_teachings", []):
                teaching_text = f"{teaching.get('topic', '')} {teaching.get('text', '')} {teaching.get('meaning', '')}".lower()
                if any(word in teaching_text for word in lower.split() if len(word) > 3):
                    results.append({
                        "type": "apostle_teaching",
                        "apostle": apostle_group.get("apostle"),
                        "topic": teaching.get("topic"),
                        "reference": teaching.get("reference"),
                        "text": teaching.get("text"),
                        "meaning": teaching.get("meaning")
                    })

        return self._format_bible_results(results, query, quantity)

    def _format_gita_results(self, results: list[dict], query: str, quantity: str = "medium") -> str:
        """Format Bhagavad Gita search results into a readable response."""
        if not results:
            return ""

        response_parts = []

        # Add general introduction if it's a broad question (but NOT for short answers)
        if quantity != "short" and any(word in query.lower() for word in ["about", "explain", "tell me", "what is", "teaching"]):
            response_parts.append(
                "The Bhagavad Gita is a 700-verse Hindu scripture that is part of the epic Mahabharata. "
                "It is a dialogue between Prince Arjuna and Lord Krishna, who serves as his charioteer. "
                "The text addresses the moral and philosophical dilemmas faced by Arjuna on the battlefield of Kurukshetra."
            )

        # Process results by type
        chapters = [r for r in results if r.get("type") == "chapter"]
        verses = [r for r in results if r.get("type") == "famous_verse"]
        teachings = [r for r in results if r.get("type") == "core_teaching"]

        # Determine limits based on quantity
        if quantity == "short":
            chapter_limit = 0
            verse_limit = 1
            teaching_limit = 0
        elif quantity == "long":
            chapter_limit = 5
            verse_limit = 5
            teaching_limit = 5
        else:  # medium
            chapter_limit = 1
            verse_limit = 2
            teaching_limit = 2

        # Add chapter information
        for chapter in chapters[:chapter_limit]:
            response_parts.append(f"\n**Chapter {chapter['chapter']}: {chapter['title']}** ({chapter['subtitle']})")
            response_parts.append(f"{chapter['summary']}")
            if chapter.get('teachings'):
                response_parts.append("\nKey teachings:")
                for teaching in chapter['teachings'][:3]:
                    response_parts.append(f"• {teaching}")

        # Add famous verses
        for verse in verses[:verse_limit]:
            response_parts.append(f"\n**Famous Verse (BG {verse['verse']}):**")
            response_parts.append(f"{verse['text']}")
            response_parts.append(f"Translation: {verse['translation']}")
            if quantity != "short":
                response_parts.append(f"Significance: {verse['significance']}")

        # Add core teachings
        for teaching in teachings[:teaching_limit]:
            response_parts.append(f"\n**{teaching['title']}:**")
            response_parts.append(f"{teaching['description']}")
            if quantity != "short":
                response_parts.append(f"Key principle: {teaching['key_principle']}")
                response_parts.append(f"Reference: {teaching['reference']}")

        return "\n".join(response_parts) if response_parts else ""

    def _format_bible_results(self, results: list[dict], query: str, quantity: str = "medium") -> str:
        """Format Bible search results into a readable response."""
        if not results:
            return ""

        response_parts = []

        # Add general introduction if it's a broad question (but NOT for short answers)
        if quantity != "short" and any(word in query.lower() for word in ["about", "explain", "tell me", "what is", "teaching"]):
            response_parts.append(
                "The Holy Bible is the sacred scripture of Christianity, consisting of the Old Testament (39 books) and "
                "New Testament (27 books). It contains religious texts, histories, prophecies, teachings of Jesus Christ, "
                "parables, commandments, and spiritual guidance."
            )

        # Process results by type
        parables = [r for r in results if r.get("type") == "parable"]
        beatitudes = [r for r in results if r.get("type") == "beatitude"]
        jesus_teachings = [r for r in results if r.get("type") == "jesus_teaching"]
        stories = [r for r in results if r.get("type") == "ot_story"]
        psalms = [r for r in results if r.get("type") == "psalm"]
        proverbs = [r for r in results if r.get("type") == "proverb"]
        commandments = [r for r in results if r.get("type") == "commandment"]
        apostle_teachings = [r for r in results if r.get("type") == "apostle_teaching"]

        # Determine limits based on quantity
        if quantity == "short":
            limit = 1
        elif quantity == "long":
            limit = 5
        else:  # medium
            limit = 2

        # Add parables
        for parable in parables[:limit]:
            response_parts.append(f"\n**The Parable of the {parable['title']}** ({parable['reference']})")
            if quantity == "short":
                response_parts.append(f"Teaching: {parable['teaching']}")
            else:
                response_parts.append(f"Story: {parable['story']}")
                response_parts.append(f"Teaching: {parable['teaching']}")
                response_parts.append(f"Lesson: {parable['lesson']}")

        # Add Beatitudes
        for beatitude in beatitudes[:limit]:
            response_parts.append(f"\n**Beatitude** ({beatitude['verse']})")
            response_parts.append(f"\"{beatitude['text']}\"")
            if quantity != "short":
                response_parts.append(f"Meaning: {beatitude['meaning']}")

        # Add Jesus teachings
        for teaching in jesus_teachings[:limit]:
            response_parts.append(f"\n**{teaching['topic']}** ({teaching['reference']})")
            if quantity == "short":
                response_parts.append(f"• {teaching['key_teachings'][0] if teaching['key_teachings'] else ''}")
            else:
                response_parts.append("Key teachings:")
                for t in teaching['key_teachings'][:3]:
                    response_parts.append(f"• {t}")
                if teaching.get('significance'):
                    response_parts.append(f"Significance: {teaching['significance']}")

        # Add Old Testament stories
        for story in stories[:limit]:
            response_parts.append(f"\n**The Story of {story['title']}** ({story['reference']})")
            if quantity == "short":
                response_parts.append(f"Teaching: {story['teaching']}")
            else:
                response_parts.append(f"{story['summary']}")
                response_parts.append(f"Teaching: {story['teaching']}")

        # Add Psalms
        for psalm in psalms[:limit]:
            response_parts.append(f"\n**{psalm['reference']}**")
            response_parts.append(f"\"{psalm['text']}\"")
            if quantity != "short":
                response_parts.append(f"Meaning: {psalm['meaning']}")

        # Add Proverbs
        for proverb in proverbs[:limit]:
            response_parts.append(f"\n**{proverb['reference']}**")
            response_parts.append(f"\"{proverb['text']}\"")
            if quantity != "short":
                response_parts.append(f"Meaning: {proverb['meaning']}")

        # Add Commandments
        for cmd in commandments[:limit]:
            response_parts.append(f"\n**Commandment #{cmd['number']}** ({cmd['reference']})")
            response_parts.append(f"\"{cmd['command']}\"")
            if quantity != "short":
                response_parts.append(f"Meaning: {cmd['meaning']}")

        # Add Apostle teachings
        for teaching in apostle_teachings[:limit]:
            response_parts.append(f"\n**{teaching['apostle']} on {teaching['topic']}** ({teaching['reference']})")
            response_parts.append(f"\"{teaching['text']}\"")
            if quantity != "short":
                response_parts.append(f"Meaning: {teaching['meaning']}")

        return "\n".join(response_parts) if response_parts else ""

    def _get_religious_response(self, user_message: str) -> str | None:
        """Generate a response for religious/spiritual questions using knowledge bases."""
        lower = user_message.lower()

        # Detect answer quantity (short/medium/long)
        quantity = self._detect_answer_quantity(user_message)
        print(f"[Malati] Detected quantity: {quantity}")

        # Determine which knowledge base to search
        is_gita = any(kw in lower for kw in [
            "bhagavad gita", "bhagavad gita", "bhagwat gita", "bhagvat gita",
            "geeta", "gita", "krishna", "arjuna", "kurukshetra",
            "karma yoga", "bhakti yoga", "jnana yoga", "raja yoga",
            "geeta gyan", "geeta updesh", "krishna updesh",
            "mahabharata", "atman", "moksha", "dharma", "karma",
            "reincarnation", "samsara", "yoga philosophy"
        ])

        is_bible = any(kw in lower for kw in [
            "bible", "jesus", "christ", "christian", "gospel",
            "old testament", "new testament", "psalm", "psalms",
            "proverb", "proverbs", "apostle", "apostles",
            "commandment", "commandments", "parable", "parables",
            "lord's prayer", "golden rule", "good samaritan",
            "prodigal son", "lost sheep", "mustard seed",
            "sermon", "beatitudes", "salvation", "sin",
            "forgiveness", "grace", "heaven", "hell",
            "resurrection", "crucifixion", "cross",
            "matthew", "mark", "luke", "john", "acts",
            "romans", "corinthians", "ephesians"
        ])

        # Search both if not specific
        responses = []

        if is_gita or not is_bible:
            gita_response = self._search_bhagavad_gita(user_message, quantity)
            if gita_response:
                responses.append(("bhagavad_gita", gita_response))

        if is_bible or not is_gita:
            bible_response = self._search_bible(user_message, quantity)
            if bible_response:
                responses.append(("bible", bible_response))

        if not responses:
            # Provide a general spiritual response for detected religious questions
            return self._get_general_religious_response(user_message, is_gita, is_bible)

        # Combine responses
        combined = []
        for source, response in responses:
            if source == "bhagavad_gita":
                combined.append(f"**From the Bhagavad Gita:**\n{response}")
            elif source == "bible":
                combined.append(f"**From the Holy Bible:**\n{response}")

        final_response = "\n\n---\n\n".join(combined)

        # Add appropriate closing based on quantity
        if quantity == "short":
            final_response += "\n\nWant more details? Just ask! 🙏"
        elif quantity == "long":
            final_response += "\n\nIs there anything specific you'd like me to elaborate on? 🙏"
        else:
            final_response += "\n\nWould you like me to explain any specific teaching in more detail? 🙏"

        return final_response

    def _get_general_religious_response(self, query: str, is_gita: bool, is_bible: bool) -> str:
        """Provide a general response for religious questions when specific search fails."""
        lower = query.lower()

        # Bhagavad Gita related responses
        if is_gita or any(kw in lower for kw in ["hindu", "vedic", "sanatan", "dharma", "karma", "yoga"]):
            if "reincarnation" in lower or "rebirth" in lower or "cycle" in lower or "samsara" in lower:
                return (
                    "**Reincarnation (Samsara) in Hindu Philosophy:**\n\n"
                    "Reincarnation is a core concept in Hinduism, known as Samsara — the cycle of birth, death, and rebirth. "
                    "According to the Bhagavad Gita, the soul (Atman) is eternal and indestructible. It merely changes bodies "
                    "like a person changes worn-out clothes.\n\n"
                    "**Key Teaching (BG 2.22):**\n"
                    "\"As a person puts on new garments, giving up old ones, the soul similarly accepts new material bodies, "
                    "giving up the old and useless ones.\"\n\n"
                    "**How it works:**\n"
                    "• The soul takes birth based on its karma (actions) and desires\n"
                    "• Good karma leads to better circumstances in future births\n"
                    "• Bad karma leads to less favorable circumstances\n"
                    "• The ultimate goal is Moksha — liberation from this cycle\n\n"
                    "**The Gita's perspective on transcending reincarnation:**\n"
                    "• Self-realization (knowing the true nature of the soul)\n"
                    "• Devotion to God (Bhakti Yoga)\n"
                    "• Selfless action (Karma Yoga)\n"
                    "• Knowledge (Jnana Yoga)\n\n"
                    "Would you like to know more about any specific aspect? 🙏"
                )
            elif "yoga" in lower and "philosophy" in lower:
                return (
                    "**Yoga Philosophy in the Bhagavad Gita:**\n\n"
                    "Yoga in the Bhagavad Gita is not just physical exercise — it is a comprehensive spiritual path "
                    "to union with the Divine. Krishna teaches four main paths of Yoga:\n\n"
                    "**1. Karma Yoga (Path of Action):**\n"
                    "Performing one's duty without attachment to results. Every action becomes an offering to God.\n"
                    "Key verse: \"You have the right to work, but never to the fruit of work.\" (BG 2.47)\n\n"
                    "**2. Bhakti Yoga (Path of Devotion):**\n"
                    "Loving devotion to God. The easiest and most accessible path. Complete surrender to the Divine.\n"
                    "Key verse: \"Surrender unto Me alone. I shall deliver you from all sinful reactions.\" (BG 18.66)\n\n"
                    "**3. Jnana Yoga (Path of Knowledge):**\n"
                    "Pursuit of spiritual wisdom and self-realization. Understanding the distinction between the eternal self and the temporary body.\n"
                    "Key verse: \"The soul is never born nor dies. It is unborn, eternal, ever-existing.\" (BG 2.20)\n\n"
                    "**4. Raja Yoga (Path of Meditation):**\n"
                    "Practicing meditation to control the mind and realize the self. Regular practice in a quiet place.\n"
                    "Key verse: \"The mind is the friend of the conditioned soul, and its enemy as well.\" (BG 6.5)\n\n"
                    "**The Goal of All Yoga:**\n"
                    "All paths lead to the same destination — liberation (Moksha) and eternal union with God.\n\n"
                    "Would you like to explore any specific path in more detail? 🙏"
                )
            else:
                return (
                    "**The Bhagavad Gita — Eternal Wisdom:**\n\n"
                    "The Bhagavad Gita is a 700-verse Hindu scripture that is part of the epic Mahabharata. "
                    "It is a dialogue between Prince Arjuna and Lord Krishna, who serves as his charioteer.\n\n"
                    "**Core Teachings:**\n"
                    "• **Karma Yoga:** Selfless action without attachment to results\n"
                    "• **Bhakti Yoga:** Loving devotion to God\n"
                    "• **Jnana Yoga:** Pursuit of spiritual knowledge\n"
                    "• **Dhyana Yoga:** Practice of meditation\n\n"
                    "**The Eternal Soul (Atman):**\n"
                    "\"The soul is never born nor dies at any time. It has not come into being, does not come into being, "
                    "and will not come into being. It is unborn, eternal, ever-existing, and primeval. It is not slain when the body is slain.\" (BG 2.20)\n\n"
                    "**The Supreme Teaching:**\n"
                    "\"Abandon all varieties of duty and just surrender unto Me. I shall deliver you from all sinful reactions. Do not grieve.\" (BG 18.66)\n\n"
                    "What specific aspect of the Gita would you like to learn more about? 🙏"
                )

        # Bible related responses
        if is_bible or any(kw in lower for kw in ["christian", "church", "faith"]):
            return (
                "**The Holy Bible — Divine Revelation:**\n\n"
                "The Holy Bible is the sacred scripture of Christianity, consisting of the Old Testament (39 books) "
                "and New Testament (27 books).\n\n"
                "**The Two Greatest Commandments (Matthew 22:37-40):**\n"
                "1. Love the Lord your God with all your heart, soul, and mind\n"
                "2. Love your neighbor as yourself\n\n"
                "**The Golden Rule (Matthew 7:12):**\n"
                "\"Do to others what you would have them do to you.\"\n\n"
                "**Key Teachings of Jesus:**\n"
                "• Forgive others as God forgives you\n"
                "• Love your enemies and pray for those who persecute you\n"
                "• The Good Shepherd lays down his life for the sheep\n"
                "• Faith as small as a mustard seed can move mountains\n\n"
                "**The Fruit of the Spirit (Galatians 5:22-23):**\n"
                "Love, joy, peace, forbearance, kindness, goodness, faithfulness, gentleness, and self-control.\n\n"
                "What specific topic from the Bible would you like to explore? ✝️"
            )

        # General spiritual response
        return (
            "**Spiritual Wisdom:**\n\n"
            "Both the Bhagavad Gita and the Bible offer profound spiritual wisdom:\n\n"
            "**Bhagavad Gita (Hinduism):**\n"
            "• The soul is eternal and indestructible\n"
            "• Perform selfless action (Karma Yoga)\n"
            "• Devote yourself to God (Bhakti Yoga)\n"
            "• Seek spiritual knowledge (Jnana Yoga)\n\n"
            "**The Bible (Christianity):**\n"
            "• Love God with all your heart\n"
            "• Love your neighbor as yourself\n"
            "• Forgive others as God forgives you\n"
            "• Have faith and trust in God\n\n"
            "Both traditions teach love, compassion, and the pursuit of spiritual truth. "
            "Would you like to explore any specific teaching in more detail? 🙏"
        )

    def _train(self) -> None:
        """Build a TF-IDF matrix from all intent patterns."""
        for intent in self.intents:
            tag = intent["tag"]
            for pattern in intent.get("patterns", []):
                self._all_patterns.append(self._clean_text(pattern))
                self._pattern_tags.append(tag)

        if self._all_patterns:
            self._tfidf_matrix = self.vectorizer.fit_transform(self._all_patterns)
            self._is_trained = True

    # ------------------------------------------------------------------
    # Web search (DuckDuckGo — free, no API key)
    # ------------------------------------------------------------------

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web with smarter query construction. Handles combined questions."""
        from datetime import datetime
        current_year = datetime.now().year
        lower = query.lower()

        # Detect combined questions ("X and Y", "X and also Y")
        and_parts = self._split_combined_question(query)
        if and_parts:
            # Search each part separately and combine results
            all_results = []
            for part in and_parts:
                part_result = self._single_search(part, current_year, max_results=3)
                if part_result:
                    all_results.append(f"[Results for: {part}]\n{part_result}")
            return "\n\n".join(all_results)

        return self._single_search(query, current_year, max_results)

    def _split_combined_question(self, query: str) -> list[str]:
        """Split combined questions like 'X and Y' into separate questions."""
        lower = query.lower()
        # Only split if it looks like a real combined question
        # e.g., "who is the president and prime minister of nepal"
        # but NOT "who is the president of nepal and china" (different meaning)
        connectors = [
            " and the ", " and also ", " and who is ",
            " and what is ", " and where is ",
        ]
        for conn in connectors:
            if conn in lower:
                parts = query.split(conn, 1)
                if len(parts) == 2 and len(parts[0].strip()) > 10 and len(parts[1].strip()) > 5:
                    return [parts[0].strip(), parts[1].strip()]

        # Also handle "X, Y" patterns like "home minister, prime minister"
        if ", " in lower and ("minister" in lower or "president" in lower or "governor" in lower):
            # Check if it's a list of roles for the same country
            import re
            match = re.match(r"(who (is|are) (the )?)(.*?),\s*(.*?)( of .+)?$", lower)
            if match:
                prefix = query[:match.start(4)]
                suffix = query[match.start(5):] if match.lastindex >= 5 else ""
                role1 = match.group(4).strip().rstrip(",");
                role2 = match.group(5).strip()
                # Reconstruct as separate questions
                country = suffix if suffix else ""
                q1 = f"{prefix}{role1} {country}".strip()
                q2 = f"{prefix}{role2} {country}".strip()
                if q1 != q2:
                    return [q1, q2]

        return []

    def _single_search(self, query: str, current_year: int, max_results: int = 5) -> str:
        """Search for a single query with smart variations."""
        lower = query.lower()
        queries_to_try = []

        if lower.startswith("who is") or lower.startswith("who was") or lower.startswith("who are"):
            stripped = lower.replace("who is", "").replace("who was", "").replace("who are", "").strip()
            # Remove "the" prefix
            stripped = re.sub(r"^the\s+", "", stripped)
            queries_to_try = [
                f"current {stripped} name {current_year}",
                f"{stripped} name {current_year}",
                f"who is {stripped} {current_year}",
                f"{stripped}",
            ]
        elif "latest" in lower or "newest" in lower:
            queries_to_try = [
                f"{query} {current_year}",
                f"{query}",
                f"{query} release date",
            ]
        else:
            queries_to_try = [
                f"{query} {current_year}",
                f"{query}",
                f"{query} latest",
                f"{query} current",
            ]

        for q in queries_to_try:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(q, max_results=max_results))
                if results:
                    lines = []
                    for r in results:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        if title or body:
                            lines.append(f"- {title}: {body}")
                    return "\n".join(lines[:max_results])
            except Exception as exc:
                print(f"[Malati] Search error for '{q}': {exc}")
                continue

        # Last resort: try news search
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
            if results:
                lines = []
                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    if title or body:
                        lines.append(f"- {title}: {body}")
                return "\n".join(lines[:max_results])
        except Exception as exc:
            print(f"[Malati] News search error: {exc}")

        return ""

    def _is_knowledge_question(self, user_message: str) -> bool:
        """Detect if a question likely needs real-time/web info."""
        lower = user_message.lower().strip()

        # Religious/spiritual questions should NOT be treated as web knowledge questions
        if self._is_religious_question(user_message):
            return False

        # Self-referential questions → should use local intents, not web search
        self_ref = [
            "who are you", "who am i", "what is your name", "what's your name",
            "how are you", "what can you do", "what do you know",
            "tell me about yourself", "introduce yourself",
        ]
        if any(lower.startswith(s) or lower == s for s in self_ref):
            return False

        # Direct question starters that almost always need current info
        starters = [
            "who is", "who was", "who are", "who will", "who's",
            "what is the", "what are the", "what was the", "what's the",
            "when is", "when did", "when was", "when will",
            "where is", "where was", "where are",
            "how many", "how much", "how old", "how far",
            "which is", "which country", "which city",
        ]
        if any(lower.startswith(s) or f" {s}" in lower for s in starters):
            return True

        # Also match simple "what is" / "what are" / "what's" (without "the")
        # but only for non-religious contexts (already filtered above)
        simple_starters = [
            "who is", "who was", "who are", "who will", "who's",
            "what is", "what are", "what was", "what's",
        ]
        if any(lower.startswith(s) for s in simple_starters):
            return True

        # Current/factual keywords
        factual = [
            "current", "latest", "today", "now", "recent", "newest",
            "prime minister", "president", "minister", "governor",
            "capital", "population", "currency", "language",
            "price", "cost", "rate", "value", "worth",
            "score", "result", "winner", "champion",
            "election", "war", "news", "happening", "event",
            "invented", "discovered", "founded", "established",
            "tallest", "biggest", "smallest", "longest", "fastest",
            "distance", "area", "size",
            "gdp", "economy", "inflation",
            "stock", "market", "crypto", "bitcoin",
            "weather", "temperature", "climate",
            "team", "player", "match", "score", "tournament",
            "movie", "film", "album", "song", "release",
            "company", "ceo", "founder",
            "population of", "capital of", "president of",
            "current year", "this year", "this month",
        ]
        return any(kw in lower for kw in factual)

    # ------------------------------------------------------------------
    # Multi-API provider setup
    # ------------------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict]:
        """Build the chat messages array with system prompt, search results, and history."""
        system_prompt = (
            "You are Malati, a friendly and humble AI chatbot "
            "built by Pujan Subedi. You are casual, warm, and love chatting with people. "
            "Keep responses short (2-3 sentences max), use emojis occasionally. "
            "If you don't know something, say so honestly and humbly. "
            "RULE: When asking riddles or playing guessing games, NEVER use emojis that "
            "reveal the answer. For example, don't use 🎹 if the answer is piano, or 🗺️ "
            "if the answer is map. Keep the mystery alive!"
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history
        for m in self.conversation_history[-6:]:
            role = "assistant" if m["role"] == "bot" else "user"
            messages.append({"role": role, "content": m["content"]})

        # Add web search results as a separate context message
        if self._is_knowledge_question(user_message):
            print(f"[Malati] Searching web for: {user_message}")
            search_context = self._web_search(user_message)
            if search_context:
                messages.append({
                    "role": "user",
                    "content": (
                        f"IMPORTANT: Below are real-time web search results. You MUST base your "
                        f"answer on these results. Do NOT say 'based on my knowledge cutoff' or "
                        f"'as of my training data'. The search results ARE your source.\n\n"
                        f"SEARCH RESULTS:\n{search_context}\n\n"
                        f"NOW ANSWER THE USER'S QUESTION using ONLY the search results above. "
                        f"If the results contain the answer, state it confidently. "
                        f"If they don't contain the exact answer, say 'I couldn't find the latest "
                        f"info on this — I'd recommend checking a news source.'"
                    ),
                })
                messages.append({
                    "role": "assistant",
                    "content": "Understood. I will answer using only the search results provided.",
                })

        # Add the actual user question
        messages.append({"role": "user", "content": user_message})

        return messages

    def _setup_providers(self) -> None:
        """Configure all available API providers in priority order.
        
        Tier 1: Ollama (local/cloud) — user-configured, most current
        Tier 2: Groq (Llama 3.3 70B) — most accurate, fastest
        Tier 3: OpenRouter DeepSeek R1 — strong reasoning
        Tier 4: OpenRouter Llama 3.1 8B — lightweight fallback
        """
        # ── Tier 1: Ollama Cloud ──────────────────────────────────────
        ollama_key = os.getenv("OLLAMA_API_KEY")
        if ollama_key and OLLAMA_AVAILABLE:
            try:
                client = ollama_lib.Client(
                    host="https://ollama.com",
                    headers={"Authorization": f"Bearer {ollama_key}"},
                )
                # Test the connection with a simple request
                test_response = client.chat(
                    model="gemma4:31b",
                    messages=[{"role": "user", "content": "hi"}],
                    options={"num_predict": 10},
                )
                self._providers.append({
                    "name": "Ollama Cloud",
                    "client": client,
                    "model": "gemma4:31b",
                    "max_tokens": 300,
                    "tier": 1,
                    "type": "ollama",
                })
                print("[Malati] Tier 1: Ollama Cloud (gemma4:31b) — connected.")
            except Exception as exc:
                print(f"[Malati] Failed to init Ollama Cloud: {exc}")
                print("[Malati] Falling through to next tier...")
        elif ollama_key and not OLLAMA_AVAILABLE:
            print("[Malati] Ollama key found but 'ollama' package not installed. Run: pip install ollama")

        # ── Tier 2: Groq ──────────────────────────────────────
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
                self._providers.append({
                    "name": "Groq",
                    "client": client,
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 300,
                    "tier": 2,
                })
                print("[Malati] Tier 2: Groq Llama 3.3 70B — connected.")
            except Exception as exc:
                print(f"[Malati] Failed to init Groq: {exc}")

        # ── Tier 3: OpenRouter (DeepSeek R1 — strong reasoning) ──
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
                self._providers.append({
                    "name": "OpenRouter",
                    "client": client,
                    "model": "deepseek/deepseek-r1:free",
                    "max_tokens": 300,
                    "tier": 3,
                })
                print("[Malati] Tier 3: OpenRouter DeepSeek R1 — connected.")
            except Exception as exc:
                print(f"[Malati] Failed to init OpenRouter (DeepSeek): {exc}")

            # ── Tier 4: OpenRouter (Llama 3.1 8B — lightweight) ──
            try:
                client2 = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
                self._providers.append({
                    "name": "OpenRouter",
                    "client": client2,
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "max_tokens": 300,
                    "tier": 4,
                })
                print("[Malati] Tier 4: OpenRouter Llama 3.1 8B — connected.")
            except Exception as exc:
                print(f"[Malati] Failed to init OpenRouter (Llama): {exc}")

        if not self._providers:
            print("[Malati] No API keys found — using local intents only.")
        else:
            print(f"[Malati] {len(self._providers)} providers loaded — multi-tier fallback active.")

    def _query_provider(self, provider: dict, user_message: str) -> str | None:
        """Query a single API provider. Returns response text or None."""
        try:
            messages = self._build_messages(user_message)
            if provider.get("type") == "ollama":
                response = provider["client"].chat(
                    model=provider["model"],
                    messages=messages,
                    options={"num_predict": provider["max_tokens"], "temperature": 0.7},
                )
                content = response["message"]["content"].strip()
            else:
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=messages,
                    max_tokens=provider["max_tokens"],
                    temperature=0.7,
                )
                content = response.choices[0].message.content.strip()

            # Unescape HTML entities like &quot; &amp; &lt; &gt;
            content = html.unescape(content)
            return content
        except Exception as exc:
            print(f"[Malati] {provider['name']} error: {exc}")
            return None

    def _get_ai_response(self, user_message: str) -> str | None:
        """
        Try each API provider in tier order until one succeeds.
        Tier 1 (Groq) → Tier 2 (DeepSeek R1) → Tier 3 (Llama 3.1 8B)
        Falls back to local response if all fail.
        """
        for provider in self._providers:
            reply = self._query_provider(provider, user_message)
            if reply:
                tier = provider.get("tier", "?")
                print(f"[Malati] Response from Tier {tier} ({provider['name']}/{provider['model']})")
                return reply
            else:
                tier = provider.get("tier", "?")
                print(f"[Malati] Tier {tier} failed, trying next...")
        print("[Malati] All providers failed — using local fallback.")
        return None

    # ------------------------------------------------------------------
    # Intent matching
    # ------------------------------------------------------------------

    def _get_intent_tag(self, user_message: str) -> tuple[str, float]:
        """Return the best-matching intent tag and its confidence score."""
        if not self._is_trained:
            return "unknown", 0.0

        cleaned = self._clean_text(user_message)
        query_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self.CONFIDENCE_THRESHOLD:
            return "unknown", best_score

        return self._pattern_tags[best_idx], best_score

    def _get_response_for_tag(self, tag: str) -> str:
        """Pick a random response, avoiding recent repeats until all are exhausted."""
        for intent in self.intents:
            if intent["tag"] == tag:
                responses = intent.get("responses", [])
                if not responses:
                    break

                # Get recent responses for this tag
                recent = self._recent_responses_by_tag.get(tag, [])

                # Filter out recently used responses
                available = [r for r in responses if r not in recent]

                # If all exhausted, reset and allow repeats
                if not available:
                    self._recent_responses_by_tag[tag] = []
                    available = list(responses)

                choice = random.choice(available)

                # Track it
                recent.append(choice)
                # Keep only last N to avoid repeats for a while
                max_recent = max(1, len(responses) - 1)
                if len(recent) > max_recent:
                    recent = recent[-max_recent:]
                self._recent_responses_by_tag[tag] = recent

                return choice

        return "I'm not sure about that one! Try asking me something else. 😊"

    def _apply_context(self, tag: str, user_message: str) -> str:
        """Adjust response based on conversation history."""
        cleaned_msg = self._clean_text(user_message)

        # Check if user asked "who am i" and we know their name
        who_am_i_variations = ["who am i", "who am", "do you know me", "what is my name", "what's my name", "remember me"]
        if any(var in cleaned_msg for var in who_am_i_variations):
            user_name = self._get_user_name_from_history()
            if user_name:
                return f"You're {user_name}! 😊 We've been chatting — I remember you! What else can I help with? 🌟"

        return self._get_response_for_tag(tag)

    def _get_user_name_from_history(self) -> str | None:
        """Extract the user's name from conversation history if they introduced themselves."""
        for msg in self.conversation_history:
            if msg["role"] != "user":
                continue
            content = msg["content"].lower().strip()
            # Check for "my name is X" patterns
            import re
            patterns = [
                r"my name is (\w[\w\s]*?)(?:\.|!|,|\s+and\b|\s+but\b|\s+so\b|\s+i\b|\s+what\b|\s+how\b|$)",
                r"i am (\w[\w\s]*?)(?:\.|!|,|\s+and\b|\s+but\b|\s+so\b|\s+i\b|\s+what\b|\s+how\b|$)",
                r"i'm (\w[\w\s]*?)(?:\.|!|,|\s+and\b|\s+but\b|\s+so\b|\s+i\b|\s+what\b|\s+how\b|$)",
                r"i am called (\w[\w\s]*?)(?:\.|!|,|\s+and\b|\s+but\b|\s+so\b|\s+i\b|\s+what\b|\s+how\b|$)",
                r"this is (\w[\w\s]*?)(?:\.|!|,|\s+and\b|\s+but\b|\s+so\b|\s+i\b|\s+what\b|\s+how\b|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    name = match.group(1).strip().title()
                    # Filter out common false positives
                    skip_words = ["a", "an", "the", "not", "just", "a student", "a developer"]
                    if name.lower() not in skip_words and len(name) > 1:
                        return name
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_response(self, user_message: str) -> dict:
        """
        Process a user message and return a structured response dict.
        Routes to local intents or AI providers based on confidence.
        """
        if not user_message or not user_message.strip():
            return {
                "response": "Please say something! I'm all ears. 👂",
                "intent": "empty",
                "confidence": 0.0,
            }

        user_message = user_message.strip()[:500]

        self.conversation_history.append({"role": "user", "content": user_message})
        if len(self.conversation_history) > self.MAX_CONTEXT_LENGTH * 2:
            self.conversation_history = self.conversation_history[-(self.MAX_CONTEXT_LENGTH * 2):]

        tag, confidence = self._get_intent_tag(user_message)

        # Check for religious/spiritual questions first using knowledge bases
        # This takes priority over TF-IDF matching
        if self._is_religious_question(user_message):
            print(f"[Malati] Religious question detected: {user_message}")
            religious_response = self._get_religious_response(user_message)
            if religious_response:
                self.conversation_history.append({"role": "bot", "content": religious_response})
                return {
                    "response": religious_response,
                    "intent": "religious_knowledge",
                    "confidence": round(confidence, 4),
                }

        # Always route knowledge/factual questions to AI with web search
        if self._is_knowledge_question(user_message):
            ai_reply = self._get_ai_response(user_message)
            if ai_reply:
                self.conversation_history.append({"role": "bot", "content": ai_reply})
                return {
                    "response": ai_reply,
                    "intent": "ai_fallback",
                    "confidence": round(confidence, 4),
                }

        # If TF-IDF isn't very confident, try AI providers
        if confidence < self.HIGH_CONFIDENCE:
            ai_reply = self._get_ai_response(user_message)
            if ai_reply:
                self.conversation_history.append({"role": "bot", "content": ai_reply})
                return {
                    "response": ai_reply,
                    "intent": "ai_fallback",
                    "confidence": round(confidence, 4),
                }

        response_text = self._apply_context(tag, user_message)
        self.conversation_history.append({"role": "bot", "content": response_text})

        return {
            "response": response_text,
            "intent": tag,
            "confidence": round(confidence, 4),
        }

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self.conversation_history.clear()
        self._last_response_by_tag.clear()
        self._recent_responses_by_tag.clear()

    def get_history(self) -> list[dict]:
        """Return a copy of the conversation history."""
        return list(self.conversation_history)
