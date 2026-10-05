"""Check if generated feedback is supported by retrieved context."""

import re

import structlog

logger = structlog.get_logger()

# Common words that never count as overlap between a claim and its context
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "and",
        "or",
        "but",
        "in",
        "of",
        "to",
        "for",
        "that",
    }
)

# Generic resume wording that says someone has a skill without naming it,
# so "Knows Python" and "python expert" are both judged on "python" alone
FILLER_WORDS = frozenset(
    {
        "developer",
        "candidate",
        "has",
        "have",
        "had",
        "shows",
        "shown",
        "knows",
        "know",
        "experience",
        "experienced",
        "expert",
        "expertise",
        "skills",
        "skilled",
        "knowledge",
        "strong",
    }
)


class FaithfulnessChecker:
    """Verify that feedback claims are supported by context."""

    def check(self, feedback: str, context_chunks: list[dict]) -> float:
        """Check faithfulness of feedback to context.

        Args:
            feedback: Generated feedback text
            context_chunks: Retrieved context chunks

        Returns:
            Faithfulness score 0.0-1.0 (ratio of supported claims)
        """
        if not feedback or not context_chunks:
            logger.info(
                "faithfulness_empty_input",
                has_feedback=bool(feedback),
                has_chunks=bool(context_chunks),
            )
            return 0.0

        # Extract key claims from feedback (sentences)
        claims = self._extract_claims(feedback)
        if not claims:
            logger.info("faithfulness_no_claims_extracted")
            return 0.5  # Default to neutral if no extractable claims

        # Concatenate context text
        context_text = " ".join([chunk.get("text", "") for chunk in context_chunks])

        # Score each claim by the share of its content words found in the context
        ratios = [self._support_ratio(claim, context_text) for claim in claims]
        ratios = [r for r in ratios if r is not None]
        if not ratios:
            logger.info("faithfulness_no_content_claims")
            return 0.5  # Same neutral default as when no claims are extracted

        supported = sum(1 for r in ratios if r >= 0.5)
        score = sum(ratios) / len(ratios)

        logger.info(
            "faithfulness_checked", claims_count=len(claims), supported_count=supported, score=score
        )

        return score

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        """Extract key claims from feedback text.

        Args:
            text: Feedback text

        Returns:
            List of claims (sentences)
        """
        # Split by sentence (simple regex)
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if len(s.split()) >= 2]
        return claims[:10]  # Limit to 10 claims for scoring

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        """Lowercase word tokens with punctuation, stop words, and filler removed.

        Args:
            text: Claim or context text

        Returns:
            Set of content tokens
        """
        tokens = set(re.findall(r"[a-z0-9+#]+", text.lower()))
        return tokens - STOP_WORDS - FILLER_WORDS

    @classmethod
    def _support_ratio(cls, claim: str, context: str) -> float | None:
        """Share of a claim's content tokens that appear in the context.

        Args:
            claim: Claim text
            context: Context text

        Returns:
            Ratio 0.0-1.0, or None if the claim has no content tokens
        """
        claim_tokens = cls._content_tokens(claim)
        if not claim_tokens:
            return None
        return len(claim_tokens & cls._content_tokens(context)) / len(claim_tokens)

    @classmethod
    def _is_supported(cls, claim: str, context: str) -> bool:
        """Check if a claim is supported by context.

        Args:
            claim: Claim text
            context: Context text

        Returns:
            True if at least half of the claim's content tokens appear in the context
        """
        ratio = cls._support_ratio(claim, context)
        return ratio is not None and ratio >= 0.5
