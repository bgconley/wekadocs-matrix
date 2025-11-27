import re
from typing import Dict, List

# Stopwords / generic terms to exclude from entity trie
GENERIC_BLACKLIST = {
    "at",
    "in",
    "id",
    "to",
    "of",
    "for",
    "the",
    "and",
    "or",
    "is",
    "it",
    "help",
    "color",
    "profile",
    "json",
    "output",
    "format",
    "filter",
    "verbose",
    "config",
    "data",
    "info",
    "test",
    "name",
    "type",
    "value",
    "true",
    "false",
    "none",
    "yes",
    "no",
    "default",
}


class EntityExtractor:
    """Lightweight trie-based entity extractor sourced from Neo4j Entity nodes."""

    def __init__(self, neo4j_driver):
        self.neo4j_driver = neo4j_driver
        self.trie: Dict = {}
        self._build_trie()

    def _insert(self, phrase: str, canonical: str) -> None:
        node = self.trie
        tokens = phrase.split()
        for token in tokens:
            node = node.setdefault(token, {})
        node["__end__"] = canonical

    def _should_include(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        if len(t) < 4:
            return False
        if t in GENERIC_BLACKLIST:
            return False
        return True

    def _build_trie(self) -> None:
        cypher = """
        MATCH (e:Entity)
        WHERE e.name IS NOT NULL AND size(trim(e.name)) > 0
        RETURN e.name AS name, e.aliases AS aliases, e.canonical_name AS canonical_name
        """
        with self.neo4j_driver.session() as session:
            result = session.run(cypher)
            for record in result:
                name = (record.get("name") or "").strip()
                canonical_name = (record.get("canonical_name") or "").strip()
                aliases = record.get("aliases") or []

                primary = canonical_name or name
                if self._should_include(primary):
                    self._insert(primary.lower(), primary)

                if self._should_include(name):
                    self._insert(name.lower(), name)

                for alias in aliases:
                    alias_str = str(alias or "").strip().lower()
                    if self._should_include(alias_str):
                        self._insert(alias_str, primary or name)

    def extract_entities(self, query: str, max_len: int = 5) -> List[str]:
        tokens = re.split(r"\s+", (query or "").lower().strip())
        entities: List[str] = []
        i = 0
        while i < len(tokens):
            best = None
            best_len = 0
            for length in range(min(max_len, len(tokens) - i), 0, -1):
                candidate = " ".join(tokens[i : i + length])
                node = self.trie
                found = True
                for t in candidate.split():
                    if t not in node:
                        found = False
                        break
                    node = node[t]
                if found and "__end__" in node:
                    best = node["__end__"]
                    best_len = length
                    break
            if best:
                entities.append(best)
                i += best_len
            else:
                i += 1
        return entities
