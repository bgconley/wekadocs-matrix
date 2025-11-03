# src/ingestion/chunk_assembler.py
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Protocol

from src.providers.tokenizer_service import TokenizerService
from src.shared.chunk_utils import (
    create_chunk_metadata,
    create_combined_chunk_metadata,
    generate_chunk_id,
)
from src.shared.logging import get_logger

log = get_logger(__name__)


class ChunkAssembler(Protocol):
    def assemble(self, document_id: str, sections: List[Dict]) -> List[Dict]: ...


def get_chunk_assembler() -> ChunkAssembler:
    # Allow swapping to an alternate pipeline later (e.g., CHUNK_ASSEMBLER=pipeline)
    name = (os.getenv("CHUNK_ASSEMBLER") or "greedy").lower().strip()
    if name == "pipeline":
        try:
            from src.ingestion.pipeline_combiner import PipelineCombiner

            return PipelineCombiner()
        except Exception as e:
            log.warning(
                "CHUNK_ASSEMBLER=pipeline requested but unavailable: %s; falling back to greedy",
                e,
            )
    return GreedyCombinerV2()


class GreedyCombinerV2:
    """
    Phase 7E combiner: combine adjacent sections within the same 'block' (H1/H2),
    grow toward target_max tokens, then split only if above provider max.

    Fixes vs earlier attempts:
      - While condition uses g_tokens < target_max (not target_min)
      - H2 seeds CAN absorb H3/H4 descendants (same block), and *same-level siblings* too
      - Hard max is the provider limit (e.g., 8192) with safety headroom (7900)
      - Single-section chunks use create_chunk_metadata (is_combined=False)
      - Clean text assembly (no duplicate headings / random gaps)
      - Balance pass merges tiny trailing chunks within a block
      - Debug logging for every skip reason
    """

    def __init__(self):
        # Back-compat mapping: prefer COMBINE_* but accept legacy TARGET_* if present
        if os.getenv("TARGET_MIN_TOKENS") and not os.getenv("COMBINE_MIN_TOKENS"):
            os.environ["COMBINE_MIN_TOKENS"] = os.getenv("TARGET_MIN_TOKENS")
        if os.getenv("TARGET_MAX_TOKENS") and not os.getenv("COMBINE_TARGET_TOKENS"):
            os.environ["COMBINE_TARGET_TOKENS"] = os.getenv("TARGET_MAX_TOKENS")

        self.min_tokens = int(os.getenv("COMBINE_MIN_TOKENS", "900"))
        self.target_max = int(os.getenv("COMBINE_TARGET_TOKENS", "1300"))
        # Provider safety ceiling (use tokenizer’s max if available; otherwise safe default 7900)
        self.provider_max = int(os.getenv("EMBED_MAX_TOKENS", "8192"))
        self.hard_max = int(
            os.getenv("COMBINE_MAX_TOKENS", str(min(7900, self.provider_max)))
        )

        self.max_sections = int(os.getenv("COMBINE_MAX_SECTIONS", "12"))
        self.respect_levels = (
            os.getenv("COMBINE_RESPECT_MAJOR_LEVELS", "true").lower() == "true"
        )
        # H1/H2 act as "block anchors" by default; we combine within a block
        self.stop_at_level = int(os.getenv("COMBINE_STOP_AT_LEVEL", "2"))
        self.break_re = re.compile(
            os.getenv(
                "COMBINE_BREAK_KEYWORDS",
                r"(faq|faqs|glossary|reference|api reference|cli reference|changelog|release notes|troubleshooting)",
            ),
            re.I,
        )
        self.debug = os.getenv("COMBINE_DEBUG", "false").lower() == "true"

        self.tok = TokenizerService()

        if self.debug:
            log.debug(
                "Combiner cfg: min=%d target_max=%d hard_max=%d max_secs=%d stop_at_level=%d",
                self.min_tokens,
                self.target_max,
                self.hard_max,
                self.max_sections,
                self.stop_at_level,
            )

    # ---------- helpers ----------

    def _clean_text(self, s: Dict) -> str:
        """Return clean body text only; do not double-insert headings."""
        return (s.get("text") or "").strip()

    def _heading(self, s: Dict) -> str:
        return (s.get("title") or "").strip()

    def _is_block_anchor(self, s: Dict) -> bool:
        lvl = int(s.get("level", 3))
        return (self.respect_levels and lvl <= self.stop_at_level) or bool(
            self.break_re.search(self._heading(s))
        )

    def _build_blocks(self, sections: List[Dict]) -> List[int]:
        """
        Assign a block_id to each section.
        New block starts at H1/H2 or break keyword. Otherwise inherit current block.
        """
        block_ids: List[int] = []
        current_block = None
        for idx, s in enumerate(sections):
            if self._is_block_anchor(s) or current_block is None:
                current_block = idx  # choose deterministic block leader id
            block_ids.append(current_block)
        return block_ids

    def _build_citation_units(
        self, document_id: str, chunk_id: str, sections: List[Dict]
    ) -> List[Dict]:
        units: List[Dict] = []
        for section in sections:
            units.append(
                {
                    "id": section.get("id"),
                    "document_id": document_id,
                    "parent_chunk_id": chunk_id,
                    "heading": section.get("title", ""),
                    "text": section.get("body", ""),
                    "level": int(section.get("level", 3)),
                    "order": int(section.get("order", 0)),
                    "token_count": int(section.get("tokens", 0)),
                }
            )
        return units

    # ---------- main assemble ----------

    def assemble(self, document_id: str, sections: List[Dict]) -> List[Dict]:
        if not sections:
            return []

        block_ids = self._build_blocks(sections)
        chunks: List[Dict] = []
        i, N = 0, len(sections)

        while i < N:
            seed = sections[i]
            seed_block = block_ids[i]
            seed_level = int(seed.get("level", 3))

            group = [seed]
            texts = [self._clean_text(seed)]
            toks = [self.tok.count_tokens(texts[0])]
            g_tokens = toks[0]

            j = i + 1
            while (
                j < N
                and block_ids[j] == seed_block  # stay within same block
                and len(group) < self.max_sections
                and g_tokens < self.target_max  # <----- TARGET MAX controls growth
            ):
                cand = sections[j]
                # allow descendants OR same-level siblings within the block
                cand_level = int(cand.get("level", 3))
                same_or_descendant = cand_level >= seed_level

                # Avoid explicit break keywords inside a block
                if self.break_re.search(self._heading(cand)):
                    if self.debug:
                        log.debug(
                            "skip: break_keyword title=%r @j=%d", self._heading(cand), j
                        )
                    break

                if not same_or_descendant:
                    # next sibling of a parent anchor; stop this group
                    if self.debug:
                        log.debug(
                            "stop: encountered ancestor/sibling (level %d < seed %d) @j=%d",
                            cand_level,
                            seed_level,
                            j,
                        )
                    break

                cand_text = self._clean_text(cand)
                cand_tok = self.tok.count_tokens(cand_text)

                # Provider ceiling check only against hard_max (safety)
                if g_tokens + cand_tok > self.hard_max:
                    if self.debug:
                        log.debug(
                            "stop: provider hard_max would be exceeded (%d + %d > %d)",
                            g_tokens,
                            cand_tok,
                            self.hard_max,
                        )
                    break

                # Accept candidate
                group.append(cand)
                texts.append(cand_text)
                toks.append(cand_tok)
                g_tokens += cand_tok
                j += 1

            # If still below min and we have room & same block, try to keep adding even if > target_max until we reach min (but never exceed hard_max)
            while (
                g_tokens < self.min_tokens
                and j < N
                and block_ids[j] == seed_block
                and len(group) < self.max_sections
            ):
                cand = sections[j]
                cand_text = self._clean_text(cand)
                cand_tok = self.tok.count_tokens(cand_text)
                cand_level = int(cand.get("level", 3))
                if self.break_re.search(self._heading(cand)):
                    if self.debug:
                        log.debug("tail-stop: break_keyword @j=%d", j)
                    break
                # allow same-level or deeper within block
                if cand_level < seed_level:
                    if self.debug:
                        log.debug("tail-stop: encountered ancestor @j=%d", j)
                    break
                if g_tokens + cand_tok > self.hard_max:
                    if self.debug:
                        log.debug(
                            "tail-stop: provider hard_max would be exceeded (%d + %d > %d)",
                            g_tokens,
                            cand_tok,
                            self.hard_max,
                        )
                    break

                group.append(cand)
                texts.append(cand_text)
                toks.append(cand_tok)
                g_tokens += cand_tok
                j += 1

            # Build combined text (heading+body for first; body-only for following)
            parts: List[str] = []
            boundaries = {"combined": len(group) > 1, "sections": []}
            first_heading = self._heading(group[0])
            first_body = texts[0]
            if first_heading:
                parts.append(f"{first_heading}\n\n{first_body}")
            else:
                parts.append(first_body)

            for s, t, tt in zip(group[1:], texts[1:], toks[1:]):
                h = self._heading(s)
                if h:
                    parts.append(f"{h}\n\n{t}")
                else:
                    parts.append(t)
                boundaries["sections"].append(
                    {
                        "id": s["id"],
                        "order": int(s.get("order", 0)),
                        "level": int(s.get("level", 3)),
                        "tokens": int(tt),
                        "title": h or "",
                        "body": t,
                    }
                )
            # record first section too
            boundaries["sections"].insert(
                0,
                {
                    "id": group[0]["id"],
                    "order": int(group[0].get("order", 0)),
                    "level": int(group[0].get("level", 3)),
                    "tokens": int(toks[0]),
                    "title": first_heading or "",
                    "body": texts[0],
                },
            )

            combined = "\n\n".join([p.strip() for p in parts if p.strip()]).strip()
            c_tokens = self.tok.count_tokens(combined)

            orig_ids = [s["id"] for s in group]
            level = min(int(s.get("level", 3)) for s in group)
            order = int(group[0].get("order", 0))
            heading = first_heading
            parent_section_id = group[0]["id"]

            raw_sections = list(boundaries["sections"])
            clean_sections = []
            for sec in raw_sections:
                clean_sections.append(
                    {
                        "id": sec["id"],
                        "order": int(sec.get("order", 0)),
                        "level": int(sec.get("level", 3)),
                        "tokens": int(sec.get("tokens", 0)),
                        "title": sec.get("title") or "",
                    }
                )
            clean_boundaries = {
                "combined": boundaries["combined"],
                "sections": clean_sections,
            }

            boundaries = clean_boundaries

            if len(group) == 1:
                base_boundaries = {
                    "combined": False,
                    "sections": [clean_sections[0]],
                }
                if c_tokens <= self.hard_max:
                    # Single-section chunk (within provider limits)
                    meta = create_chunk_metadata(
                        section_id=group[0]["id"],
                        document_id=document_id,
                        level=level,
                        order=order,
                        heading=heading,
                        parent_section_id=parent_section_id,
                        is_combined=False,
                        is_split=False,
                        boundaries_json=json.dumps(
                            base_boundaries, separators=(",", ":")
                        ),
                        token_count=c_tokens,
                    )
                    meta["anchor"] = group[0].get("anchor", "")
                    meta["tokens"] = c_tokens
                    meta["text"] = combined
                    meta["checksum"] = hashlib.sha256(
                        combined.encode("utf-8")
                    ).hexdigest()
                    meta["doc_tag"] = group[0].get("doc_tag")
                    meta["_citation_units"] = self._build_citation_units(
                        document_id=document_id,
                        chunk_id=meta["id"],
                        sections=raw_sections,
                    )
                    chunks.append(meta)
                else:
                    # Single section exceeds provider limit → split deterministically
                    parts = self.tok.split_to_chunks(
                        combined, section_id=parent_section_id
                    )
                    for part in parts:
                        payload = {
                            "parent": base_boundaries,
                            "chunk_index": part["chunk_index"],
                            "total_chunks": part["total_chunks"],
                            "token_count": part["token_count"],
                            "overlap_start": part["overlap_start"],
                            "overlap_end": part["overlap_end"],
                        }
                        pj = json.dumps(payload, separators=(",", ":"))
                        bhash = hashlib.sha256(pj.encode("utf-8")).hexdigest()[:12]
                        sub_original_ids = [group[0]["id"], bhash]
                        sub_id = generate_chunk_id(document_id, sub_original_ids)

                        meta = create_chunk_metadata(
                            section_id=group[0]["id"],
                            document_id=document_id,
                            level=level,
                            order=order * 100000 + part["chunk_index"],
                            heading=heading,
                            parent_section_id=parent_section_id,
                            is_combined=False,
                            is_split=True,
                            boundaries_json=json.dumps(payload, separators=(",", ":")),
                            token_count=part["token_count"],
                        )
                        meta["id"] = sub_id
                        meta["original_section_ids"] = sub_original_ids
                        meta["anchor"] = group[0].get("anchor", "")
                        meta["tokens"] = part["token_count"]
                        meta["text"] = part["text"]
                        meta["checksum"] = part["integrity_hash"]
                        meta["doc_tag"] = group[0].get("doc_tag")
                        meta["_citation_units"] = self._build_citation_units(
                            document_id=document_id,
                            chunk_id=sub_id,
                            sections=raw_sections,
                        )
                        chunks.append(meta)
            else:
                if c_tokens <= self.hard_max:
                    meta = create_combined_chunk_metadata(
                        document_id=document_id,
                        original_section_ids=orig_ids,
                        level=level,
                        order=order,
                        heading=heading,
                        parent_section_id=parent_section_id,
                        token_count=c_tokens,
                        boundaries=clean_boundaries,
                    )
                    meta["anchor"] = group[0].get("anchor", "")
                    meta["tokens"] = c_tokens
                    meta["text"] = combined
                    meta["checksum"] = hashlib.sha256(
                        combined.encode("utf-8")
                    ).hexdigest()
                    meta["doc_tag"] = group[0].get("doc_tag")
                    meta["_citation_units"] = self._build_citation_units(
                        document_id=document_id,
                        chunk_id=meta["id"],
                        sections=raw_sections,
                    )
                    chunks.append(meta)
                else:
                    # Split combined unit (rare here because hard_max is high)
                    parts = self.tok.split_to_chunks(
                        combined, section_id=parent_section_id
                    )
                    for part in parts:
                        pj = json.dumps(
                            {
                                "parent": boundaries,
                                "chunk_index": part["chunk_index"],
                                "total_chunks": part["total_chunks"],
                                "token_count": part["token_count"],
                            },
                            separators=(",", ":"),
                        )
                        bhash = hashlib.sha256(pj.encode("utf-8")).hexdigest()[:12]
                        sub_id = generate_chunk_id(document_id, orig_ids + [bhash])
                        sub_chunk = {
                            "id": sub_id,
                            "document_id": document_id,
                            "level": level,
                            "order": order * 100000 + part["chunk_index"],
                            "heading": heading,
                            "parent_section_id": parent_section_id,
                            "original_section_ids": orig_ids,
                            "is_combined": True,
                            "is_split": True,
                            "boundaries_json": pj,
                            "token_count": int(part["token_count"]),
                            "updated_at": datetime.utcnow(),
                            "text": part["text"],
                            "checksum": hashlib.sha256(
                                part["text"].encode("utf-8")
                            ).hexdigest(),
                            "anchor": group[0].get("anchor", ""),
                            "tokens": int(part["token_count"]),
                            "doc_tag": group[0].get("doc_tag"),
                        }
                        sub_chunk["_citation_units"] = self._build_citation_units(
                            document_id=document_id,
                            chunk_id=sub_id,
                            sections=raw_sections,
                        )
                        chunks.append(sub_chunk)

            if self.debug:
                log.debug(
                    "assembled chunk: group=%d tokens=%d level=%d first=%r",
                    len(group),
                    c_tokens,
                    level,
                    heading,
                )

            i = j

        # Second pass: balance tiny tails within the same block
        chunks = self._balance_small_tails(chunks)

        # Ensure deterministic order
        chunks.sort(key=lambda c: (int(c.get("order", 0)), c["id"]))
        return chunks

    def _balance_small_tails(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks
        out: List[Dict] = []
        i = 0
        while i < len(chunks):
            if i == len(chunks) - 1:
                out.append(chunks[i])
                break
            cur, nxt = chunks[i], chunks[i + 1]
            # simple same-block heuristic: leading order bucket
            same_block = (int(cur.get("order", 0)) // 100000) == (
                int(nxt.get("order", 0)) // 100000
            )
            if (
                nxt["token_count"] < self.min_tokens
                and same_block
                and (cur["token_count"] + nxt["token_count"]) <= self.hard_max
            ):
                merged_text = (
                    cur["text"].rstrip() + "\n\n" + nxt["text"].lstrip()
                ).strip()
                merged_tokens = self.tok.count_tokens(merged_text)
                cur["text"] = merged_text
                cur["token_count"] = merged_tokens
                cur["tokens"] = merged_tokens
                cur["original_section_ids"] = list(
                    dict.fromkeys(
                        (
                            cur.get("original_section_ids")
                            or [cur.get("parent_section_id")]
                        )
                        + (
                            nxt.get("original_section_ids")
                            or [nxt.get("parent_section_id")]
                        )
                    )
                )
                cur["is_combined"] = len(cur["original_section_ids"]) > 1
                cur["is_split"] = cur.get("is_split", False) or nxt.get(
                    "is_split", False
                )
                try:
                    cur_boundaries = json.loads(cur.get("boundaries_json") or "{}")
                except json.JSONDecodeError:
                    cur_boundaries = {}
                try:
                    nxt_boundaries = json.loads(nxt.get("boundaries_json") or "{}")
                except json.JSONDecodeError:
                    nxt_boundaries = {}
                cur_sections = cur_boundaries.get("sections") or []
                nxt_sections = nxt_boundaries.get("sections") or []
                if nxt_sections:
                    if not isinstance(cur_sections, list):
                        cur_sections = []
                    cur_sections.extend(nxt_sections)
                    cur_boundaries["sections"] = cur_sections
                if cur_sections:
                    cur_boundaries["combined"] = True
                cur["boundaries_json"] = json.dumps(
                    cur_boundaries, separators=(",", ":")
                )
                cur["checksum"] = hashlib.sha256(
                    cur["text"].encode("utf-8")
                ).hexdigest()
                cur["updated_at"] = datetime.utcnow()
                out.append(cur)
                i += 2
            else:
                out.append(cur)
                i += 1
        return out
