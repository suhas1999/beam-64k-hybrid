import random
import os
import re
import time
import math
import pickle
from collections import defaultdict
from typing import List, Dict
import json
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from json_repair import repair_json
import concurrent.futures
import threading
import cProfile
import pstats
import asyncio
import ast
import multiprocessing
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from concurrent.futures import ProcessPoolExecutor

from src.llm import *
from src.prompts import *
from src.beam.profile_creation import create_profile
from src.beam.utils import extract_time_anchor, get_token_number


class ConversationSummaryBuffer:
    def __init__(self, llm, max_tokens=8000, recent_messages_count=6, max_summary_tokens=2000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.recent_messages_count = recent_messages_count
        self.max_summary_tokens = max_summary_tokens
        self.messages = []
        self.summary = ""

        self.token_encoder = self._get_token_encoder()

    def _get_token_encoder(self):
        """Initialize the best available token encoder"""
        try:
            encoders_to_try = ["cl100k_base", "p50k_base", "r50k_base"]

            for encoding_name in encoders_to_try:
                try:
                    encoder = tiktoken.get_encoding(encoding_name)
                    print(f"Using tiktoken with {encoding_name} encoding")
                    return encoder
                except:
                    continue
        except:
            pass

    def estimate_tokens(self, text):
        """Count tokens using the initialized encoder"""
        if not text:
            return 0

        text = str(text)

        if self.token_encoder:
            try:
                return len(self.token_encoder.encode(text))
            except:
                return len(text) // 3
        else:
            return len(text) // 3

    def get_total_tokens(self):
        """Calculate total tokens in current messages + summary"""
        message_tokens = sum(self.estimate_tokens(
            msg["content"]) for msg in self.messages)
        summary_tokens = self.estimate_tokens(
            self.summary) if self.summary else 0
        return message_tokens + summary_tokens

    def add_message(self, role, content):
        """Add a new message and trigger compression if needed"""
        self.messages.append({"role": role, "content": content})

        self._ensure_token_limit()

    def save_context(self, user_input, ai_output):
        """Add user-AI exchange"""
        self.add_message("user", user_input)
        self.add_message("assistant", ai_output)

    def _ensure_token_limit(self):
        """Ensure total tokens stay within limit through multiple strategies"""
        max_iterations = 10
        iteration = 0

        while self.get_total_tokens() > self.max_tokens and iteration < max_iterations:
            iteration += 1

            # Strategy 1: Normal compression (if we have enough messages)
            if self.messages:
                self._compress_messages()
                continue

            # Strategy 2: Reduce recent message count if normal compression isn't enough
            if self.messages:  # Keep at least 2 messages
                self._reduce_recent_messages()
                continue

            # Strategy 3: Truncate individual messages if still too many tokens
            if self.messages:
                self._truncate_long_messages()
                continue

            # Strategy 4: Compress summary further if it's too large
            if self.summary:
                self._compress_summary_further()
                continue

            break

        if iteration >= max_iterations:
            print(
                f"Warning: Could not reduce tokens below limit after {max_iterations} attempts")

    def _compress_messages(self):
        """Compress messages based on token budget, not fixed count"""
        if not self.messages:
            return

        available_tokens = self.max_tokens - self.max_summary_tokens - 400

        cumulative_tokens = 0
        messages_to_keep = []
        messages_to_summarize = []

        for msg in reversed(self.messages):
            msg_tokens = self.estimate_tokens(msg["content"])

            if cumulative_tokens + msg_tokens <= available_tokens and len(messages_to_keep) < self.recent_messages_count:
                messages_to_keep.insert(0, msg)
                cumulative_tokens += msg_tokens
            else:
                messages_to_summarize.insert(0, msg)

        if messages_to_summarize:
            new_summary = self._create_summary(messages_to_summarize)

            if self.summary:
                combined_text = f"{self.summary}\n\nAdditional context: {new_summary}"
                if self.estimate_tokens(combined_text) > self.max_summary_tokens:
                    self.summary = self._create_meta_summary(
                        self.summary, new_summary)
                else:
                    self.summary = combined_text
            else:
                self.summary = new_summary

            if self.estimate_tokens(self.summary) > self.max_summary_tokens:
                self.summary = self._truncate_summary(self.summary)

            print(f"Compressed {len(messages_to_summarize)} messages into summary. "
                  f"Keeping {len(messages_to_keep)} recent messages.")

        self.messages = messages_to_keep

    def _reduce_recent_messages(self):
        """Reduce messages using pure token-aware selection"""
        if not self.messages:
            return

        summary_tokens = self.estimate_tokens(
            self.summary) if self.summary else 0
        available_tokens = self.max_tokens - summary_tokens - 400

        cumulative_tokens = 0
        messages_to_keep = []
        messages_to_summarize = []

        for msg in reversed(self.messages):
            msg_tokens = self.estimate_tokens(msg["content"])

            if cumulative_tokens + msg_tokens <= available_tokens and len(messages_to_keep) < self.recent_messages_count:
                messages_to_keep.insert(0, msg)
                cumulative_tokens += msg_tokens
            else:
                messages_to_summarize.insert(0, msg)

        if not messages_to_keep and self.messages:
            most_recent = self.messages[-1].copy()
            max_chars = available_tokens * 3

            if len(most_recent["content"]) > max_chars:
                most_recent["content"] = most_recent["content"][:max_chars -
                                                                20] + "... [truncated]"

            messages_to_keep = [most_recent]
            messages_to_summarize = self.messages[:-1]

            print(
                f"Truncated most recent message to fit within {available_tokens} token budget")

        if messages_to_summarize:
            new_summary = self._create_summary(messages_to_summarize)

            if self.summary:
                combined_text = f"{self.summary}\n\nAdditional context: {new_summary}"
                if self.estimate_tokens(combined_text) > self.max_summary_tokens:
                    self.summary = self._create_meta_summary(
                        self.summary, new_summary)
                else:
                    self.summary = combined_text
            else:
                self.summary = new_summary

            if self.estimate_tokens(self.summary) > self.max_summary_tokens:
                self.summary = self._truncate_summary(self.summary)

            self.messages = messages_to_keep

            removed_tokens = sum(self.estimate_tokens(
                msg["content"]) for msg in messages_to_summarize)
            kept_tokens = sum(self.estimate_tokens(
                msg["content"]) for msg in messages_to_keep)

            print(f"Reduced messages: moved {len(messages_to_summarize)} messages ({removed_tokens} tokens) to summary, "
                  f"keeping {len(messages_to_keep)} messages ({kept_tokens} tokens)")

    def _truncate_long_messages(self):
        """Truncate individual messages that are too long"""
        max_message_tokens = (
            self.max_tokens - self.max_summary_tokens) // max(len(self.messages), 1)

        for i, msg in enumerate(self.messages):
            msg_tokens = self.estimate_tokens(msg["content"])
            if msg_tokens > max_message_tokens:
                char_limit = max_message_tokens * 3
                if len(msg["content"]) > char_limit:
                    truncated_content = msg["content"][:char_limit -
                                                       50] + "... [truncated]"
                    self.messages[i]["content"] = truncated_content
                    print(
                        f"Truncated message {i} from {msg_tokens} to ~{max_message_tokens} tokens")

    def _compress_summary_further(self):
        """Compress the summary to be smaller"""
        if not self.summary:
            return

        target_tokens = self.max_summary_tokens // 2

        prompt = f"""Compress this summary to NO MORE than {target_tokens * 3} characters while keeping the most important information:
        {self.summary}
        Compressed summary (under {target_tokens * 3} characters):"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                compressed = response.content.strip()
            else:
                compressed = str(response).strip()

            if self.estimate_tokens(compressed) < self.estimate_tokens(self.summary):
                self.summary = compressed
                print(
                    f"Compressed summary further to {self.estimate_tokens(compressed)} tokens")
            else:
                self.summary = self._truncate_summary(
                    self.summary, target_tokens)
        except:
            self.summary = self._truncate_summary(self.summary, target_tokens)

    def _create_summary(self, messages):
        """Create summary of messages using LLM with size constraint"""
        if not messages:
            return ""

        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"

        max_chars = self.max_summary_tokens * 3
        summary_prompt = f"""Summarize the following conversation in NO MORE than {max_chars} characters. Focus on key decisions, important information, and progress made. Be concise but preserve important context:
        {conversation_text}
        Summary (keep under {max_chars} characters):"""

        try:
            summary_response = self.llm.invoke(summary_prompt)
            if hasattr(summary_response, 'content'):
                summary = summary_response.content.strip()
            else:
                summary = str(summary_response).strip()

            if self.estimate_tokens(summary) > self.max_summary_tokens:
                summary = self._truncate_summary(summary)

            return summary
        except Exception as e:
            print(f"Error creating summary: {e}")
            fallback = f"Previous conversation covered {len(messages)} exchanges."
            return self._truncate_summary(fallback)

    def _create_meta_summary(self, old_summary, new_summary):
        """Re-summarize when combined summary would be too long"""
        max_chars = self.max_summary_tokens * 3
        meta_prompt = f"""Combine these two summaries into one concise summary of NO MORE than {max_chars} characters:
        Previous Summary: {old_summary}
        Recent Summary: {new_summary}
        Combined Summary (under {max_chars} characters):"""

        try:
            response = self.llm.invoke(meta_prompt)
            if hasattr(response, 'content'):
                meta_summary = response.content.strip()
            else:
                meta_summary = str(response).strip()

            if self.estimate_tokens(meta_summary) > self.max_summary_tokens:
                meta_summary = self._truncate_summary(meta_summary)

            return meta_summary
        except Exception as e:
            print(f"Error creating meta summary: {e}")
            return self._truncate_summary(new_summary)

    def _truncate_summary(self, summary, max_tokens=None):
        """Truncate summary to fit within token limit"""
        target_tokens = max_tokens or self.max_summary_tokens

        if self.estimate_tokens(summary) <= target_tokens:
            return summary

        char_limit = target_tokens * 3
        if len(summary) > char_limit:
            truncated = summary[:char_limit-3] + "..."
            return truncated

        return summary

    def get_messages_for_llm(self, system_prompt=None):
        """Get messages formatted for LLM with summary context"""
        formatted_messages = []

        if system_prompt:
            formatted_messages.append(
                {"role": "system", "content": system_prompt})

        if self.summary:
            summary_content = f"Previous conversation summary: {self.summary}"
            formatted_messages.append(
                {"role": "system", "content": summary_content})

        formatted_messages.extend(self.messages)
        return formatted_messages

    def get_conversation_history_text(self):
        """Get formatted conversation history for templates"""
        if not self.messages:
            return ""

        history_text = ""
        if self.summary:
            history_text += f"Previous Context: {self.summary}\n\n"

        history_text += "Recent Conversations:\n"
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n\n"

        return history_text.strip()

    def clear(self):
        """Clear all messages and summary"""
        self.messages = []
        self.summary = ""

    def get_summary_stats(self):
        """Get statistics about current memory state"""
        return {
            "total_messages": len(self.messages),
            "summary_tokens": self.estimate_tokens(self.summary) if self.summary else 0,
            "message_tokens": sum(self.estimate_tokens(msg["content"]) for msg in self.messages),
            "total_tokens": self.get_total_tokens(),
            "summary_length": len(self.summary),
            "has_summary": bool(self.summary),
            "within_limit": self.get_total_tokens() <= self.max_tokens
        }


def prepare_prompt_within_budget(template, 
                                 replacements, 
                                 memory_obj, 
                                 max_tokens=16000):
    """
    Prepare a prompt while respecting token limits

    Args:
        template: The prompt template with placeholders
        replacements: Dict of key->value replacements
        memory_obj: The ConversationSummaryBuffer instance (for token estimation)
        max_tokens: Maximum allowed tokens
    """
    # Start with the template
    prompt = template

    # Track tokens as we build
    token_count = memory_obj.estimate_tokens(prompt)

    # Sort replacements by priority (most important first)
    priority_order = [
        "current_batch_messages",  # Most recent/relevant
        "ai_last_message",         # Latest response
        "current_plan",            # Current context
        "topic",                   # Core info
        "theme",                   # Core info
        "previous_plans_summary",  # Can be truncated
        "previous_batches"         # Can be truncated most
    ]

    for key in priority_order:
        if f"<{key}>" in prompt and key in replacements:
            value = replacements[key]
            value_tokens = memory_obj.estimate_tokens(value)

            # Check if adding this would exceed budget
            if token_count + value_tokens > max_tokens - 1000:  # Leave buffer
                # Handle based on priority
                if key in ["previous_batches", "previous_plans_summary"]:
                    # These can be truncated
                    available_tokens = max_tokens - token_count - 1000
                    if available_tokens > 100:
                        value = truncate_to_tokens(
                            value, available_tokens, memory_obj)
                    else:
                        value = "[Content omitted due to length]"

                elif key == "current_plan":
                    # Try to truncate but keep essential parts
                    available_tokens = max_tokens - token_count - 1000
                    if available_tokens > 500:
                        value = truncate_to_tokens(
                            value, available_tokens, memory_obj)
                    else:
                        # Extract just the key points
                        value = f"[Plan summary: {value[:200]}...]"

                elif key == "ai_last_message":
                    # Truncate but keep beginning and end
                    available_tokens = max_tokens - token_count - 1000
                    if available_tokens > 300:
                        value = truncate_smart(
                            value, available_tokens, memory_obj)
                    else:
                        value = f"{value[:100]}... [truncated] ...{value[-100:]}"

                elif key in ["topic", "theme"]:
                    # These are usually short, but if not, truncate
                    if value_tokens > 200:
                        value = value[:200] + "..."

                elif key == "current_batch_messages":
                    # This is critical - if it's from memory's conversation history,
                    # we might have access to the summary
                    available_tokens = max_tokens - token_count - 1000
                    if available_tokens > 1000:
                        value = "...[earlier messages truncated]...\n" + \
                            value[-(available_tokens*3):]
                    else:
                        value = "[Messages too long - see summary]"

            prompt = prompt.replace(f"<{key}>", value)
            token_count = memory_obj.estimate_tokens(prompt)

    # Final check
    if token_count > max_tokens:
        print(
            f"Warning: Final prompt still exceeds limit ({token_count} > {max_tokens})")

    return prompt


def truncate_to_tokens(text, 
                       max_tokens, 
                       memory_obj):
    """Truncate text to approximately max_tokens"""
    # Use the memory object's token estimation
    current_tokens = memory_obj.estimate_tokens(text)

    if current_tokens <= max_tokens:
        return text

    # Binary search for the right length
    left, right = 0, len(text)
    result = text

    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid] + "... [truncated]"
        tokens = memory_obj.estimate_tokens(truncated)

        if tokens <= max_tokens:
            result = truncated
            left = mid
        else:
            right = mid - 1

    return result


def truncate_smart(text, 
                   max_tokens, 
                   memory_obj):
    """Truncate text keeping beginning and end for context"""
    current_tokens = memory_obj.estimate_tokens(text)

    if current_tokens <= max_tokens:
        return text

    # Keep first 40% and last 40% of the token budget
    part_tokens = int(max_tokens * 0.4)

    # Find how many characters give us roughly part_tokens
    # This is approximate since we can't easily reverse the tokenization
    estimated_chars_per_token = len(text) / current_tokens
    chars_each_side = int(part_tokens * estimated_chars_per_token)

    return f"{text[:chars_each_side]}... [content truncated] ...{text[-chars_each_side:]}"


def generate_labels(topic: str, 
                    theme: str, 
                    domain: str) -> list[str]:
    
    if domain == "general":
        prompt = label_generation_prompt_template.format(topic, theme)
    elif domain == "coding":
        prompt = coding_label_generation_prompt_template.format(topic, theme)
    elif domain == "math":
        prompt = math_label_generation_prompt_template.format(topic, theme)

    response = llama_llm.invoke(prompt).content

    return response


def extract_labels(labels_text: str) -> List[Dict]:
    HEADER_RE = re.compile(
        r"""
        ^\*{0,2}              
        (?P<title>.+? Labels) 
        :\*{0,2}$          
        """,
        re.VERBOSE,
    )

    lines = [ln.strip() for ln in labels_text.splitlines() if ln.strip()]
    records, current = [], None

    for idx, ln in enumerate(lines):
        header = HEADER_RE.match(ln)
        if header:
            if current:
                records.append(current)
            current = {
                "category": header.group("title"),
                "description": "",
                "sublabels": [],
            }
            continue

        if ln.startswith("-") and current:
            bullet = ln.lstrip("- ").strip()

            next_is_bullet = (
                idx + 1 < len(lines) and lines[idx + 1].startswith("-")
            )

            if not current["description"] and next_is_bullet:
                current["description"] = bullet
            else:
                current["sublabels"] = [s.strip() for s in bullet.split(",")]

    if current:
        records.append(current)
    return records


def format_labels_for_llm(records):
    """Format labels in a clear, structured way for LLM processing"""
    formatted_labels = []

    for i, record in enumerate(records, 1):
        category = record['category'].replace(':', '').strip()
        description = record['description']
        sublabels = ', '.join(
            record['sublabels']) if record['sublabels'] else "General application"

        formatted_labels.append(
            f"{i}. LABEL CATEOGRY: {category}\n   LABEL DESCRIPTION: {description}\n   Focus Areas: {sublabels}")

    return "\n\n".join(formatted_labels)


def format_user_profile_for_llm(main_spec):
    """Format user profile in a clear, readable structure"""
    profile_text = f"""
    USER PROFILE:
    • Name: {main_spec['name']}
    • Age: {main_spec['age']} years old
    • Gender: {main_spec['gender'].title()}
    • Location: {main_spec['living location']}
    • Profession: {main_spec['job_title'].title()}

    PERSONALITY OVERVIEW:
    {main_spec['personality_traits']}
    """
    return profile_text.strip()


def format_relationships_for_llm(relationships):
    """Format relationships in a clear, hierarchical structure"""
    relationship_sections = []

    # Define relationship categories with descriptions
    relationship_types = {
        'parent': 'PARENTS & GUARDIANS',
        'partner': 'ROMANTIC PARTNER',
        'children': 'CHILDREN',
        'friends': 'CLOSE FRIENDS',
        'acquaintances': 'ACQUAINTANCES & COLLEAGUES'
    }

    for rel_type, section_title in relationship_types.items():
        if rel_type in relationships and relationships[rel_type]:
            people = []
            for person in relationships[rel_type]:
                if isinstance(person, dict) and 'name' in person:
                    people.append(
                        f"• {person['name']} ({person['gender']}, age {person['age']})")
                elif isinstance(person, (int, str)):
                    people.append(f"• [ID: {person}]")

            if people:
                relationship_sections.append(
                    f"{section_title}:\n" + "\n".join(people))

    return "\n\n".join(relationship_sections)


def get_unique_lines(raw: str, 
                     already: set[str]) -> list[str]:
    
    lines = [l.strip() for l in raw.splitlines() if l.strip()
             and not l.strip().startswith("###")]
    return [l for l in lines if l.lower() not in {s.lower() for s in already}]


def get_unique_messages(raw: str, 
                        already: set[str]) -> list[str]:
    """Extract messages using the separator specified in prompt"""

    # Remove completion marker
    content = re.sub(r'###\s*COMPLETE\s*###.*$', '', raw,
                     flags=re.IGNORECASE | re.DOTALL).strip()

    # Split by the separator
    messages = [msg.strip() for msg in content.split(
        '---MESSAGE_SEPARATOR---') if msg.strip()]

    # Filter duplicates
    unique_messages = []
    already_lower = {msg.lower() for msg in already}

    for msg in messages:
        if msg.lower() not in already_lower:
            unique_messages.append(msg)
            already_lower.add(msg.lower())

    return unique_messages


def extract_plan_bullets(plan_text: str) -> list[str]:
    """Extract individual bullets from plan text"""
    bullets = re.findall(
        r"• \*\*(.+?):\*\*(.+?)(?=• \*\*|\n|$)", plan_text, re.DOTALL)
    return [f"{label.strip()}: {content.strip()}" for label, content in bullets]


def distribute_bullets_across_batches(bullets: list[str], 
                                      num_sub_batches: int, 
                                      distrubtion_type: str) -> list[list[str]]:
    """Distribute bullets across sub-batches to ensure even coverage"""

    if distrubtion_type == "round-robin":
        sub_batches = [[] for _ in range(num_sub_batches)]

        for i, bullet in enumerate(bullets):
            sub_batches[i % num_sub_batches].append(bullet)

        return sub_batches
    elif distrubtion_type == "normal":
        n = len(bullets)
        q, r = divmod(n, num_sub_batches)
        batches = []
        start = 0
        for i in range(num_sub_batches):
            size = q + (1 if i < r else 0)
            batches.append(bullets[start: start + size])
            start += size
        return batches


# ================================ LABEL GENERATION ================================


def get_labels(topic: str, 
               theme: str, 
               domain: str) -> str:
    labels_text = generate_labels(topic=topic, theme=theme, domain=domain)
    labels = extract_labels(labels_text=labels_text)
    labels = format_labels_for_llm(labels)

    return labels

# ================================ PROFILE GENERATION ================================


def get_profile():
    main_spec, relationships = create_profile(
        all_profile_size=70, friends_size=5, acquaintances_size=5)
    main_spec = format_user_profile_for_llm(main_spec)
    relationships = format_relationships_for_llm(relationships)

    return main_spec, relationships


# ================================ PLAN GENERATION ================================


def _plan_batch_prompt(topic: str,
                       theme: str,
                       timeline: str,
                       batch_number: int,
                       num_batches: int,
                       num_bullets: int,
                       labels: str,
                       main_spec: str,
                       relationships: str,
                       previous_plans: list[str],
                       domain: str) -> str:
    """Build a bounded BEAM plan request that smaller hosted models can follow."""

    previous = "\n\n".join(previous_plans[-2:]) if previous_plans else "None"
    return f"""You are generating one batch of a BEAM long-conversation plan.

DOMAIN: {domain}
TOPIC: {topic}
THEME: {theme}
TIMELINE: {timeline}
BATCH: {batch_number} of {num_batches}
AVAILABLE LABELS:
{labels}
USER PROFILE:
{main_spec}
USER RELATIONSHIPS:
{relationships}
PREVIOUS BATCH PLANS:
{previous}

Return only `BATCH {batch_number} PLAN` followed by exactly {num_bullets}
bullet lines. Every line must use this exact parseable format:
• **Label:** detailed content

Labels may include BEAM's category and description separated by a colon when
appropriate, for example `• **Technical Problem-Solving:Debugging:** ...`.

The first bullet must be `• **Time Anchor:** Month DD, YYYY` and its date
must be later than prior batches. The remaining bullets must form a coherent,
progressive {domain} conversation plan. Pack every bullet with specific names,
numbers, dates, constraints, decisions, outcomes, and dependencies. Use the
profile consistently, avoid repetition, and make each bullet support a distinct
future user question. Do not write headings, numbered lists, commentary, or any
text after the {num_bullets}th bullet.
"""


def _parse_single_plan_batch(text: str,
                             batch_number: int,
                             num_bullets: int) -> str | None:
    header = re.search(
        rf"BATCH\s+{batch_number}\s+PLAN", text, flags=re.IGNORECASE
    )
    # Hosted models occasionally omit the redundant header or use `-` in
    # place of `•`. The downstream BEAM contract is the normalized bullet
    # structure, so accept those harmless formatting differences locally.
    body = text[header.end():] if header else text
    bullets = re.findall(
        r"^\s*(?:•|-)\s+\*\*(.+?):\*\*\s*(.+?)\s*$",
        body,
        flags=re.MULTILINE,
    )
    if len(bullets) < num_bullets:
        return None
    # Extra trailing bullets are not dependencies for downstream generation;
    # retain the requested chronological plan density exactly.
    bullets = bullets[:num_bullets]
    lines = [
        f"• **{label.strip()}:** {content.strip()}"
        for label, content in bullets
    ]
    if not lines[0].lower().startswith("• **time anchor:"):
        return None
    return f"BATCH {batch_number} PLAN\n" + "\n".join(lines)


def generate_plans(topic: str, 
                   theme: str, 
                   timeline: str, 
                   num_batches: int,
                   num_bullets: int, 
                   labels: str, 
                   main_spec: str, 
                   relationships: str, 
                   llm_name: str,
                   save_address: str, 
                   input_address: str = None, 
                   domain: str = "general") -> List[str]:

    if input_address:
        with open(input_address, 'rb') as f:
            plans = pickle.load(f)

        return plans

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    partial_address = f"{save_address}.partial.pickle"
    failed_attempts_address = f"{save_address}_failed_attempts.jsonl"
    plans = []
    raw_responses = []
    if os.path.isfile(partial_address):
        try:
            with open(partial_address, "rb") as handle:
                partial = pickle.load(handle)
            plans = partial.get("plans", [])
            raw_responses = partial.get("raw_responses", [])
            print(f"Resuming plan generation after {len(plans)} complete batches")
        except (OSError, EOFError, pickle.PickleError, AttributeError):
            plans = []
            raw_responses = []

    for batch_number in range(len(plans) + 1, num_batches + 1):
        plan = None
        last_response = ""
        for attempt in range(1, 7):
            prompt = _plan_batch_prompt(
                topic=topic,
                theme=theme,
                timeline=timeline,
                batch_number=batch_number,
                num_batches=num_batches,
                num_bullets=num_bullets,
                labels=labels,
                main_spec=main_spec,
                relationships=relationships,
                previous_plans=plans,
                domain=domain,
            )
            if attempt > 1:
                prompt += (
                    f"\nYour prior response did not contain exactly {num_bullets} "
                    "parseable bullet lines. Correct the structure exactly."
                )
            last_response = llm.invoke(prompt, max_tokens=4_000).content
            plan = _parse_single_plan_batch(
                last_response, batch_number, num_bullets
            )
            if plan is not None:
                break
            with open(failed_attempts_address, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "batch_number": batch_number,
                    "attempt": attempt,
                    "response": last_response,
                }))
                handle.write("\n")
        if plan is None:
            raise RuntimeError(
                f"Could not generate valid plan batch {batch_number}/{num_batches} "
                f"after 6 attempts; inspect {failed_attempts_address}"
            )
        plans.append(plan)
        raw_responses.append(last_response)
        _atomic_pickle_dump(
            {"plans": plans, "raw_responses": raw_responses},
            partial_address,
        )

    with open(f'{save_address}.pickle', 'wb') as f:
        pickle.dump(plans, f)

    with open(f'{save_address}.txt', 'w', encoding='utf-8') as f:
        f.write("\n\n".join(raw_responses))

    if os.path.exists(partial_address):
        os.remove(partial_address)

    return plans


def generate_detailed_plans(topic: str, 
                            theme: str, 
                            timeline: str, 
                            num_messages: int, 
                            batch_size: int, 
                            num_batches: int,
                            labels: str, 
                            main_spec: str, 
                            relationships: str, 
                            llm_name: str, 
                            save_address: str, 
                            input_address: str = None, 
                            domain: str = "general") -> List[str]:

    plans = []
    new_plans = []
    original_plan = ""
    if input_address:
        with open(input_address, 'rb') as f:
            plans = pickle.load(f)
    else:
        if domain == "general":
            plan_generation_prompt = plan_generation_prompt_profile_and_topic_given_template\
                .replace("<topic>", topic)\
                .replace("<theme>", theme)\
                .replace("<timeline>", timeline)\
                .replace("<total_messages>", str(num_messages))\
                .replace("<batch_size>", str(batch_size))\
                .replace("<num_batches>", str(num_batches))\
                .replace("<provided_labels>", str(labels))\
                .replace("<user_profile>", str(main_spec))\
                .replace("<user_relationships>", str(relationships))

        elif domain == "coding":
            plan_generation_prompt = coding_plan_generation_prompt\
                .replace("<topic>", topic)\
                .replace("<theme>", theme)\
                .replace("<timeline>", timeline)\
                .replace("<total_messages>", str(num_messages))\
                .replace("<batch_size>", str(batch_size))\
                .replace("<num_batches>", str(num_batches))\
                .replace("<provided_labels>", str(labels))\
                .replace("<user_profile>", str(main_spec))

        elif domain == "math":
            plan_generation_prompt = math_plan_generation_prompt\
                .replace("<topic>", topic)\
                .replace("<theme>", theme)\
                .replace("<timeline>", timeline)\
                .replace("<total_messages>", str(num_messages))\
                .replace("<batch_size>", str(batch_size))\
                .replace("<num_batches>", str(num_batches))\
                .replace("<provided_labels>", str(labels))\
                .replace("<user_profile>", str(main_spec))

        if llm_name == "gpt":
            llm = gpt_llm
        elif llm_name == "qwen":
            llm = qwen_llm
        elif llm_name == "llama":
            llm = llama_llm

        plan_response = llm.invoke(plan_generation_prompt)
        plan_response = plan_response.content

        raw_batches = re.split(r'(BATCH \d+ PLAN)', plan_response)

        for i in range(1, len(raw_batches), 2):
            header = raw_batches[i].strip()
            content = raw_batches[i + 1].strip()
            plans.append(f"{header}\n{content}")
            original_plan += f"{header}\n{content}"

    for plan in plans:
        original_plan += plan + "\n\n"

    if domain == "general":
        plan_generation_prompt = convert_broad_plan_to_detailed_plan_prompt\
            .replace("<original_plan>", original_plan)\
            .replace("<topic>", topic)\
            .replace("<theme>", theme)\
            .replace("<timeline>", timeline)\
            .replace("<user_profile>", str(main_spec))\
            .replace("<user_relationships>", str(relationships))\

        if llm_name == "gpt":
            llm = gpt_llm
        elif llm_name == "qwen":
            llm = qwen_llm
        elif llm_name == "llama":
            llm = llama_llm

        plan_response = llm.invoke(plan_generation_prompt)
        plan_response = plan_response.content

        raw_batches = re.split(r'(BATCH \d+ PLAN)', plan_response)

        for i in range(1, len(raw_batches), 2):
            header = raw_batches[i].strip()
            content = raw_batches[i + 1].strip()
            new_plans.append(f"{header}\n{content}")

        with open(f'{save_address}.pickle', 'wb') as f:
            pickle.dump(plans, f)

        with open(f'{save_address}.txt', 'w', encoding='utf-8') as f:
            f.writelines(plan_response)

# ================================ PLAN REVISION ================================


def plan_revision(topic: str, 
                  theme: str, 
                  plans: list, 
                  llm_name: str, 
                  save_address: str) -> List[str]:
    
    plan_refinement_prompt = plan_revision_prompt_template\
        .replace("<topic>", topic)\
        .replace("<theme>", theme)\
        .replace("<generated_plans>", "\n\n".join(plans))

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    new_plan = llm.invoke(plan_refinement_prompt).content

    raw_batches = re.split(r'(BATCH \d+ PLAN)', new_plan)

    plans = []
    for i in range(1, len(raw_batches), 2):
        header = raw_batches[i].strip()
        content = raw_batches[i + 1].strip()
        plans.append(f"{header}\n{content}")

    with open(f'{save_address}.pickle', 'wb') as f:
        pickle.dump(plans, f)

    with open(f'{save_address}.txt', 'w', encoding='utf-8') as f:
        f.writelines(new_plan)

    return plans

# ================================ PLAN SPECIAL BULLETS ================================


def add_special_bullets_to_plan(plan_address: str,
                                num_bullets: int,
                                llm_name: str) -> None:

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    with open(plan_address+".pickle", 'rb') as f:
        plan = pickle.load(f)

    completed_plans = []
    for batch_number, batch_plan in enumerate(plan, start=1):
        response = ""
        special = []
        for attempt in range(1, 5):
            prompt = f"""Add BEAM's three memory-test facts to this batch plan.

PLAN:
{batch_plan}

Return exactly three lines and nothing else, in this order and format:
• **Information Update:** naturally change one explicit number, date, status, or decision from the plan
• **User Instruction:** Always [specific action] when I ask about [specific condition]
• **Logical Contradiction:** state a natural absolute fact that makes one completed event in the plan impossible

The update and contradiction must each be grounded in an identifiable earlier
bullet. Do not reproduce the plan. Do not number the lines.
"""
            if attempt > 1:
                prompt += "\nCorrect the prior formatting error; return exactly three bullet lines."
            response = llm.invoke(prompt, max_tokens=700).content
            special = re.findall(
                r"^\s*•\s+\*\*(Information Update|User Instruction|Logical Contradiction):\*\*\s*(.+?)\s*$",
                response,
                flags=re.MULTILINE,
            )
            if [label for label, _ in special] == [
                "Information Update",
                "User Instruction",
                "Logical Contradiction",
            ]:
                break
        if len(special) != 3:
            raise RuntimeError(
                f"Could not generate the three special bullets for batch {batch_number}"
            )
        special_lines = [
            f"• **{label}:** {content.strip()}" for label, content in special
        ]
        completed_plans.append(batch_plan.rstrip() + "\n" + "\n".join(special_lines))

    with open(plan_address+"_new.txt", 'w', encoding='utf-8') as f:
        f.write("\n\n".join(completed_plans))

    with open(plan_address+"_new.pickle", 'wb') as f:
        pickle.dump(completed_plans, f)

# ================================ USER MESSAGES GENERATION ================================


def estimate_token_count(text):
    return int(len(text) / 3.7)


def trim_previous_plans(previous_plans, 
                        max_tokens=4000):
    """
    Trim previous plans to stay within token limit.
    Keeps most recent plans and adds summary of removed ones.
    """
    if not previous_plans:
        return previous_plans

    batch_plans = previous_plans.split("BATCH")
    batch_plans = [f"BATCH{plan}" for plan in batch_plans if plan.strip()]

    if not batch_plans:
        return previous_plans

    kept_plans = []
    current_tokens = 0

    for plan in reversed(batch_plans):
        plan_tokens = estimate_token_count(plan)
        if current_tokens + plan_tokens < max_tokens:
            kept_plans.insert(0, plan)
            current_tokens += plan_tokens
        else:
            break

    if len(kept_plans) < len(batch_plans):
        removed_count = len(batch_plans) - len(kept_plans)
        summary = f"[Previous {removed_count} batch plans omitted for context length. Showing most recent {len(kept_plans)} plans.]\n\n"
        return summary + "\n".join(kept_plans)

    return previous_plans


def user_messages_generation(topic: str, 
                             theme: str, 
                             plans: list, 
                             num_batches: int,
                             sub_batches_per_batch: int, 
                             sub_batch_size: int, 
                             batch_size: int, 
                             llm_name: str, 
                             save_address: str, 
                             domain: str, 
                             sepcial_bullets: bool = False) -> None:
    all_messages = []
    global_history = set()

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    temp_sub_batches_per_batch = sub_batches_per_batch

    for batch_idx in range(num_batches):
        print(f"Processing Batch {batch_idx + 1}/{num_batches}")

        # Get context
        past_plans = "\n\n".join(plans[:batch_idx][::-1])
        cur_plan = plans[batch_idx]

        # Extract bullets from current plan
        plan_bullets = extract_plan_bullets(cur_plan)

        time_anchor = extract_time_anchor(plan_text=plan_bullets[0])

        sepcial_bullets_list = []
        if sepcial_bullets:
            sepcial_bullets_list = plan_bullets[-3:]
            plan_bullets = plan_bullets[:-3]

        sub_batches_per_batch = temp_sub_batches_per_batch
        # Distribute bullets across sub-batches
        bullet_groups = distribute_bullets_across_batches(
            plan_bullets, sub_batches_per_batch, distrubtion_type="normal")

        if sepcial_bullets:
            bullet_groups.append(sepcial_bullets_list)
            sub_batches_per_batch += 1

        batch_lines = []
        batch_history = []

        for sub_batch_idx in range(sub_batches_per_batch):
            print(f"  Sub-batch {sub_batch_idx + 1}/{sub_batches_per_batch}")

            focused_bullets = "\n".join(
                f"{index+1}) {bullet}" for index, bullet in enumerate(bullet_groups[sub_batch_idx]))
            batch_history_text = "\n".join(
                batch_history) if batch_history else "(none)"

            previous_sub_batch_plans = ""
            for bullets in bullet_groups[:sub_batch_idx]:
                previous_sub_batch_plans += "\n".join(
                    f"• {bullet}" for bullet in bullets) + "\n\n"

            if sepcial_bullets and (sub_batch_idx == sub_batches_per_batch - 1):
                # if "Logical Contradiction:" in bullet_groups[sub_batch_idx][2]:
                #     temp_contradiction = bullet_groups[sub_batch_idx][2].split("Logical Contradiction:")[1].strip()
                temp_sub_batch_size = 2
                temp_focused_bullets = bullet_groups[sub_batch_idx][0] + \
                    "\n\n" + bullet_groups[sub_batch_idx][2]
                temp_instruction = bullet_groups[sub_batch_idx][1]

            else:
                temp_sub_batch_size = sub_batch_size

            if domain == "general":
                if temp_sub_batch_size == 2 and (sub_batch_idx == sub_batches_per_batch - 1):
                    system_prompt = message_generation_prompt_focused_template_special\
                        .replace("<TOPIC>", topic)\
                        .replace("<THEME>", theme)\
                        .replace("<FOCUSED_BULLETS>", temp_focused_bullets)\
                        .replace("<PREVIOUS_SUB_BATCH_PLANS>", "none" if previous_sub_batch_plans else "(none)")\
                        .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                        .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")
                else:
                    system_prompt = message_generation_prompt_focused_template\
                        .replace("<TOPIC>", topic)\
                        .replace("<THEME>", theme)\
                        .replace("<FOCUSED_BULLETS>", focused_bullets)\
                        .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_sub_batch_plans if previous_sub_batch_plans else "(none)")\
                        .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                        .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")

            elif domain == "coding":
                system_prompt = coding_message_generation_prompt\
                    .replace("<TOPIC>", topic)\
                    .replace("<THEME>", theme)\
                    .replace("<FOCUSED_BULLETS>", focused_bullets)\
                    .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_sub_batch_plans if previous_sub_batch_plans else "(none)")\
                    .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                    .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")

            elif domain == "math":
                system_prompt = math_message_generation_prompt\
                    .replace("<TOPIC>", topic)\
                    .replace("<THEME>", theme)\
                    .replace("<FOCUSED_BULLETS>", focused_bullets)\
                    .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_sub_batch_plans if previous_sub_batch_plans else "(none)")\
                    .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                    .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")

            sub_batch_lines = []
            retries = 0

            while len(sub_batch_lines) < temp_sub_batch_size and retries < 3:
                messages = [
                    SystemMessage(content=system_prompt),
                ]

                try:
                    if retries > 0:
                        system_prompt.replace(
                            "<BATCH_HISTORY>", "\n\n".join(sub_batch_lines))
                        messages = [
                            SystemMessage(content=system_prompt),
                        ]

                    raw = llm.invoke(messages, max_tokens=6000).content

                    if domain == "general":
                        new_lines = get_unique_lines(
                            raw, global_history | set(batch_lines))
                    elif domain == "coding" or domain == "math":
                        new_lines = get_unique_messages(
                            raw, global_history | set(batch_lines))

                    # Additional filtering for this sub-batch to avoid immediate repetition
                    new_lines = [line for line in new_lines if line.lower() not in {
                        l.lower() for l in sub_batch_lines}]

                    sub_batch_lines.extend(
                        new_lines[:temp_sub_batch_size - len(sub_batch_lines)])
                    retries += 1

                except Exception as e:
                    print(f"Error in sub-batch {sub_batch_idx}: {e}")
                    retries += 1

            if temp_sub_batch_size == 2 and (sub_batch_idx == sub_batches_per_batch - 1):
                if "User Instruction:" in temp_instruction:
                    temp_instruction = temp_instruction.split(
                        "User Instruction:")[1].strip()

                sub_batch_lines.insert(1, temp_instruction)

            questions = ""
            for index, question in enumerate(sub_batch_lines):
                questions += f"{index+1}) {question} \n\n"

            prompt = f"""Match each question to its corresponding bulletpoint.
                    Questions:
                    {questions}

                    Bulletpoints:
                    {focused_bullets}

                    Instructions:
                    - Process each question from Q1 to Q{len(sub_batch_lines)} in order
                    - For each question, write: [question text] ->-> [bulletpoint number or N/A]
                    - Use exactly ONE number after ->->
                    - If a question doesn't match any bulletpoint, use N/A
                    - Separate questions with: ---MESSAGE_SEPARATOR---
                    - Do NOT add separator after the last question
                    - List each question only ONCE
                    - Stop after processing all questions

                    Output format example:
                    question one ->-> 1 ---MESSAGE_SEPARATOR--- question two ->-> N/A ---MESSAGE_SEPARATOR--- question three ->-> 2

                    Begin matching:"""

            response = llm.invoke(prompt).content

            batch_number = batch_idx+1
            previous_bullet_num = 0
            for i in range(sub_batch_idx):
                previous_bullet_num += len(bullet_groups[i])

            sub_batch_lines = []
            for question in response.split("---MESSAGE_SEPARATOR---"):
                if "->->" in question:
                    question_temp = question.split("->->")[0].strip()
                    bullet_num = question.split("->->")[1].strip()
                    if bullet_num == "N/A":
                        new_question = question_temp + \
                            " ->-> " + f"{batch_number},N/A"
                    else:
                        if bullet_num.isdigit():
                            new_question = question_temp + " ->-> " + \
                                f"{batch_number},{previous_bullet_num+int(bullet_num)}"
                        else:
                            new_question = question + \
                                " ->-> " + f"{batch_number},N/A"

                    sub_batch_lines.append(new_question)
                elif question.strip() != "":
                    new_question = question + " ->-> " + f"{batch_number},N/A"
                    sub_batch_lines.append(new_question)

            temp_sub_batch_lines = []
            for question in sub_batch_lines:
                formatted_question = question
                if ")" in question:
                    formatted_question = question.split(") ", 1)[1]
                temp_sub_batch_lines.append(formatted_question)

            # Add successful lines to main batch
            batch_lines.extend(temp_sub_batch_lines)
            batch_history.extend(temp_sub_batch_lines)

            print(
                f"  Generated {len(temp_sub_batch_lines)} lines for sub-batch {sub_batch_idx + 1}")

        # Final cleanup and deduplication for the entire batch
        final_batch_lines = []
        seen_lower = set()

        for line in batch_lines:
            line_lower = line.lower()
            # More sophisticated similarity check
            is_similar = any(
                len(set(line_lower.split()) & set(existing.split())) /
                max(len(line_lower.split()), len(existing.split())) > 0.7
                for existing in seen_lower
            )

            if not is_similar:
                final_batch_lines.append(line)
                seen_lower.add(line_lower)

        final_object = {
            "time_anchor": time_anchor,
            "messages": [final_batch_lines]
        }
        # Ensure we don't exceed batch size
        all_messages.append(final_object)
        global_history.update(l.lower()
                              for l in final_batch_lines)

        print(
            f"Batch {batch_idx + 1} complete: {len(final_batch_lines)} messages")

    # Save results
    with open(save_address, 'wb') as f:
        pickle.dump(all_messages, f)

    print(
        f"Generation complete. Total messages: {sum(len(batch) for batch in all_messages)}")


def user_messages_generation_fast(topic: str, 
                                  theme: str, 
                                  plans: list,
                                  num_batches: int,
                                  sub_batches_per_batch: int, 
                                  sub_batch_size: int, 
                                  batch_size: int, 
                                  llm_name: str, 
                                  save_address: str, 
                                  domain: str, 
                                  sepcial_bullets: bool = False) -> None:
    partial_address = save_address.replace(".pickle", ".partial.pickle")
    all_messages = []
    global_history = set()

    if os.path.isfile(partial_address):
        with open(partial_address, "rb") as f:
            all_messages = pickle.load(f)
        global_history.update(
            line.lower()
            for batch in all_messages
            for line in batch.get("messages", [[]])[0]
        )
        print(f"Resuming question generation after {len(all_messages)} complete batches")

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    temp_sub_batches_per_batch = sub_batches_per_batch

    for batch_idx in range(len(all_messages), num_batches):
        print(f"Processing Batch {batch_idx + 1}/{num_batches}")

        # Get context
        past_plans = "\n\n".join(plans[:batch_idx][::-1])
        cur_plan = plans[batch_idx]

        # Extract bullets from current plan
        plan_bullets = extract_plan_bullets(cur_plan)
    
        time_anchor = extract_time_anchor(plan_text=plan_bullets[0])

        sepcial_bullets_list = []
        if sepcial_bullets:
            sepcial_bullets_list = plan_bullets[-3:]
            plan_bullets = plan_bullets[:-3]

        sub_batches_per_batch = temp_sub_batches_per_batch
        # Distribute bullets across sub-batches
        bullet_groups = distribute_bullets_across_batches(
            plan_bullets, sub_batches_per_batch, distrubtion_type="normal")

        if sepcial_bullets:
            bullet_groups.append(sepcial_bullets_list)
            sub_batches_per_batch += 1

        batch_lines = []
        batch_history = []

        for sub_batch_idx in range(sub_batches_per_batch):
            print(f"  Sub-batch {sub_batch_idx + 1}/{sub_batches_per_batch}")

            batch_number = batch_idx+1
            previous_bullet_num = 0
            for i in range(sub_batch_idx):
                previous_bullet_num += len(bullet_groups[i])

            focused_bullets = "\n".join(
                f"{previous_bullet_num+index+1}) {bullet}" for index, bullet in enumerate(bullet_groups[sub_batch_idx]))
            batch_history_text = "\n".join(
                batch_history) if batch_history else "(none)"

            previous_sub_batch_plans = ""
            for bullets in bullet_groups[:sub_batch_idx]:
                previous_sub_batch_plans += "\n".join(
                    f"• {bullet}" for bullet in bullets) + "\n\n"

            if sepcial_bullets and (sub_batch_idx == sub_batches_per_batch - 1):
                # if "Logical Contradiction:" in bullet_groups[sub_batch_idx][2]:
                #     temp_contradiction = bullet_groups[sub_batch_idx][2].split("Logical Contradiction:")[1].strip()
                temp_sub_batch_size = 2
                temp_focused_bullets = f"{previous_bullet_num+1}) " + bullet_groups[sub_batch_idx][0] + \
                    "\n\n" + f"{previous_bullet_num+3}) " + \
                    bullet_groups[sub_batch_idx][2]
                temp_instruction = bullet_groups[sub_batch_idx][1] + \
                    f" ->-> {previous_bullet_num+2}"

            else:
                temp_sub_batch_size = sub_batch_size

            if domain == "general":
                if temp_sub_batch_size == 2 and (sub_batch_idx == sub_batches_per_batch - 1):
                    base_prompt = message_generation_prompt_focused_template_fast_special
                    focused_bullets_to_use = temp_focused_bullets
                else:
                    base_prompt = message_generation_prompt_focused_template_fast
                    focused_bullets_to_use = focused_bullets

                total_context = (
                    len(base_prompt) +
                    len(topic) +
                    len(theme) +
                    len(focused_bullets_to_use) +
                    len(previous_sub_batch_plans or "") +
                    len(past_plans or "") +
                    len(str(temp_sub_batch_size))
                )

                total_tokens = total_context // 3.7
                if total_tokens > 12500:
                    past_plans = trim_previous_plans(
                        past_plans, max_tokens=4000)

                    total_context_after_trim = (
                        len(base_prompt) +
                        len(topic) +
                        len(theme) +
                        len(focused_bullets_to_use) +
                        len(previous_sub_batch_plans or "") +
                        len(past_plans or "") +
                        len(str(temp_sub_batch_size))
                    )

                    total_tokens = total_context_after_trim // 3.7
                    if total_tokens > 12500:
                        past_plans = trim_previous_plans(
                            past_plans, max_tokens=2000)
                        if previous_sub_batch_plans and len(previous_sub_batch_plans) > 8000:
                            previous_sub_batch_plans = "...[Earlier sub-batches trimmed]\n" + \
                                previous_sub_batch_plans[-8000:]

                if temp_sub_batch_size == 2 and (sub_batch_idx == sub_batches_per_batch - 1):
                    system_prompt = message_generation_prompt_focused_template_fast_special\
                        .replace("<TOPIC>", topic)\
                        .replace("<THEME>", theme)\
                        .replace("<FOCUSED_BULLETS>", temp_focused_bullets)\
                        .replace("<PREVIOUS_SUB_BATCH_PLANS>", "none" if previous_sub_batch_plans else "(none)")\
                        .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                        .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")
                else:
                    system_prompt = message_generation_prompt_focused_template_fast\
                        .replace("<TOPIC>", topic)\
                        .replace("<THEME>", theme)\
                        .replace("<FOCUSED_BULLETS>", focused_bullets)\
                        .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_sub_batch_plans if previous_sub_batch_plans else "(none)")\
                        .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                        .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")

            elif domain == "coding":
                total_context = (
                    len(coding_message_generation_prompt_fast) +
                    len(topic) +
                    len(theme) +
                    len(focused_bullets) +
                    len(previous_sub_batch_plans or "") +
                    len(past_plans or "") +
                    len(str(temp_sub_batch_size))
                )

                total_tokens = total_context // 3.7
                if total_tokens > 12500:
                    past_plans = trim_previous_plans(
                        past_plans, max_tokens=4000)

                system_prompt = coding_message_generation_prompt_fast\
                    .replace("<TOPIC>", topic)\
                    .replace("<THEME>", theme)\
                    .replace("<FOCUSED_BULLETS>", focused_bullets)\
                    .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_sub_batch_plans if previous_sub_batch_plans else "(none)")\
                    .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                    .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")

            elif domain == "math":
                total_context = (
                    len(math_message_generation_prompt_fast) +
                    len(topic) +
                    len(theme) +
                    len(focused_bullets) +
                    len(previous_sub_batch_plans or "") +
                    len(past_plans or "") +
                    len(str(temp_sub_batch_size))
                )

                total_tokens = total_context // 3.7
                if total_tokens > 12500:
                    past_plans = trim_previous_plans(
                        past_plans, max_tokens=4000)

                system_prompt = math_message_generation_prompt_fast\
                    .replace("<TOPIC>", topic)\
                    .replace("<THEME>", theme)\
                    .replace("<FOCUSED_BULLETS>", focused_bullets)\
                    .replace("<PREVIOUS_SUB_BATCH_PLANS>", previous_sub_batch_plans if previous_sub_batch_plans else "(none)")\
                    .replace("<SUB_BATCH_SIZE>", str(temp_sub_batch_size))\
                    .replace("<PREVIOUS_BATCH_PLANS>", past_plans if past_plans else "(none)")

            sub_batch_lines = []
            retries = 0

            while len(sub_batch_lines) < temp_sub_batch_size and retries < 3:
                messages = [
                    SystemMessage(content=system_prompt),
                ]

                try:
                    if retries > 0:
                        if estimate_token_count(system_prompt + "\n\n".join(sub_batch_lines)) < 12800:

                            system_prompt = system_prompt.replace(
                                "<BATCH_HISTORY>", "\n\n".join(sub_batch_lines))
                            messages = [
                                SystemMessage(content=system_prompt),
                            ]

                    tokens_per_message = {
                        "coding": 1_600,
                        "math": 1_000,
                        "general": 800,
                    }[domain]
                    raw = llm.invoke(
                        messages,
                        max_tokens=max(600, temp_sub_batch_size * tokens_per_message),
                    ).content

                    if domain == "general":
                        new_lines = get_unique_lines(
                            raw, global_history | set(batch_lines))
                    elif domain == "coding" or domain == "math":
                        new_lines = get_unique_messages(
                            raw, global_history | set(batch_lines))

                    # Additional filtering for this sub-batch to avoid immediate repetition
                    new_lines = [line for line in new_lines if line.lower() not in {
                        l.lower() for l in sub_batch_lines}]

                    sub_batch_lines.extend(
                        new_lines[:temp_sub_batch_size - len(sub_batch_lines)])
                    retries += 1

                except Exception as e:
                    print(f"Error in sub-batch {sub_batch_idx}: {e}")
                    error_file_address = save_address.split(".pickle")[
                        0] + ".txt"
                    with open(error_file_address, "a") as f:
                        f.write(f"{e}\n")
                    retries += 1

            if len(sub_batch_lines) < temp_sub_batch_size:
                raise RuntimeError(
                    f"Batch {batch_idx + 1} sub-batch {sub_batch_idx + 1} "
                    f"produced {len(sub_batch_lines)}/{temp_sub_batch_size} questions"
                )

            if temp_sub_batch_size == 2 and (sub_batch_idx == sub_batches_per_batch - 1):
                temp_instruction = temp_instruction.split(
                    "User Instruction:")[1].strip()
                sub_batch_lines.insert(1, temp_instruction)

            temp_sub_batch_lines = []
            for question in sub_batch_lines:
                formatted_question = question
                formatted_question = re.sub(r'^\d+\)\s*', '', question)
                temp_sub_batch_lines.append(formatted_question)

            final_sub_batch_lines = []
            for question in temp_sub_batch_lines:
                if "->->" in question:
                    bullet_number = question.split("->->")[1].strip()
                    if len(bullet_number) > 3:
                        print()
                    index = f"{batch_idx+1},{bullet_number}"
                    temp_question = question.split(
                        "->->")[0].strip() + " ->-> " + index
                else:
                    index = f"{batch_idx+1},N/A"
                    temp_question = question + " ->-> " + index

                final_sub_batch_lines.append(temp_question)

            # Add successful lines to main batch
            batch_lines.extend(final_sub_batch_lines)
            batch_history.extend(final_sub_batch_lines)

            print(
                f"  Generated {len(final_sub_batch_lines)} lines for sub-batch {sub_batch_idx + 1}")

        # Final cleanup and deduplication for the entire batch
        final_batch_lines = []
        seen_lower = set()

        for line in batch_lines:
            line_lower = line.lower()
            # More sophisticated similarity check
            is_similar = any(
                len(set(line_lower.split()) & set(existing.split())) /
                max(len(line_lower.split()), len(existing.split())) > 0.7
                for existing in seen_lower
            )

            if not is_similar:
                final_batch_lines.append(line)
                seen_lower.add(line_lower)

        # Ensure we don't exceed batch size
        final_object = {
            "time_anchor": time_anchor,
            "messages": [final_batch_lines]
        }
        all_messages.append(final_object)
        global_history.update(l.lower()
                              for l in final_batch_lines)

        with open(partial_address, "wb") as f:
            pickle.dump(all_messages, f)

        print(
            f"Batch {batch_idx + 1} complete: {len(final_batch_lines)} messages")

    # Save results
    with open(save_address, 'wb') as f:
        pickle.dump(all_messages, f)

    if os.path.exists(partial_address):
        os.remove(partial_address)

    print(
        f"Generation complete. Total messages: {sum(len(batch) for batch in all_messages)}")


def _question_generation_llm(llm_name: str):
    """Resolve the legacy model slot used by BEAM question generation."""

    if llm_name == "gpt":
        return gpt_llm
    if llm_name == "qwen":
        return qwen_llm
    if llm_name == "llama":
        return llama_llm
    raise ValueError(f"Unknown llm_name: {llm_name}")


def _build_parallel_question_tasks(topic: str,
                                   theme: str,
                                   plans: list,
                                   num_batches: int,
                                   sub_batches_per_batch: int,
                                   sub_batch_size: int,
                                   domain: str,
                                   sepcial_bullets: bool) -> tuple[list[dict], dict[int, str]]:
    """Build immutable prompts for every independent BEAM question draft.

    The prompts contain chronological plan context, but no generated dialogue.
    Once the plans are fixed, every task in this list is therefore independent.
    """

    tasks = []
    time_anchors = {}
    for batch_idx in range(num_batches):
        past_plans = "\n\n".join(plans[:batch_idx][::-1])
        plan_bullets = extract_plan_bullets(plans[batch_idx])
        if not plan_bullets:
            raise RuntimeError(f"Plan batch {batch_idx + 1} has no parseable bullets")
        time_anchors[batch_idx] = extract_time_anchor(plan_text=plan_bullets[0])

        special_bullets = []
        if sepcial_bullets:
            if len(plan_bullets) < 4:
                raise RuntimeError(
                    f"Plan batch {batch_idx + 1} is missing BEAM special bullets"
                )
            special_bullets = plan_bullets[-3:]
            plan_bullets = plan_bullets[:-3]

        bullet_groups = distribute_bullets_across_batches(
            plan_bullets,
            sub_batches_per_batch,
            distrubtion_type="normal",
        )
        if sepcial_bullets:
            bullet_groups.append(special_bullets)

        previous_bullet_num = 0
        for sub_batch_idx, bullet_group in enumerate(bullet_groups):
            is_special = sepcial_bullets and sub_batch_idx == len(bullet_groups) - 1
            focused_bullets = "\n".join(
                f"{previous_bullet_num + index + 1}) {bullet}"
                for index, bullet in enumerate(bullet_group)
            )
            previous_sub_batch_plans = "".join(
                "\n".join(f"• {bullet}" for bullet in prior_group) + "\n\n"
                for prior_group in bullet_groups[:sub_batch_idx]
            )

            expected = 2 if is_special else sub_batch_size
            instruction = None
            if is_special:
                special_focus = (
                    f"{previous_bullet_num + 1}) {bullet_group[0]}\n\n"
                    f"{previous_bullet_num + 3}) {bullet_group[2]}"
                )
                instruction = (
                    f"{bullet_group[1]} ->-> {previous_bullet_num + 2}"
                )
            else:
                special_focus = focused_bullets

            task_past_plans = past_plans
            task_previous_plans = previous_sub_batch_plans
            if domain == "general":
                template = (
                    message_generation_prompt_focused_template_fast_special
                    if is_special
                    else message_generation_prompt_focused_template_fast
                )
                focus = special_focus if is_special else focused_bullets
                total_tokens = (
                    len(template)
                    + len(topic)
                    + len(theme)
                    + len(focus)
                    + len(task_previous_plans)
                    + len(task_past_plans)
                    + len(str(expected))
                ) // 3.7
                if total_tokens > 12_500:
                    task_past_plans = trim_previous_plans(
                        task_past_plans, max_tokens=4_000
                    )
                    total_tokens = (
                        len(template)
                        + len(topic)
                        + len(theme)
                        + len(focus)
                        + len(task_previous_plans)
                        + len(task_past_plans)
                        + len(str(expected))
                    ) // 3.7
                    if total_tokens > 12_500:
                        task_past_plans = trim_previous_plans(
                            task_past_plans, max_tokens=2_000
                        )
                        if len(task_previous_plans) > 8_000:
                            task_previous_plans = (
                                "...[Earlier sub-batches trimmed]\n"
                                + task_previous_plans[-8_000:]
                            )
            elif domain == "coding":
                template = coding_message_generation_prompt_fast
                # The persistent instruction is inserted verbatim below; the
                # model only drafts the update and contradiction messages.
                focus = special_focus if is_special else focused_bullets
                total_tokens = (
                    len(template)
                    + len(topic)
                    + len(theme)
                    + len(focus)
                    + len(task_previous_plans)
                    + len(task_past_plans)
                    + len(str(expected))
                ) // 3.7
                if total_tokens > 12_500:
                    task_past_plans = trim_previous_plans(
                        task_past_plans, max_tokens=4_000
                    )
            elif domain == "math":
                template = math_message_generation_prompt_fast
                focus = special_focus if is_special else focused_bullets
                total_tokens = (
                    len(template)
                    + len(topic)
                    + len(theme)
                    + len(focus)
                    + len(task_previous_plans)
                    + len(task_past_plans)
                    + len(str(expected))
                ) // 3.7
                if total_tokens > 12_500:
                    task_past_plans = trim_previous_plans(
                        task_past_plans, max_tokens=4_000
                    )
            else:
                raise ValueError(f"Unknown BEAM domain: {domain}")

            prompt = (
                template.replace("<TOPIC>", topic)
                .replace("<THEME>", theme)
                .replace("<FOCUSED_BULLETS>", focus)
                .replace(
                    "<PREVIOUS_SUB_BATCH_PLANS>",
                    task_previous_plans or "(none)",
                )
                .replace("<SUB_BATCH_SIZE>", str(expected))
                .replace("<PREVIOUS_BATCH_PLANS>", task_past_plans or "(none)")
            )
            tasks.append({
                "key": (batch_idx, sub_batch_idx),
                "batch_idx": batch_idx,
                "sub_batch_idx": sub_batch_idx,
                "expected": expected,
                "is_special": is_special,
                "instruction": instruction,
                "prompt": prompt,
                "domain": domain,
                "topic": topic,
                "theme": theme,
                "focus": focus,
            })
            previous_bullet_num += len(bullet_group)

    return tasks, time_anchors


def _question_candidate_errors(candidate: str, task: dict) -> list[str]:
    """Reject format-only, prompt-leaking, or non-substantive BEAM drafts."""

    errors = []
    candidate = candidate.strip()
    if candidate.count("->->") != 1:
        errors.append("expected exactly one ->-> plan marker")
        return errors
    text, suffix = candidate.rsplit("->->", 1)
    text = text.strip().strip("*").strip()
    suffix_match = re.match(r"\s*(\d+)(?:\D|$)", suffix)
    if not suffix_match:
        errors.append("plan marker must end in a bullet number")

    minimum_characters = 80 if task["domain"] == "general" else 120
    if len(text) < minimum_characters or len(text.split()) < 20:
        errors.append("message is too short to be a substantive user request")

    lowered = text.lower()
    prompt_leak_markers = (
        "current focus areas bullet type identification",
        "mandatory detail coverage & tracking",
        "absolute source restriction",
        "required format pattern",
        "verification: before submitting",
    )
    if any(marker in lowered for marker in prompt_leak_markers):
        errors.append("message copied generation instructions")
    if lowered.startswith(("here is the question", "here are the questions")):
        errors.append("message contains a meta preface")

    focus = task.get("focus", "")
    technical_markers = (
        "```", "`", "flask", "sql", "database", "api", "route", "function",
        "class", "html", "css", "javascript", "python", "error", "config",
    )
    coding_without_technical_detail = (
        task["domain"] == "coding"
        and "Time Anchor:" not in focus
        and "Personal Introduction:" not in focus
        and not any(marker in lowered for marker in technical_markers)
    )
    if coding_without_technical_detail:
        errors.append("coding request lacks concrete technical detail")
    return errors


def _task_plan_numbers(task: dict) -> list[int]:
    """Return the immutable plan numbers represented by a question task."""

    numbers = [
        int(match)
        for match in re.findall(r"(?m)^\s*(\d+)\)", task.get("focus", ""))
    ]
    if len(numbers) != task["expected"]:
        raise RuntimeError(
            f"Question task {task.get('key')} maps to {len(numbers)} plan "
            f"numbers, expected {task['expected']}: {numbers}"
        )
    return numbers


def _normalize_question_plan_marker(candidate: str, plan_number: int) -> str:
    """Attach known BEAM plan metadata without asking the model to reproduce it."""

    content = re.sub(
        r"###\s*COMPLETE\s*###.*$", "", candidate,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    content = re.sub(
        r"^---MESSAGE_SEPARATOR---\s*", "", content,
        flags=re.IGNORECASE,
    ).strip()
    # A plan number is orchestration metadata, not generated content. Small
    # models often emit `N`, restart numbering at 1, or omit this suffix. Keep
    # the substantive message and replace only a trailing marker.
    content = re.sub(
        r"\s*->->\s*(?:\d+|N)\s*[.,;:]?\s*$", "", content,
        flags=re.IGNORECASE,
    ).strip()
    return f"{content} ->-> {plan_number}"


def _extract_parallel_question_candidates(raw: str,
                                          task: dict,
                                          already: list[str]) -> list[str]:
    """Recover individually marked messages even if a small model skips separators."""

    if task["domain"] == "general":
        return get_unique_lines(raw, set(already))
    candidates = get_unique_messages(raw, set(already))
    if all(candidate.count("->->") <= 1 for candidate in candidates):
        return candidates

    content = re.sub(
        r'###\s*COMPLETE\s*###.*$', '', raw,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    recovered = []
    start = 0
    for marker in re.finditer(r"->->\s*\d+", content):
        candidate = content[start:marker.end()].strip()
        candidate = re.sub(
            r"^---MESSAGE_SEPARATOR---\s*", "", candidate
        ).strip()
        if candidate:
            recovered.append(candidate)
        start = marker.end()
    already_lower = {item.lower() for item in already}
    return [
        candidate for candidate in recovered
        if candidate.lower() not in already_lower
    ]


def _question_retry_prompt(task: dict,
                           generated: list[str],
                           avoid: list[str],
                           rejection_reasons: list[str],
                           variation_index: int = 0) -> str:
    """Use a compact repair prompt that small instruction models can follow."""

    code_requirement = ""
    focus = task.get("focus", "")
    if (
        task["domain"] == "coding"
        and "Time Anchor:" not in focus
        and "Personal Introduction:" not in focus
    ):
        code_requirement = (
            "Include a realistic fenced code sample of at least 20 lines. "
        )
    prior = "\n\n".join(generated + avoid) or "(none)"
    reasons = "; ".join(sorted(set(rejection_reasons))) or "malformed output"
    separation_rule = (
        "Separate the messages with the exact line `---MESSAGE_SEPARATOR---`."
        if task["expected"] > 1
        else "Return one message only."
    )
    variation_styles = (
        "Open with the concrete behavior or failure you observed, then ask for a fix.",
        "Ask for an implementation review with explicit acceptance criteria.",
        "Frame the request as a debugging task with constraints and expected behavior.",
        "Ask for a step-by-step implementation grounded in the named files and values.",
    )
    variation_rule = variation_styles[variation_index % len(variation_styles)]
    return f"""The previous draft was rejected because: {reasons}.

Write exactly {task['expected']} realistic first-person user message(s) for this
BEAM conversation.

TOPIC: {task.get('topic', '')}
THEME: {task.get('theme', '')}
CURRENT PLAN ITEM(S):
{focus}

Rules:
- Use only facts in the current plan item(s), topic, and theme.
- Directly address the current plan item and reuse at least two of its unique
  file names, identifiers, routes, values, or constraints.
- Make each request at least 120 characters and technically substantive.
- {code_requirement}Do not discuss or copy these generation instructions.
- Uniqueness style: {variation_rule}
- End every message with exactly `->-> N`, where N is its current plan number.
- Return only the message(s); no preface. {separation_rule}
- Do not duplicate these earlier drafts: {prior}
"""


def _invoke_parallel_question_task(task: dict,
                                   llm,
                                   avoid: list[str] | None = None,
                                   max_attempts: int = 5,
                                   variation_seed: int = 0) -> dict:
    """Generate one plan-indexed sub-batch, retrying only malformed output."""

    generated = []
    request_seconds = []
    errors = []
    rejection_reasons = []
    for attempt in range(1, max_attempts + 1):
        batch_history = "\n\n".join((avoid or []) + generated) or "(none)"
        if attempt == 1 and not avoid:
            prompt = task["prompt"].replace("<BATCH_HISTORY>", batch_history)
        else:
            prompt = _question_retry_prompt(
                task,
                generated,
                avoid or [],
                rejection_reasons,
                variation_index=variation_seed + attempt - 1,
            )
        started = time.perf_counter()
        try:
            tokens_per_message = {
                "coding": 1_600,
                "math": 1_000,
                "general": 800,
            }[task["domain"]]
            raw = llm.invoke(
                [SystemMessage(content=prompt)],
                max_tokens=max(600, task["expected"] * tokens_per_message),
            ).content
            request_seconds.append(time.perf_counter() - started)
            candidates = _extract_parallel_question_candidates(
                raw, task, generated
            )
            plan_numbers = _task_plan_numbers(task)
            for candidate in candidates:
                if len(generated) >= task["expected"]:
                    break
                candidate = _normalize_question_plan_marker(
                    candidate, plan_numbers[len(generated)]
                )
                candidate_errors = _question_candidate_errors(candidate, task)
                if candidate_errors:
                    rejection_reasons.extend(candidate_errors)
                    continue
                if candidate.lower() not in {item.lower() for item in generated}:
                    generated.append(candidate)
                if len(generated) == task["expected"]:
                    break
            if len(generated) == task["expected"]:
                return {
                    "candidates": generated,
                    "attempts": attempt,
                    "request_seconds": request_seconds,
                    "errors": errors,
                }
            if not candidates:
                rejection_reasons.append("no parseable message found")
        except Exception as exc:
            request_seconds.append(time.perf_counter() - started)
            errors.append(f"{type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"Batch {task['batch_idx'] + 1} sub-batch "
        f"{task['sub_batch_idx'] + 1} produced "
        f"{len(generated)}/{task['expected']} questions after {max_attempts} attempts; "
        f"validation={sorted(set(rejection_reasons))[-4:]}; errors={errors[-2:]}"
    )


def _format_parallel_question(question: str, batch_idx: int) -> str:
    """Normalize one model draft to BEAM's `text ->-> batch,bullet` form."""

    question = re.sub(r"^\d+\)\s*", "", question.strip())
    if "->->" not in question:
        return f"{question} ->-> {batch_idx + 1},N/A"
    text, bullet_number = question.rsplit("->->", 1)
    bullet_number = bullet_number.strip().split()[0].rstrip(",.;")
    if not bullet_number or len(bullet_number) > 3:
        bullet_number = "N/A"
    return f"{text.strip()} ->-> {batch_idx + 1},{bullet_number}"


def _question_content_key(question: str) -> str:
    """Compare question text independently of BEAM's plan-index suffix."""

    return question.rsplit("->->", 1)[0].strip().lower()


def _is_fuzzy_question_duplicate(question: str, accepted: list[str]) -> bool:
    words = set(_question_content_key(question).split())
    if not words:
        return True
    for existing in accepted:
        existing_words = set(_question_content_key(existing).split())
        if len(words & existing_words) / max(len(words), len(existing_words), 1) > 0.7:
            return True
    return False


def _atomic_pickle_dump(value, path: str) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle)
    os.replace(temporary, path)


def user_messages_generation_parallel(topic: str,
                                      theme: str,
                                      plans: list,
                                      num_batches: int,
                                      sub_batches_per_batch: int,
                                      sub_batch_size: int,
                                      batch_size: int,
                                      llm_name: str,
                                      save_address: str,
                                      domain: str,
                                      sepcial_bullets: bool = False,
                                      max_workers: int = 8) -> dict:
    """Generate BEAM main questions concurrently without changing chat order.

    All model drafts fan out only after the full chronological plan exists.
    Results are restored to `(batch, sub-batch)` order, deduplicated in that
    deterministic order, and only rejected slots are regenerated. The actual
    answer/follow-up chain remains sequential in :func:`answer_generation`.
    """

    started = time.perf_counter()
    llm = _question_generation_llm(llm_name)
    tasks, time_anchors = _build_parallel_question_tasks(
        topic=topic,
        theme=theme,
        plans=plans,
        num_batches=num_batches,
        sub_batches_per_batch=sub_batches_per_batch,
        sub_batch_size=sub_batch_size,
        domain=domain,
        sepcial_bullets=sepcial_bullets,
    )
    cache_address = save_address.replace(".pickle", ".parallel.partial.pickle")
    cache = {"version": 2, "results": {}}
    if os.path.isfile(cache_address):
        try:
            with open(cache_address, "rb") as handle:
                loaded = pickle.load(handle)
            if loaded.get("version") in {1, 2}:
                cache["results"] = loaded.get("results", {})
                invalid_keys = []
                for key, result in cache["results"].items():
                    task = next(
                        (candidate for candidate in tasks if candidate["key"] == key),
                        None,
                    )
                    if task is None or len(result.get("candidates", [])) != task["expected"]:
                        invalid_keys.append(key)
                        continue
                    plan_numbers = _task_plan_numbers(task)
                    normalized = [
                        _normalize_question_plan_marker(candidate, plan_number)
                        for candidate, plan_number in zip(
                            result["candidates"], plan_numbers
                        )
                    ]
                    if any(
                        _question_candidate_errors(candidate, task)
                        for candidate in normalized
                    ):
                        invalid_keys.append(key)
                        continue
                    result["candidates"] = normalized
                for key in invalid_keys:
                    del cache["results"][key]
                _atomic_pickle_dump(cache, cache_address)
                print(
                    f"Resuming {len(cache['results'])}/{len(tasks)} cached "
                    "parallel question tasks"
                )
        except (OSError, EOFError, pickle.PickleError, AttributeError):
            pass

    task_by_key = {task["key"]: task for task in tasks}
    missing = [task for task in tasks if task["key"] not in cache["results"]]
    print(
        f"Generating {len(missing)} independent question tasks with "
        f"{min(max_workers, max(len(missing), 1))} threads"
    )
    if missing:
        failed_tasks = []
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(missing)))) as executor:
            futures = {
                executor.submit(_invoke_parallel_question_task, task, llm): task
                for task in missing
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_tasks.append((task, exc))
                    print(
                        f"  Will retry batch {task['batch_idx'] + 1} sub-batch "
                        f"{task['sub_batch_idx'] + 1} after parallel pass: {exc}"
                    )
                    continue
                cache["results"][task["key"]] = result
                _atomic_pickle_dump(cache, cache_address)
                print(
                    f"  Drafted batch {task['batch_idx'] + 1} sub-batch "
                    f"{task['sub_batch_idx'] + 1} in "
                    f"{sum(result['request_seconds']):.1f}s"
                )
        for task, initial_error in failed_tasks:
            print(
                f"  Repairing batch {task['batch_idx'] + 1} sub-batch "
                f"{task['sub_batch_idx'] + 1} with an extended retry budget"
            )
            try:
                result = _invoke_parallel_question_task(
                    task, llm, max_attempts=8
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Extended repair failed after initial error: {initial_error}"
                ) from exc
            cache["results"][task["key"]] = result
            _atomic_pickle_dump(cache, cache_address)
            print(
                f"  Repaired batch {task['batch_idx'] + 1} sub-batch "
                f"{task['sub_batch_idx'] + 1} in "
                f"{sum(result['request_seconds']):.1f}s"
            )

    all_messages = []
    global_history = set()
    retry_calls = 0
    retry_request_seconds = []
    for batch_idx in range(num_batches):
        batch_lines = []
        batch_tasks = [task for task in tasks if task["batch_idx"] == batch_idx]
        for task in batch_tasks:
            result = cache["results"][task["key"]]
            slot_values = [
                _format_parallel_question(candidate, batch_idx)
                for candidate in result["candidates"]
            ]
            if task["is_special"]:
                instruction = task["instruction"] or ""
                _, _, instruction_text = instruction.partition("User Instruction:")
                instruction_text = instruction_text or instruction
                slot_values.insert(
                    1, _format_parallel_question(instruction_text.strip(), batch_idx)
                )

            accepted_slots = [None] * len(slot_values)
            generated_slot_indexes = [
                index for index in range(len(slot_values))
                if not (task["is_special"] and index == 1)
            ]
            for slot_index, candidate in enumerate(slot_values):
                key = _question_content_key(candidate)
                if (
                    key not in global_history
                    and key not in {_question_content_key(item) for item in batch_lines}
                    and not _is_fuzzy_question_duplicate(candidate, batch_lines)
                ):
                    accepted_slots[slot_index] = candidate
                    batch_lines.append(candidate)

            if task["is_special"] and accepted_slots[1] is None:
                raise RuntimeError(
                    f"The plan's user instruction duplicates an earlier question in batch {batch_idx + 1}"
                )

            for retry_number in range(1, 7):
                missing_slots = [
                    index for index in generated_slot_indexes
                    if accepted_slots[index] is None
                ]
                if not missing_slots:
                    break
                rejected_drafts = [
                    _question_content_key(slot_values[index])
                    for index in missing_slots
                ]
                retry = _invoke_parallel_question_task(
                    task,
                    llm,
                    # Always show the model the draft that was rejected. This
                    # matters for an exact duplicate from an earlier batch:
                    # `batch_lines` is still empty at the first sub-batch, so
                    # the old retry prompt otherwise reproduced the same text.
                    avoid=(
                        rejected_drafts
                        + [_question_content_key(item) for item in batch_lines[-12:]]
                    ),
                    variation_seed=retry_number,
                )
                retry_calls += retry["attempts"]
                retry_request_seconds.extend(retry["request_seconds"])
                retry_values = [
                    _format_parallel_question(candidate, batch_idx)
                    for candidate in retry["candidates"]
                ]
                if task["is_special"]:
                    retry_values.insert(1, accepted_slots[1])
                cache_changed = False
                for slot_index in missing_slots:
                    candidate = retry_values[slot_index]
                    key = _question_content_key(candidate)
                    if (
                        key not in global_history
                        and key not in {_question_content_key(item) for item in batch_lines}
                        and not _is_fuzzy_question_duplicate(candidate, batch_lines)
                    ):
                        accepted_slots[slot_index] = candidate
                        batch_lines.append(candidate)
                        candidate_index = (
                            slot_index
                            if not task["is_special"] or slot_index < 1
                            else slot_index - 1
                        )
                        cache["results"][task["key"]]["candidates"][candidate_index] = (
                            retry["candidates"][candidate_index]
                        )
                        cache_changed = True
                if cache_changed:
                    _atomic_pickle_dump(cache, cache_address)

            if any(value is None for value in accepted_slots):
                raise RuntimeError(
                    f"Could not replace duplicate questions in batch {batch_idx + 1}, "
                    f"sub-batch {task['sub_batch_idx'] + 1}"
                )

            # Re-establish exact plan order after concurrent completion and retries.
            for candidate in accepted_slots:
                if candidate in batch_lines:
                    batch_lines.remove(candidate)
            batch_lines.extend(accepted_slots)

        expected_batch_size = batch_size + (3 if sepcial_bullets else 0)
        if len(batch_lines) != expected_batch_size:
            raise RuntimeError(
                f"Batch {batch_idx + 1} has {len(batch_lines)}/{expected_batch_size} questions"
            )
        all_messages.append({
            "time_anchor": time_anchors[batch_idx],
            "messages": [batch_lines],
        })
        global_history.update(_question_content_key(line) for line in batch_lines)
        print(f"Batch {batch_idx + 1} complete: {len(batch_lines)} messages")

    _atomic_pickle_dump(all_messages, save_address)
    if os.path.exists(cache_address):
        os.remove(cache_address)

    initial_results = [cache["results"][task["key"]] for task in tasks]
    request_seconds = [
        seconds
        for result in initial_results
        for seconds in result["request_seconds"]
    ] + retry_request_seconds
    metrics = {
        "mode": "parallel_plan_indexed",
        "wall_seconds": time.perf_counter() - started,
        "max_workers": max_workers,
        "task_count": len(tasks),
        "model_request_count": sum(result["attempts"] for result in initial_results)
        + retry_calls,
        "model_request_seconds_sum": sum(request_seconds),
        "model_request_seconds_max": max(request_seconds, default=0.0),
        "dedupe_retry_request_count": retry_calls,
        "message_count": sum(len(batch["messages"][0]) for batch in all_messages),
    }
    print(
        f"Parallel question generation complete: {metrics['message_count']} messages "
        f"in {metrics['wall_seconds']:.1f}s"
    )
    return metrics

# ================================ ANSWER GENERATION ================================


def check_include_questions(assistant_response: str,
                            llm) -> bool:

    prompt = check_include_question_template\
        .replace("<assistant_response>", assistant_response)

    response = llm.invoke(prompt, max_tokens=12).content.strip().lower()

    if "yes" in response:
        return True
    else:
        return False


def check_include_questions_safe(assistant_response,
                                 assistant_memory,
                                 llm,
                                 max_tokens=25000):

    response_tokens = assistant_memory.estimate_tokens(assistant_response)
    if response_tokens > max_tokens:
        assistant_response = truncate_to_tokens(
            assistant_response, max_tokens, assistant_memory)

    return check_include_questions(assistant_response=assistant_response, llm=llm)


def check_need_followup(assistant_response: str,
                        messages_history: list,
                        topic: str,
                        theme: str,
                        llm) -> bool:

    formatted_history = "\n".join([
        f"USER: {msg}" if i % 2 == 0 else f"ASSISTANT: {msg}"
        for i, msg in enumerate(messages_history)
    ])

    prompt = check_need_followup_template\
        .replace("<topic>", topic)\
        .replace("<theme>", theme)\
        .replace("<formatted_history>", formatted_history)\
        .replace("<assistant_response>", assistant_response)

    response = llm.invoke(prompt, max_tokens=12).content.strip().lower()

    if "yes" in response:
        return True
    else:
        return False


def check_need_followup_safe(assistant_response,
                             assistant_memory,
                             messages_history,
                             topic,
                             theme,
                             llm,
                             max_tokens=25000):

    # First, check if assistant_response alone is too large
    assistant_response_tokens = assistant_memory.estimate_tokens(
        assistant_response)
    topic_theme_tokens = assistant_memory.estimate_tokens(f"{topic}{theme}")

    # If assistant_response is too large, truncate it first
    # Reserve half for response, half for history
    max_response_tokens = (max_tokens - 1000) // 2

    if assistant_response_tokens > max_response_tokens:
        # Truncate assistant_response to fit
        assistant_response = truncate_to_tokens(
            assistant_response,
            max_response_tokens,
            assistant_memory
        )
        assistant_response_tokens = assistant_memory.estimate_tokens(
            assistant_response)

    # Calculate remaining budget for history
    base_tokens = assistant_response_tokens + topic_theme_tokens
    available_for_history = max_tokens - base_tokens - 1000  # Buffer

    # If we've already exceeded the limit with just base content
    if available_for_history <= 0:
        print(
            f"Warning: Base content alone uses {base_tokens} tokens, exceeding limit")
        # Try to call with minimal content
        return check_need_followup(
            # Aggressive truncation
            assistant_response=assistant_response[:1000] + "...[truncated]",
            messages_history=[],  # No history
            topic=topic[:100] if len(topic) > 100 else topic,
            theme=theme[:100] if len(theme) > 100 else theme,
            llm=llm
        )

    # Truncate history if needed
    if messages_history:
        history_tokens = sum(assistant_memory.estimate_tokens(
            msg.get("content", "")) for msg in messages_history
        )

        if history_tokens > available_for_history:
            # Keep only recent messages that fit
            truncated_history = []
            current_tokens = 0

            for msg in reversed(messages_history):
                msg_tokens = assistant_memory.estimate_tokens(
                    msg.get("content", ""))
                if current_tokens + msg_tokens < available_for_history:
                    truncated_history.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    # Try to include a truncated version of this message
                    remaining_tokens = available_for_history - current_tokens
                    if remaining_tokens > 100:  # Only if we have reasonable space
                        truncated_msg = msg.copy()
                        truncated_msg["content"] = truncate_to_tokens(
                            msg.get("content", ""),
                            remaining_tokens,
                            assistant_memory
                        )
                        truncated_history.insert(0, truncated_msg)
                    break

            messages_history = truncated_history

            print(
                f"Truncated history from {history_tokens} to {current_tokens} tokens")

    # Final safety check
    total_tokens = (
        assistant_memory.estimate_tokens(assistant_response) +
        assistant_memory.estimate_tokens(f"{topic}{theme}") +
        sum(assistant_memory.estimate_tokens(msg.get("content", ""))
            for msg in messages_history)
    )

    if total_tokens > max_tokens:
        print(
            f"Warning: Total tokens ({total_tokens}) still exceed limit ({max_tokens})")

    return check_need_followup(
        assistant_response=assistant_response,
        messages_history=messages_history,
        topic=topic,
        theme=theme,
        llm=llm
    )


def format_history_messages(message_history: list) -> str:
    history_text = ""
    for message in message_history:
        if message["role"] == "user":
            history_text += f"You: {message["content"]}" + "\n\n"
        elif message["role"] == "assistant":
            history_text += f"AI Assistant: {message["content"]}" + "\n\n"

    return history_text


def _count_saved_chat_tokens(batches: list,
                             tokenizer_model: str) -> int:
    """Count BEAM main turns exactly as the JSON converter/truncator does."""

    from src.beam.adjust_chats_length import count_message_tokens

    total = 0
    for batch in batches:
        current_turn = []
        turns = []
        for message in batch:
            if (
                message.get("question_type") == "main_question"
                and current_turn
            ):
                turns.append(current_turn)
                current_turn = []
            current_turn.append(message)
        if current_turn:
            turns.append(current_turn)
        total += sum(
            count_message_tokens(turn, tokenizer_model)
            for turn in turns
        )
    return total


def _memory_checkpoint(memory: ConversationSummaryBuffer) -> dict:
    return {
        "messages": list(memory.messages),
        "summary": memory.summary,
    }


def _restore_memory(memory: ConversationSummaryBuffer, state: dict) -> None:
    memory.messages = list(state.get("messages", []))
    memory.summary = state.get("summary", "")


def answer_generation(input_address: str,
                      output_address: str,
                      plans_address: str,
                      topic: str,
                      theme: str,
                      llm,
                      previous_plans_summary: str = "",
                      stop_token_target: int | None = None,
                      stop_token_limit: int | None = None,
                      tokenizer_model: str = "meta-llama/llama-3.1-8b-instruct",
                      response_max_tokens: int = 1_000):

    start = time.perf_counter()

    if not os.path.exists(input_address):
        raise FileNotFoundError(f"Input file not found: {input_address}")
    if not os.path.exists(plans_address):
        raise FileNotFoundError(f"Plans file not found: {plans_address}")

    with open(plans_address, 'rb') as f:
        plans = pickle.load(f)

    with open(input_address, 'rb') as f:
        batches = pickle.load(f)

    checkpoint_address = output_address.replace(
        ".pickle", ".answer.partial.pickle"
    )
    all_messages = []
    id = 0
    resume_checkpoint = None

    if os.path.isfile(checkpoint_address):
        try:
            with open(checkpoint_address, "rb") as f:
                resume_checkpoint = pickle.load(f)
            if resume_checkpoint.get("version") == 1:
                all_messages = resume_checkpoint["all_messages"]
                id = resume_checkpoint["next_id"]
                print(
                    "Resuming answer generation at batch "
                    f"{resume_checkpoint['batch_index'] + 1}, question "
                    f"{resume_checkpoint['next_question_index'] + 1}"
                )
            else:
                resume_checkpoint = None
        except (OSError, EOFError, pickle.PickleError, KeyError, AttributeError):
            resume_checkpoint = None

    if resume_checkpoint is None and os.path.isfile(output_address):
        with open(output_address, 'rb') as f:
            messages = pickle.load(f)
            all_messages = messages
        existing_ids = [
            message.get("id")
            for batch in all_messages
            for message in batch
            if isinstance(message, dict) and isinstance(message.get("id"), int)
        ]
        id = max(existing_ids, default=-1) + 1

    saved_for_target = all_messages + (
        [resume_checkpoint.get("current_batch_messages", [])]
        if resume_checkpoint is not None
        and resume_checkpoint.get("current_batch_messages")
        else []
    )
    if (
        stop_token_target is not None
        and saved_for_target
        and _count_saved_chat_tokens(saved_for_target, tokenizer_model) >= stop_token_target
    ):
        _atomic_pickle_dump(saved_for_target, output_address)
        if os.path.exists(checkpoint_address):
            os.remove(checkpoint_address)
        return

    start_batch_index = (
        resume_checkpoint["batch_index"]
        if resume_checkpoint is not None
        else len(all_messages)
    )
    for batch_index in range(start_batch_index, len(batches)):
        batch = batches[batch_index]
        try:
            previous_plans = ""
            for i in range(batch_index):
                plan = extract_plan_bullets(plan_text=plans[i])
                plan = "\n\n".join(plan)
                previous_plans += plan + "\n\n\n"

            current_plan = ""
            plan = extract_plan_bullets(plan_text=plans[batch_index])
            plan = "\n\n".join(plan)
            current_plan += plan + "\n\n\n"

        except Exception as e:
            print(f"Error extracting plan for batch {batch_index}: {str(e)}")

        if batch_index == 0:
            ai_assitant_system_prompt = ai_assistant_llm_template\
                .replace("<topic>", "N/A")\
                .replace("<theme>", "N/A")\
                .replace("<previous_plans_summary>", previous_plans_summary)\
                .replace("<previous_batches>", "N/A")
        else:
            ai_assitant_system_prompt = ai_assistant_llm_template\
                .replace("<topic>", topic)\
                .replace("<theme>", theme)\
                .replace("<previous_plans_summary>", previous_plans_summary)\
                .replace("<previous_batches>", previous_plans)

        try:
            assistant_prompt_tokens = get_token_number(
                ai_assitant_system_prompt)
            assistant_max_tokens = 28000 - assistant_prompt_tokens

            assistant_memory = ConversationSummaryBuffer(
                llm=llm,
                max_tokens=assistant_max_tokens,
                recent_messages_count=10,
                max_summary_tokens=1000
            )

            user_memory = ConversationSummaryBuffer(
                llm=llm,
                max_tokens=19000,
                recent_messages_count=10,
                max_summary_tokens=2000
            )
        except NameError:
            print("ConversationSummaryBuffer or llm not defined")
            continue
        except Exception as e:
            print(f"Error initializing memory objects: {str(e)}")
            continue

        ai_assistant_llm_messages_history = [
            {"role": "system", "content": ai_assitant_system_prompt}
        ]
        user_llm_messages_history = []
        start_question_index = 0

        if (
            resume_checkpoint is not None
            and batch_index == resume_checkpoint["batch_index"]
        ):
            user_llm_messages_history = list(
                resume_checkpoint.get("current_batch_messages", [])
            )
            start_question_index = resume_checkpoint["next_question_index"]
            _restore_memory(
                assistant_memory,
                resume_checkpoint.get("assistant_memory", {}),
            )
            _restore_memory(
                user_memory,
                resume_checkpoint.get("user_memory", {}),
            )

        ai_assitant_llm = llm
        user_llm = llm

        try:
            messages = batch['messages'][0]
            time_anchor = batch['time_anchor']
        except (KeyError, IndexError, TypeError) as e:
            print(
                f"Error extracting messages from batch {batch_index}: {str(e)}")
            continue
        
        for question_index in range(start_question_index, len(messages)):
            question = messages[question_index]
            try:
                current_batches = all_messages + (
                    [user_llm_messages_history]
                    if user_llm_messages_history else []
                )
                tokens_before_question = (
                    _count_saved_chat_tokens(current_batches, tokenizer_model)
                    if stop_token_limit is not None else 0
                )
                near_length_target = (
                    stop_token_target is not None
                    and tokens_before_question >= stop_token_target - 6_000
                )

                if "->->" in question:
                    batch_number = question.split("->->")[1].strip()
                    if "," in batch_number:
                        batch_number_index = batch_number.split(",")[0].strip()
                        bullet_number_index = batch_number.split(",")[
                            1].strip()
                        if bullet_number_index == "":
                            batch_number = batch_number_index + "," + "N/A"
                else:
                    batch_number = "N/A,N/A"

                elapsed = time.perf_counter() - start
                hms = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                print(f"Elapsed time: {hms}")

                print(
                    f"Batch: {batch_index+1}/{len(batches)}, Question Index: {question_index+1}/{len(messages)}")

                # Add to user memory
                user_memory.add_message("user", question)
                assistant_memory.add_message(
                    "user", question + "Please respond only in English.")

                ai_assistant_llm_messages_history.append(
                    {"role": "assistant", "content": question}
                )

                # Get managed messages for assistant
                assistant_messages = assistant_memory.get_messages_for_llm(
                    ai_assitant_system_prompt)
                response_token_limit = response_max_tokens
                if stop_token_limit is not None:
                    from src.beam.adjust_chats_length import count_message_tokens

                    question_tokens = count_message_tokens(
                        [{"role": "user", "content": question}],
                        tokenizer_model,
                    )
                    remaining = stop_token_limit - tokens_before_question
                    # Provider and Llama token counts are not identical. The
                    # safety factor keeps the complete main turn under 64K in
                    # the common case and lets the local truncator handle any
                    # small overshoot.
                    safe_completion_budget = int(
                        max(128, remaining - question_tokens - 160) * 0.75
                    )
                    response_token_limit = max(
                        128, min(response_max_tokens, safe_completion_budget)
                    )
                ai_assistant_response = ai_assitant_llm.invoke(
                    assistant_messages, max_tokens=response_token_limit).content

                # Add response to memories
                assistant_memory.add_message(
                    "assistant", ai_assistant_response)
                user_memory.add_message("assistant", ai_assistant_response)

                ai_assistant_llm_messages_history.append(
                    {"role": "user", "content": ai_assistant_response}
                )

                if question_index == 0:
                    user_llm_messages_history.append(
                        {"role": "user", "id": id, "time_anchor": time_anchor, "index": batch_number,
                            "question_type": "main_question", "content": question}
                    )
                else:
                    user_llm_messages_history.append(
                        {"role": "user", "id": id, "index": batch_number,
                            "question_type": "main_question", "content": question}
                    )

                id += 1

                user_llm_messages_history.append(
                    {"role": "assistant", "id": id,
                        "content": ai_assistant_response}
                )
                id += 1

                num_questions = 0
                include_question = check_include_questions_safe(
                    assistant_response=ai_assistant_response, assistant_memory=assistant_memory, llm=llm)

                if near_length_target:
                    include_question = False

                while include_question and num_questions < 2:
                    history_text = user_memory.get_conversation_history_text()

                    replacements = {
                        "current_batch_messages": history_text,
                        "topic": topic,
                        "theme": theme,
                        "previous_plans_summary": previous_plans_summary,
                        "previous_batches": previous_plans,
                        "current_plan": current_plan,
                        "ai_last_message": ai_assistant_response
                    }

                    user_system_prompt = prepare_prompt_within_budget(
                        user_llm_answer_question_template,
                        replacements,
                        user_memory,
                        max_tokens=27000
                    )

                    user_response = user_llm.invoke(
                        user_system_prompt, max_tokens=350
                    ).content

                    # Add to memories
                    user_memory.add_message("user", user_response)
                    assistant_memory.add_message(
                        "user", user_response + "Please respond only in English.")

                    user_llm_messages_history.append(
                        {"role": "user", "id": id, "question_type": "answer_ai_question",
                            "content": user_response}
                    )
                    id += 1

                    ai_assistant_llm_messages_history.append(
                        {"role": "assistant", "content": user_response}
                    )

                    # Get managed messages for assistant
                    assistant_messages = assistant_memory.get_messages_for_llm(
                        ai_assitant_system_prompt)
                    ai_assistant_response = ai_assitant_llm.invoke(
                        assistant_messages, max_tokens=response_max_tokens).content

                    # Add to memories
                    user_memory.add_message("assistant", ai_assistant_response)
                    assistant_memory.add_message(
                        "assistant", ai_assistant_response)

                    user_llm_messages_history.append(
                        {"role": "assistant", "id": id,
                            "content": ai_assistant_response}
                    )
                    id += 1

                    ai_assistant_llm_messages_history.append(
                        {"role": "user", "content": ai_assistant_response}
                    )

                    include_question = check_include_questions_safe(
                        assistant_response=ai_assistant_response, assistant_memory=assistant_memory, llm=llm)

                    num_questions += 1

                messages_history = user_memory.get_messages_for_llm()

                need_followup = check_need_followup_safe(assistant_response=ai_assistant_response,
                                                         assistant_memory=assistant_memory,
                                                         messages_history=messages_history,
                                                         topic=topic, theme=theme, llm=llm)

                if near_length_target:
                    need_followup = False

                num_followup_questions = 0
                while need_followup and num_followup_questions < 2:
                    # Get managed conversation history from user memory
                    history_text = user_memory.get_conversation_history_text()

                    replacements = {
                        "current_batch_messages": history_text,
                        "topic": topic,
                        "theme": theme,
                        "previous_plans_summary": previous_plans_summary,
                        "previous_batches": previous_plans,
                        "current_plan": current_plan,
                        "ai_last_message": ai_assistant_response
                    }

                    user_system_prompt = prepare_prompt_within_budget(
                        user_llm_ask_followup_question_template,
                        replacements,
                        user_memory,
                        max_tokens=27000
                    )

                    user_response = user_llm.invoke(
                        user_system_prompt, max_tokens=350
                    ).content

                    # Add to memories
                    user_memory.add_message("user", user_response)
                    assistant_memory.add_message(
                        "user", user_response + "Please respond only in English.")

                    user_llm_messages_history.append(
                        {"role": "user", "id": id, "question_type": "followup_question",
                            "content": user_response}
                    )
                    id += 1

                    ai_assistant_llm_messages_history.append(
                        {"role": "assistant", "content": user_response}
                    )

                    # Get managed messages for assistant
                    assistant_messages = assistant_memory.get_messages_for_llm(
                        ai_assitant_system_prompt)
                    ai_assistant_response = ai_assitant_llm.invoke(
                        assistant_messages, max_tokens=response_max_tokens).content

                    # Add to memories
                    user_memory.add_message("assistant", ai_assistant_response)
                    assistant_memory.add_message(
                        "assistant", ai_assistant_response)

                    user_llm_messages_history.append(
                        {"role": "assistant", "id": id,
                            "content": ai_assistant_response}
                    )
                    id += 1

                    ai_assistant_llm_messages_history.append(
                        {"role": "user", "content": ai_assistant_response}
                    )

                    num_followup_questions += 1

                    messages_history = user_memory.get_messages_for_llm()

                    need_followup = check_need_followup_safe(assistant_response=ai_assistant_response,
                                                             assistant_memory=assistant_memory,
                                                             messages_history=messages_history,
                                                             topic=topic, theme=theme, llm=llm)

                if stop_token_target is not None:
                    candidate_batches = all_messages + [user_llm_messages_history]
                    measured_tokens = _count_saved_chat_tokens(
                        candidate_batches, tokenizer_model
                    )
                    print(
                        f"Current saved chat length: {measured_tokens}/"
                        f"{stop_token_target} target tokens"
                    )
                    if measured_tokens >= stop_token_target:
                        _atomic_pickle_dump(candidate_batches, output_address)
                        if os.path.exists(checkpoint_address):
                            os.remove(checkpoint_address)
                        print(
                            "Reached the requested saved-chat target; "
                            "stopping before later turns that truncation would discard."
                        )
                        return

                # A long BEAM batch can take many minutes. Persist every
                # completed main turn, including the exact rolling-memory
                # state needed to resume without changing later prompts.
                candidate_batches = all_messages + [user_llm_messages_history]
                _atomic_pickle_dump({
                    "version": 1,
                    "all_messages": all_messages,
                    "batch_index": batch_index,
                    "next_question_index": question_index + 1,
                    "current_batch_messages": user_llm_messages_history,
                    "next_id": id,
                    "assistant_memory": _memory_checkpoint(assistant_memory),
                    "user_memory": _memory_checkpoint(user_memory),
                }, checkpoint_address)
                _atomic_pickle_dump(candidate_batches, output_address)
            except Exception as e:
                raise RuntimeError(
                    f"Could not complete question {question_index + 1} in "
                    f"batch {batch_index + 1}; the last completed-turn checkpoint "
                    "was preserved"
                ) from e

        if user_llm_messages_history and (len(user_llm_messages_history) >= (len(messages) * 2 - 10)):
            all_messages.append(user_llm_messages_history)

            _atomic_pickle_dump(all_messages, output_address)
            if os.path.exists(checkpoint_address):
                os.remove(checkpoint_address)
            resume_checkpoint = None
        else:
            break

    _atomic_pickle_dump(all_messages, output_address)
    if os.path.exists(checkpoint_address):
        os.remove(checkpoint_address)


def _available_probe_plan_indexes(chat_address: str) -> set[tuple[int, int]]:
    """Return plan coordinates that survived the complete-turn 64K cutoff."""

    with open(chat_address, encoding="utf-8") as handle:
        chat = json.load(handle)
    available = set()
    for batch in chat:
        for turn in batch.get("turns", []):
            if not turn:
                continue
            marker = str(turn[0].get("index", ""))
            match = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", marker)
            if match:
                available.add((int(match.group(1)), int(match.group(2))))
    return available


def _probe_selection_records(value) -> list[dict]:
    """Normalize common equivalent JSON envelopes used by provider models."""

    records = []

    def joined(item, separator=","):
        if isinstance(item, list):
            return separator.join(str(part) for part in item)
        return item

    def bullet_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, list):
            parts = []
            for part in item:
                if isinstance(part, dict):
                    part = next(
                        (
                            part.get(key)
                            for key in (
                                "bullet_point", "bullet_text", "description",
                                "event", "content", "text",
                            )
                            if part.get(key)
                        ),
                        None,
                    )
                if part is not None:
                    parts.append(str(part))
            return " | ".join(parts)
        return None

    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        batch_numbers = next(
            (item.get(key) for key in ("batch_numbers", "batch_number", "batches") if item.get(key) is not None),
            None,
        )
        bullet_numbers = next(
            (item.get(key) for key in ("bullet_numbers", "bullet_number", "indices") if item.get(key) is not None),
            None,
        )
        points = next(
            (
                item.get(key)
                for key in (
                    "bullet_points", "bullet_point", "selected_bullets",
                    "events", "descriptions",
                )
                if item.get(key) is not None
            ),
            None,
        )
        normalized_points = bullet_text(points)
        if (
            batch_numbers is not None
            and bullet_numbers is not None
            and normalized_points
        ):
            normalized = dict(item)
            normalized["batch_numbers"] = joined(batch_numbers)
            normalized["bullet_numbers"] = joined(bullet_numbers)
            normalized["bullet_points"] = normalized_points
            records.append(normalized)
            return
        for child in item.values():
            if isinstance(child, (dict, list)):
                visit(child)

    visit(value)
    return records


def _question_dict_count(value) -> int:
    if isinstance(value, dict):
        return int(isinstance(value.get("question"), str) and bool(value["question"].strip())) + sum(
            _question_dict_count(child) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_question_dict_count(child) for child in value)
    return 0


def generate_probing_questions_candidate(plan_address: str,
                                         probing_question_type: str,
                                         llm_name: str,
                                         chat_address: str,
                                         save_address: str,
                                         file_name: str,
                                         domain: str):

    with open(plan_address, 'rb') as f:
        plans = pickle.load(f)

    available_indexes = _available_probe_plan_indexes(chat_address)
    original_plan = ""
    for plan_index, plan in enumerate(plans):
        current_plan = ""
        plan_bullets = extract_plan_bullets(plan)
        for bullet_index, bullet in enumerate(plan_bullets):
            if (plan_index + 1, bullet_index + 1) not in available_indexes:
                continue
            current_plan += f"Bullet Number: {bullet_index+1} -> {bullet} \n"

        if current_plan:
            original_plan += f"BATCH {plan_index+1} PLAN \n" + \
                current_plan + "\n\n"

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    if probing_question_type == "information_extraction":
        prompt = information_extraction_prompt\
            .replace("<plan>", original_plan)\
            .replace("<bullet_number>", "8-10")

    elif probing_question_type == "multi_session_reasoning":
        prompt = multi_session_reasoning_prompt.replace(
            "<plan>", original_plan)

    elif probing_question_type == "knowledge_update":
        prompt = knowledge_update_special_bullets_prompt.replace(
            "<plan>", original_plan)

    elif probing_question_type == "temporal_reasoning":
        prompt = temporal_reasoning_prompt.replace("<plan>", original_plan)

    elif probing_question_type == "preference_following":
        prompt = preference_following_prompt\
            .replace("<plan>", original_plan)\
            .replace("<bullet_number>", "8-12")
        if "Preference Statement" not in original_plan:
            prompt += """

This retained coding plan has no bullet literally labeled Preference Statement.
Select 4-8 retained bullets that express a clear technical choice, avoidance,
or favored approach (for example local-first operation, minimal dependencies,
or deliberately excluding an ORM). Treat the stated choice as the user's
technical preference. Do not select User Instruction bullets. Use the exact
JSON fields requested above.
"""

    elif probing_question_type == "event_ordering":
        prompt = event_ordering_prompt.replace(
            "<plan>", original_plan)

    elif probing_question_type == "contradiction_resolution":
        prompt = contradiction_resolution_prompt\
            .replace("<plan>", original_plan)\
            .replace("<bullet_number>", "8-12")

    elif probing_question_type == "summarization":
        prompt = summarization_prompt.replace("<plan>", original_plan)

    elif probing_question_type == "instruction_following":
        prompt = instruction_following_prompt.replace("<plan>", original_plan)

    data = []
    selection_errors = []
    for attempt in range(1, 4):
        format_reminder = "" if attempt == 1 else f"""

FORMAT RETRY {attempt}: Return a top-level JSON array. Every array item must be
an object containing `batch_numbers`, `bullet_numbers`, and `bullet_points`.
Do not nest the item inside another array or envelope.
"""
        try:
            response = llm.invoke(prompt + format_reminder).content
            parsed = json.loads(repair_json(response))
            data = _probe_selection_records(parsed)
            if data:
                break
            selection_errors.append("no normalized selection records")
        except Exception as exc:
            selection_errors.append(f"{type(exc).__name__}: {exc}")
    if not data:
        raise RuntimeError(
            f"Could not parse {probing_question_type} plan selections after "
            f"3 attempts: {selection_errors[-3:]}"
        )

    probing_questions_directory = os.path.join(
        save_address, "probing_questions")
    if not os.path.isdir(probing_questions_directory):
        os.makedirs(probing_questions_directory, exist_ok=True)

    current_probing_question_directory = os.path.join(
        probing_questions_directory, probing_question_type)
    if not os.path.isdir(current_probing_question_directory):
        os.makedirs(current_probing_question_directory, exist_ok=True)

    qas_file = os.path.join(
        current_probing_question_directory, f"{file_name}.json")

    qas = []
    print(f"Number of questions: {len(data)}")
    for index, obj in enumerate(data):
        print(f"Question Num: {index}")
        batch_numbers = str(obj['batch_numbers'])
        bullet_numbers = str(obj['bullet_numbers'])
        indexes = []
        bullets = str(obj['bullet_points']).split("|")
        if len(batch_numbers.split(",")) == len(bullet_numbers.split(",")):
            for i, batch_number in enumerate(batch_numbers.split(",")):
                indexes.append(
                    [f"{str(batch_number)}", f"{str(bullet_numbers.split(',')[i])}"])
        else:
            for i, bullet_number in enumerate(bullet_numbers.split(",")):
                indexes.append([batch_numbers, str(bullet_number)])

        response = generate_probing_questions(chat_address=chat_address,
                                              indexes=indexes,
                                              bullets=bullets,
                                              probing_question_type=probing_question_type,
                                              llm_name=llm_name,
                                              domain=domain)

        qas.append(response)

        with open(qas_file, 'w') as f:
            json.dump(qas, f, indent=4)

    # Single-fact capabilities may have only one retained special bullet at a
    # 64K cutoff. Generate a distinct second difficulty from the same evidence
    # so assembly can still select two questions per BEAM ability.
    variant_attempt = 0
    while _question_dict_count(qas) < 2 and data and variant_attempt < 3:
        variant_attempt += 1
        obj = data[0]
        batch_numbers = str(obj['batch_numbers'])
        bullet_numbers = str(obj['bullet_numbers'])
        indexes = []
        if len(batch_numbers.split(",")) == len(bullet_numbers.split(",")):
            indexes = [
                [batch.strip(), bullet.strip()]
                for batch, bullet in zip(
                    batch_numbers.split(","), bullet_numbers.split(",")
                )
            ]
        else:
            indexes = [
                [batch_numbers.strip(), bullet.strip()]
                for bullet in bullet_numbers.split(",")
            ]
        response = generate_probing_questions(
            chat_address=chat_address,
            indexes=indexes,
            bullets=str(obj['bullet_points']).split("|"),
            probing_question_type=probing_question_type,
            llm_name=llm_name,
            domain=domain,
            variation_instruction=(
                "Generate a distinctly phrased hard variant testing the same "
                f"evidence. Do not repeat the earlier wording. Variant {variant_attempt}."
            ),
        )
        qas.append(response)

    with open(qas_file, 'w') as f:
        json.dump(qas, f, indent=4)


def generate_probing_questions(chat_address: str,
                               indexes: list,
                               bullets: list,
                               probing_question_type: str,
                               llm_name: str,
                               domain: str = "general",
                               plan_address: str = None,
                               chat_length: str = None,
                               plan_number: str = None,
                               variation_instruction: str = ""):

    def parse_json_response(response: str):
        """
        Parses a potentially malformed JSON string with multiple fallbacks.
        """
        s = response.strip()

        if s.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
            if m:
                s = m.group(1).strip()

        s = s.replace("“", '"').replace(
            "”", '"').replace("’", "'").replace("‘", "'")

        s = re.sub(r',(\s*[}\]])', r'\1', s)

        s = re.sub(r'([{\s,])(\w+?)\s*:', r'\1"\2":', s)

        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            try:
                s_python = s.replace('true', 'True').replace(
                    'false', 'False').replace('null', 'None')

                py_obj = ast.literal_eval(s_python)

                return json.loads(json.dumps(py_obj))
            except Exception:
                raise ValueError(f"Failed to parse JSON: {e}")

    def parse_with_repair_library(response: str) -> dict | list:
        """
        Parses a potentially malformed JSON string from an LLM using the
        json-repair library. This is the recommended approach.
        """
        try:
            return json.loads(repair_json(response))
        except Exception as e:
            print(e)
            with open("bad_json.log", "a", encoding="utf-8") as f:
                f.write(
                    f"---- FAILED WITH json-repair ----\n{response}\nError: {e}\n")
            raise ValueError(
                f"Failed to parse JSON even with repair library: {e}")

    def _get_token_encoder():
        """Initialize the best available token encoder"""
        try:
            import tiktoken
            encoders_to_try = ["cl100k_base", "p50k_base", "r50k_base"]

            for encoding_name in encoders_to_try:
                try:
                    encoder = tiktoken.get_encoding(encoding_name)
                    # print(f"Using tiktoken with {encoding_name} encoding")
                    return encoder
                except:
                    continue
        except:
            pass

    if llm_name == "gpt":
        llm = gpt_llm
    elif llm_name == "qwen":
        llm = qwen_llm
    elif llm_name == "llama":
        llm = llama_llm

    with open(chat_address, 'r') as file:
        data = json.load(file)

    if chat_length == "10M":
        data = data[plan_number][f"plan-{plan_number+1}"]

    selected_turns = []
    for index in indexes:
        batch_number = index[0].strip()
        bullet_number = index[1].strip()

        for i, obj in enumerate(data):
            if str(i+1) == batch_number:
                turns = obj["turns"]
                for turn in turns:
                    if turn[0]["index"].split(",")[1].strip() == bullet_number:
                        selected_turns.append(turn)

    chats = []
    for turn in selected_turns:
        chat_history = ""
        for message in turn:
            if message['role'] == 'user' or (message['role'] == 'assistant' and (probing_question_type == 'information_extraction' or
                                                                                 probing_question_type == 'multi_session_reasoning' or
                                                                                 probing_question_type == 'summarization')):
                if "id" in message.keys():
                    chat_history += f"chat_id: {message['id']}, {message['role'].upper()}: {message['content']} \n\n"
                else:
                    chat_history += f"{message['role'].upper()}: {message['content']} \n\n"

        chat_history += "\n\n"

        token_num = len(_get_token_encoder().encode(chat_history))
        max_tokens = 27000 // len(selected_turns)
        if token_num > max_tokens:
            prompt = f"""Summarize the following conversation in NO MORE than {max_tokens} tokens. 
                    Focus on key decisions, important information, and progress made. Be concise but preserve important context.
                    CRITICAL NOTE: When providing summary, include "chat_id: chat_id_number" for each turn you include in the summary.
                    CRITICAL NOTE: DO NOT COMBINE chat_ids together as far as you can like: chat_id: x-x+5
                    Conversation: {chat_history}
                    NOTE: keep under {max_tokens} tokens.
                    NOTE: Just provide the summary without any explanation before and after the summary."""

            chat_history = llm.invoke(prompt).content.strip()

        chats.append(chat_history)

    bullet_texts = "\n\n".join(bullets)
    chat_texts = "\n\n\n\n".join(chats)

    prompts = []
    if probing_question_type == "information_extraction":
        prompt = information_extraction_probing_question_easy_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = information_extraction_probing_question_medium_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = information_extraction_probing_question_hard_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

    elif probing_question_type == "multi_session_reasoning":
        prompt = multi_session_reasoning_probing_question_easy_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = multi_session_reasoning_probing_question_medium_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = multi_session_reasoning_probing_question_hard_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

    elif probing_question_type == "knowledge_update":
        prompt = knowledge_update_probing_question_final_prompt\
            .replace("<bullet_points>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)
        prompts.append(prompt)

    elif probing_question_type == "temporal_reasoning":
        prompt = temporal_reasoning_probing_question_easy_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = temporal_reasoning_probing_question_medium_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = temporal_reasoning_probing_question_hard_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

    elif probing_question_type == "abstention":
        with open(plan_address, 'rb') as f:
            plans = pickle.load(f)

        original_plan = ""
        for plan_index, plan in enumerate(plans):
            current_plan = ""
            plan_bullets = extract_plan_bullets(plan)
            for bullet_index, bullet in enumerate(plan_bullets):
                current_plan += f"Bullet Number: {bullet_index+1} -> {bullet} \n"

            original_plan += f"BATCH {plan_index+1} PLAN \n" + \
                current_plan + "\n\n"

        prompt = abstention_probing_question_final_prompt\
            .replace("<plan>", original_plan)
        prompts.append(prompt)

    elif probing_question_type == "preference_following":
        prompt = preference_following_probing_question_final_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)
        prompts.append(prompt)

    elif probing_question_type == "event_ordering":
        prompt = event_ordering_probing_question_easy_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = event_ordering_probing_question_medium_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = event_ordering_probing_question_hard_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

    elif probing_question_type == "contradiction_resolution":
        prompt = contradiction_resolution_probing_question_final_prompt\
            .replace("<bullet_points>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)
        prompts.append(prompt)

    elif probing_question_type == "summarization":
        prompt = summarization_probing_question_easy_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = summarization_probing_question_medium_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

        prompt = summarization_probing_question_hard_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)

        prompts.append(prompt)

    elif probing_question_type == "instruction_following":
        prompt = instruction_following_probing_question_final_prompt\
            .replace("<bullet_point>", bullet_texts)\
            .replace("<conversation_turns>", chat_texts)
        prompts.append(prompt)

    datas = []

    for prompt in prompts:
        if variation_instruction:
            prompt += f"\n\nADDITIONAL VARIATION REQUIREMENT: {variation_instruction}"
        response = llm.invoke(prompt).content

        # data = parse_json_response(response=response)
        data = parse_with_repair_library(response=response)
        datas.append(data)

    return datas


def run_probing_question_parallel(plan_address: str,
                                  model: str,
                                  chat_address: str,
                                  save_directory: str,
                                  domain: str,
                                  max_workers: int = 3,
                                  question_types: list[str] | None = None,
                                  generate_abstention: bool = True):

    if question_types is None:
        question_types = ["information_extraction", "multi_session_reasoning", "knowledge_update",
                          "temporal_reasoning", "preference_following", "event_ordering",
                          "contradiction_resolution", "summarization", "instruction_following"]

    errors = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(question_types)))) as executor:
        futures = {
            executor.submit(
                generate_probing_questions_candidate,
                plan_address=plan_address,
                probing_question_type=q,
                llm_name=model,
                chat_address=chat_address,
                save_address=save_directory,
                file_name=q,
                domain=domain
            ): q
            for q in question_types
        }

        for future in as_completed(futures):
            qtype = futures[future]
            try:
                result = future.result()
                # print(f"[{qtype}] done")
            except Exception as e:
                print(f"[{qtype}] generated an exception: {e!r}")
                errors.append((qtype, e))

    if errors:
        failed_types = ", ".join(qtype for qtype, _ in errors)
        raise RuntimeError(f"Probe candidate generation failed for: {failed_types}")

    if generate_abstention:
        response = generate_probing_questions(chat_address=chat_address,
                                              indexes=[],
                                              bullets=[],
                                              probing_question_type="abstention",
                                              llm_name=model,
                                              plan_address=plan_address,
                                              domain=domain)

        abstention_save_address = os.path.join(
            save_directory, "probing_questions/abstention/abstention.json")
        os.makedirs(os.path.dirname(abstention_save_address), exist_ok=True)
        with open(abstention_save_address, 'w') as f:
            json.dump([response], f, indent=4)


def create_probing_questions(chats_directory: str,
                             start_index: int,
                             end_index: int,
                             model: str):

    entries = os.listdir(chats_directory)

    dirs = sorted(
        [name for name in entries if os.path.isdir(
            os.path.join(chats_directory, name))],
        key=lambda x: int(x)
    )

    dirs = dirs[start_index:end_index]

    max_workers = end_index - start_index
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for dir in dirs:
            if os.path.exists(os.path.join(chats_directory, dir, "plan_new_trunecated.pickle")):
                plan_address = os.path.join(
                    chats_directory, dir, "plan_new_trunecated.pickle")
            else:
                plan_address = os.path.join(
                    chats_directory, dir, "plan_new.pickle")
            if os.path.exists(os.path.join(chats_directory, dir, "chat_trunecated.json")):
                chat_address = os.path.join(
                    chats_directory, dir, "chat_trunecated.json")
            else:
                chat_address = os.path.join(chats_directory, dir, "chat.json")
            save_directory = os.path.join(chats_directory, dir)
            topic_address = os.path.join(chats_directory, dir, "topic.json")

            with open(topic_address, "r", encoding="utf-8") as f:
                data = json.load(f)

            category = data["category"].lower()
            if category == "math":
                domain = "math"
            elif category == "coding":
                domain = "coding"
            else:
                domain = "general"

            futures.append(
                executor.submit(
                    run_probing_question_parallel,
                    plan_address=plan_address,
                    model=model,
                    chat_address=chat_address,
                    save_directory=save_directory,
                    domain=domain
                )
            )

        for future in as_completed(futures):
            try:
                result = future.result()
                # print("Task finished:", result)
            except Exception as e:
                print("Task raised exception:", e)
