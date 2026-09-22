"""Turning a typed decision (noul / choice / score) into Shieldstral reads.

Shieldstral answers one kind of question: does this document meet the requirement
in this query, yes or no. Everything here reduces the three Jev question types to
that primitive, the way djev's `independent_levels` mode does for scores:

  noul    one read; P(yes) = sigmoid(yes - no)
  choice  one read per option ("is the correct answer X?"); the per-option log-odds
          are combined with a softmax, so the options compete
  score   the same as choice, over levels

Two prompt layouts:

  card      <Instruct> <Query> <Document>, the model card's order. Every option
            re-reads the whole document.
  docfirst  <Instruct> <Document> <Query>. The instruction and document are a
            shared prefix, prefilled once, and each option adds only its query
            (about 30 tokens). This is what makes a 9-way choice over a 4k-token
            policy affordable on a CPU. It is outside the trained format, which is
            why it is measured rather than assumed.

A read is (prefix_text, suffix_text). Reads for one decision share the prefix.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruction provided. Note that the answer can only be "yes" or "no".'
)
SYSTEM_BLOCK = f"[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT]"

INSTRUCT_PREAMBLE = ("You are a careful decision engine. Answer strictly from the facts in the Document, "
                     "applying every rule, condition and exception it states.")


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def option_lines(qtype: str, labels: list[str], criteria) -> list[tuple[str, str]]:
    """(label, description) per option, in label order."""
    out = []
    for i, lab in enumerate(labels):
        if qtype == "score":
            desc = criteria[i] if isinstance(criteria, list) and i < len(criteria) else f"level {lab}"
            out.append((lab, desc))
        else:
            desc = (criteria or {}).get(lab, "") if isinstance(criteria, dict) else ""
            out.append((lab, desc))
    return out


def build_reads(q: dict, labels: list[str], state, layout: str = "docfirst", style: str = "listed",
                noul: str = "contrast"):
    """
    Returns (prefix, suffixes, kind). `kind` says how to turn the verdicts into probabilities:
    "binary" (one read, noul) or "softmax" (one read per label, in label order).

    noul="direct" asks the question itself and reads P(yes). That read carries a topical
    bias: a document that is merely *about* the question leans yes. noul="contrast"
    turns a noul into a two-option choice between its false and true criteria, so both
    reads are equally on-topic and the bias cancels in the softmax.
    """
    qtype = q["type"]
    if qtype == "noul" and noul == "contrast":
        c = q.get("criteria") or {}
        q = {"type": "choice", "instructions": q.get("instructions", ""),
             "criteria": {"no": c.get("false") or "no, the statement does not hold",
                          "yes": c.get("true") or "yes, the statement holds"}}
        qtype, labels = "choice", ["no", "yes"]
    ins = q.get("instructions", "").strip()
    crit = q.get("criteria")
    doc = state_text(state)

    if qtype == "noul":
        c = crit or {}
        instruct = f"{INSTRUCT_PREAMBLE}\nQuestion: {ins}"
        if c.get("true") or c.get("false"):
            instruct += (f"\nAnswer yes if: {c.get('true', 'the statement holds')}"
                         f"\nAnswer no if: {c.get('false', 'the statement does not hold')}")
        queries = [ins]
        kind = "binary"
    else:
        opts = option_lines(qtype, labels, crit)
        noun = "level" if qtype == "score" else "option"
        instruct = f"{INSTRUCT_PREAMBLE}\nQuestion: {ins}"
        if style == "listed":
            instruct += "\nExactly one of these answers is correct:"
            for lab, desc in opts:
                instruct += f"\n- {noun} {lab}: {desc}" if desc else f"\n- {noun} {lab}"
        queries = []
        for lab, desc in opts:
            d = f" ({desc})" if desc else ""
            queries.append(f"Is {noun} {lab}{d} the correct answer to the question?")
        kind = "softmax"

    if layout == "docfirst":
        prefix = f"{SYSTEM_BLOCK}[INST]<Instruct>: {instruct}\n\n<Document>: {doc}"
        suffixes = [f"\n\n<Query>: {qq}[/INST]" for qq in queries]
    elif layout == "card":
        prefix = f"{SYSTEM_BLOCK}[INST]<Instruct>: {instruct}"
        suffixes = [f"\n\n<Query>: {qq}\n\n<Document>: {doc}[/INST]" for qq in queries]
    else:
        raise ValueError(layout)
    return prefix, suffixes, kind


def build_list_read(q: dict, labels: list[str], state):
    """A single read that lists lettered options and asks for the letter (for comparison only)."""
    qtype = q["type"]
    opts = option_lines(qtype, labels, q.get("criteria")) if qtype != "noul" else [
        ("no", (q.get("criteria") or {}).get("false", "")), ("yes", (q.get("criteria") or {}).get("true", ""))]
    letters = [chr(65 + i) for i in range(len(opts))]
    body = "\n".join(f"{L}. {lab}: {desc}" if desc else f"{L}. {lab}" for L, (lab, desc) in zip(letters, opts))
    instruct = f"{INSTRUCT_PREAMBLE}\nQuestion: {q.get('instructions', '')}\nOptions:\n{body}"
    prefix = f"{SYSTEM_BLOCK}[INST]<Instruct>: {instruct}\n\n<Document>: {state_text(state)}"
    suffix = "\n\n<Query>: Which option letter is correct? Reply with the letter only.[/INST]"
    return prefix, suffix, letters
