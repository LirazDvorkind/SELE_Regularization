---
name: teach
description: Teach a concept the user did not understand — from a paper, a figure, a formula, or something said earlier in the conversation. Use when the user says "teach me", "explain", "help me build intuition", "what do you mean by", or asks to learn from a document.
---

# /teach

`$ARGUMENTS` names the concept, or points at a source (a quoted phrase, a paper section, a
figure). If empty, teach the most recent technical claim the user has not acknowledged.

Teach it; do not restate it.

## Order of a good explanation

1. **Vocabulary first.** Every acronym, symbol, and term of art the explanation will use, in
   one table, one line each, before anything else. A reader who cannot decode the words
   cannot follow the ideas.
2. **The concrete setup.** What physically exists, laid out so it can be pictured: geometry,
   layers, inputs, what sits where. A drawn layout or a small table beats prose.
3. **The definition, as a mechanism.** For each quantity, say what happens when something is
   fed into it and what comes out, then name the formal object. Formulas come after the
   picture, never before.
4. **Walk the evidence.** If there is a figure, name it and where it is, then read it feature
   by feature in order — each feature gets its cause. If there is a derivation, do the same
   with its steps. End with the one-sentence intuition that summarizes the walk.
5. **The knobs.** Which parameters control which features. Table.
6. **Connect to the user's own problem.** What carries over, what breaks, what it implies
   for what they are building. Rank by consequence.
7. **Close with what is still open** — the one question this explanation naturally leads to.

## Rules

- Plain English. Any unavoidable technical term is defined in the sentence where it first
  appears, or in the vocabulary table.
- Use the user's own quantities, units, and numbers when they exist in context. Do not
  recompute what is already computed.
- Comparisons carry intuition: put the new thing next to the thing the user already
  understands, and say what is the same and what is different.
- When reading a document, cite page and figure so the user can look alongside. If the
  document's numbering and the file's numbering differ, say how to convert.
- One concept per invocation. If the source contains two, teach the one asked about and
  name the other.
- If an earlier claim was wrong, say so first, then teach the corrected version.
- No "does this make sense?", no menu of further topics.
