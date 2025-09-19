from my_config import config

class PromptBuilder:
    @staticmethod
    def build_rag_prompt(query: str, context: str) -> str:
        """Build optimized RAG prompt template for antimicrobial peptide tasks (faster, more stable)."""
        trimmed_context = context[:config.MAX_CONTEXT_LENGTH]
        return f"""
You are an antimicrobial peptide (AMP) domain assistant. Answer based ONLY on the provided 'Context'; do not speculate. If the context is insufficient, clearly state the limitation and provide next search suggestions.

Requirements:
- Use concise English; return only information relevant to the question; avoid verbose preambles
- Prioritize key information: sequence, MIC (with units/conditions), mechanism of action, source/strain
- Provide citations in the format: [source: filename]
- If context lacks required information, first state 'insufficient', then provide 2 actionable search suggestions (keywords/directions)

Output format (aim for ≤5 lines):
1) Summary: <one-sentence core answer>
2) Key elements: sequence=<...>; MIC=<...>; mechanism=<...>; strain=<...>
3) Evidence sources: [<source: filename>, ...]
4) Limitations & suggestions: <if insufficient, provide; otherwise omit>

Context:
{trimmed_context}

Question: {query}

Answer:
"""

    @staticmethod
    def build_rag_prompt_extraction(query: str, context: str) -> str:
        """Structured RAG prompt focusing on AMP extraction + mechanisms with strict JSON output and example."""
        trimmed_context = context[:config.MAX_CONTEXT_LENGTH]
        prefix = """
You are an antimicrobial peptide (AMP) domain assistant. Use ONLY the provided Context; do not invent facts. If data is missing, mark fields as null and list what is missing with next-step suggestions.

Required coverage:
- Information extraction (peptide-centric)
  For each AMP mentioned: name, sequence, antimicrobial_type (e.g., Gram+, Gram-, fungal, viral), MIC (value + units + strain/assay conditions), hemolysis_concentration (e.g., HC50 or % hemolysis at concentration), physicochemical (length, net_charge@pH7, hydrophobicity_fraction, hydrophobic_moment, pI, helix_propensity).
- Mechanism and methods
  Summarize mechanism(s) supported by evidence; list experimental methods (method, purpose, key readout) and key experimental conditions.

Output must be a single JSON object with these top-level keys:
{
  "extraction": [
    {
      "peptide_name": <string|null>,
      "sequence": <string|null>,
      "antimicrobial_type": <string|null>,
      "MIC": {"value": <number|null>, "units": <string|null>, "strain": <string|null>, "assay": <string|null>, "conditions": <string|null>},
      "hemolysis_concentration": {"metric": <"HC50"|"percent_at">, "value": <number|null>, "units": <string|null>, "condition": <string|null>},
      "physicochemical": {"length": <number|null>, "net_charge_pH7": <number|null>, "hydrophobicity_fraction": <number|null>, "hydrophobic_moment": <number|null>, "pI": <number|null>, "helix_propensity": <string|null>},
      "citations": [{"source": <filename>}]
    }
  ],
  "mechanism_methods": {
    "mechanisms": [{"label": <string>, "evidence": <short quote or paraphrase>, "citations": [{"source": <filename>}]}],
    "experimental_methods": [{"method": <string>, "purpose": <string>, "key_readout": <string>, "citations": [{"source": <filename>}]}]
  },
  "insufficient": {"missing": [<field names>], "search_suggestions": [<2 short actionable suggestions>]}
}


Rules:
- Cite every fact with [{"source": filename}] drawn from Context; if unknown, leave null and list in "insufficient".
- Do NOT include any text outside the JSON object.
"""
        return prefix + f"""

Context:
{trimmed_context}

Question: {query}

Answer (return ONLY the JSON object):
"""
    @staticmethod
    def build_rag_prompt_amp_answer(query: str, context: str) -> str:
        """Natural-language answer: first summarize properties (sequence, MIC, etc.), then mechanisms/methods; cite sources."""
        trimmed_context = context[:config.MAX_CONTEXT_LENGTH]
        return f"""
You are an antimicrobial peptide (AMP) domain assistant. Answer based ONLY on the provided Context about the user's target microbe. Do not speculate; if information is missing, say so and propose next steps.

Instructions:
- Interpret the user's microbe/strain keywords and focus the answer on AMP evidence relevant to that microbe.
- Start with a 1–3 sentence summary naming promising AMP(s) for the target microbe, including sequence and best-evidence MIC with units/assay/strain.
- Then provide a compact bullet list of properties (sequence; MIC with assay/conditions; hemolysis/toxicity; key physicochemical traits if available).
- Follow with a short paragraph on mechanism(s) of action and the experimental methods supporting them.
- Cite all concrete facts inline as [source: filename]. If data is missing, state "insufficient" and give 2 short search suggestions.
- Do not include meta phrases like 'based on the provided context' or 'from the context'.
- Use only level-3 headings (### ...). Do not use # or ## headings.
- Ensure a blank line BEFORE and AFTER every heading, and a blank line BEFORE markdown tables.
- Avoid code fences (``` ... ```), unless explicitly requested.


Answer format (use Markdown only; follow exactly):
- Insert a blank line BEFORE every "###" heading; never concatenate a heading to the end of a sentence.
- Use concise sentences; ASCII units only (ug/mL, uM); avoid meta phrases.

### Summary
<2–4 sentences>

### Candidates
| Peptide | Sequence | MIC (value + units; strain; assay; conditions) | Hemolysis/Toxicity | Physicochemical (length; net_charge@pH7; hydrophobicity_fraction; hydrophobic_moment; pI; helix_propensity) | Sources |
|---|---|---|---|---|---|
| <name> | <... or insufficient> | <... or insufficient> | <... or insufficient> | <... or insufficient> | [source: filename] |

### Mechanisms & Methods
- <mechanism> [source: filename]
- <method: purpose; key readout> [source: filename]

### Limitations & Next steps
- Gaps: <if any>
- Next steps: <exactly 2 short actionable suggestions or omit this section if complete>

Context:
{trimmed_context}

Question: {query}

Answer:
"""

    @staticmethod
    def build_modeling_report_prompt(context: str, query: str) -> str:
        """Generate Section 8 mathematical modeling report (no training process, based on existing system architecture)."""
        trimmed_context = context[:config.MAX_CONTEXT_LENGTH]
        return f"""
You are a professional technical report writing expert. Based on the provided system architecture and implementation details, write a technical report for "Section 8 Mathematical Modeling and Theoretical Foundation".

Requirements:
- Reference the format style of number9.md, but content should be based on the current system's actual architecture
- The system has no training process, focus on describing the mathematical modeling and theoretical foundation of retrieval-generation
- Write in Chinese, maintain academic rigor
- Include mathematical formulas, definitions, propositions and other formal expressions
- Each subsection contains specific mathematical derivations or theoretical analysis

Output format requirements:
- Use only level-3 headings (### 8.1, ### 8.2, etc.)
- Leave blank lines before and after each heading
- Leave blank lines before tables
- Use LaTeX mathematical formulas (\\[ ... \\] or $ ... $)
- Avoid code fences (``` ... ```)

Chapter structure:
### 8.1 Problem Formalization and Symbol System
### 8.2 Mathematical Foundation of Vector Retrieval
### 8.3 Similarity Computation and Ranking Mechanism
### 8.4 Context Aggregation Strategy
### 8.5 Probabilistic Modeling of Generation Model
### 8.6 End-to-End Information Flow Modeling
### 8.7 Complexity Analysis and Performance Modeling
### 8.8 Chapter Summary

Context (System Architecture Information):
{trimmed_context}

Query: {query}

Please generate a complete Section 8 report based on the above requirements:
"""