"""
SafeMum AI — Dataset Interpreter
Loads unsupervised findings, sends to Groq, saves a reusable
knowledge document that any part of the app can import.

Usage from any Flask route:
    from SafeMumApp.Ai_Analysis.dataset_interpreter import get_dataset_knowledge
    knowledge = get_dataset_knowledge()
    # knowledge["summary"]         — plain text overview
    # knowledge["risk_factors"]    — list of key risk factors found
    # knowledge["population"]      — who these women are
    # knowledge["care_patterns"]   — care-seeking behaviour patterns
    # knowledge["chw_insights"]    — CHW performance patterns
    # knowledge["cultural_notes"]  — cultural context
    # knowledge["raw_findings"]    — full structured data
"""

import os
import json
import joblib
from dotenv import load_dotenv
load_dotenv()


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
FINDINGS_PATH    = os.path.join(MODELS_DIR, "unsupervised_findings.joblib")
KNOWLEDGE_PATH   = os.path.join(MODELS_DIR, "dataset_knowledge.joblib")
KNOWLEDGE_TXT    = os.path.join(MODELS_DIR, "dataset_knowledge.txt")

_knowledge_cache = None


def _build_prompt(findings: dict) -> str:
    """Build a structured prompt from the findings for Groq."""

    summary_lines = []
    for name, f in findings.items():
        stats = f.get("stats", {})
        clustering = f.get("clustering")
        pca = f.get("pca")

        lines = [f"\n--- Dataset: {name} ---"]
        lines.append(f"Rows: {stats.get('row_count')} | Columns: {stats.get('column_count')}")
        lines.append(f"Missing data: {stats.get('missing_pct')}%")

        cols = stats.get("columns", [])
        lines.append(f"Columns (first 30): {cols[:30]}")

        corrs = stats.get("top_correlations", [])
        if corrs:
            lines.append("Top correlations:")
            for c in corrs[:5]:
                lines.append(f"  {c['col_a']} ↔ {c['col_b']}: {c['correlation']}")

        num_sum = stats.get("numeric_summary", {})
        if num_sum:
            lines.append("Key numeric summaries:")
            for col, vals in list(num_sum.items())[:8]:
                lines.append(f"  {col}: mean={vals.get('mean')}, std={vals.get('std')}, min={vals.get('min')}, max={vals.get('max')}")

        cat_dist = stats.get("categorical_distributions", {})
        if cat_dist:
            lines.append("Categorical distributions (top values):")
            for col, dist in list(cat_dist.items())[:5]:
                lines.append(f"  {col}: {dist}")

        if clustering:
            lines.append(f"Clustering ({clustering['n_clusters']} clusters):")
            for cid, profile in clustering.get("cluster_profiles", {}).items():
                lines.append(f"  {cid} (n={profile['size']}): top features = {profile['top_features']}")

        if pca:
            lines.append(f"PCA: {round(pca['total_variance_explained']*100, 1)}% variance explained by top components")
            for comp in pca.get("components", [])[:2]:
                lines.append(f"  PC{comp['component']}: {comp['top_features'][:3]}")

        summary_lines.extend(lines)

    data_summary = "\n".join(summary_lines)

    prompt = f"""You are a maternal health data scientist analysing research datasets from Sub-Saharan Africa (primarily Kenya).

Below are statistical findings extracted from {len(findings)} CSV datasets covering pregnancy loss, community health workers, facility delivery, cultural interventions, and social support networks.

{data_summary}

Based on this data, provide a structured interpretation in valid JSON format with exactly these keys:

{{
  "summary": "2-3 paragraph plain English overview of what these datasets collectively tell us about maternal health in this region",
  "population": "Description of who these women are — demographics, location, circumstances",
  "risk_factors": ["list of the most important risk factors for maternal complications found in the data"],
  "care_patterns": "How women seek care, what barriers exist, what predicts follow-through on referrals",
  "chw_insights": "What the CHW data reveals about community health worker effectiveness and patterns",
  "cultural_notes": "Cultural and social factors that influence health outcomes based on the data",
  "cluster_meanings": "What the natural groupings/clusters in the data represent about different patient subpopulations",
  "app_guidance": "Specific guidance for how a maternal health AI app should use these insights to personalise support, flag risk, and route care",
  "key_statistics": {{
    "stat_name": "stat_value"
  }}
}}

Return ONLY valid JSON. No markdown, no explanation outside the JSON."""

    return prompt


def _build_prompt_for_batch(batch: dict, batch_num: int, total: int) -> str:

    # Dataset context so Groq understands what it's looking at
    DATASET_CONTEXT = {
        "ddi_pds_data":      "3,215 women who experienced pregnancy loss in Kenya. Columns pds207a-n are complication flags (bleeding, infection, sepsis, shock etc). pds101=age, pds102=urban/rural, pds201=previous pregnancies, pds202=previous losses.",
        "ddi_hfs_data":      "328 health facilities in Kenya. Contains facility level, ownership, services available including post-abortion care volumes.",
        "ddi_hps_data":      "124 health providers. Contains case volumes, referral patterns, method knowledge.",
        "woman_final":       "1,001 pregnant women in Nairobi informal settlements. indicator7a=followed referral, indicator7b=received postnatal care, indicator6=4+ antenatal visits, indicator8=skilled birth attendant.",
        "chv_final":         "127 community health volunteers. Contains households visited, referrals made, follow-up outcomes, performance indicators.",
        "AKU_baseline":      "406 women before a cultural health intervention in Garissa Kenya. Contains ethnicity, religion, marital status, education, household conditions, delivery location history.",
        "AKU_endline":       "719 women after the cultural intervention. Same columns plus 192 additional behaviour change indicators.",
        "pamanech_woman_data": "849 women in Korogocho and Kariobangi Nairobi. Delivery location, antenatal visits, complications, socioeconomic profile.",
        "pamanech_child_data": "987 children linked to those women. Birth weight, child death, cause of death, growth status.",
        "W1 Mother Focal Child File-ANON": "446 single mothers in Nairobi Wave 1. Crisis_ML=crisis support score, Wealthscore=1-5 wealth index, kinship support scores, 120 child development milestones.",
        "W2 Mother Focal Child File-ANON": "411 single mothers Wave 2 follow-up. Same structure as Wave 1.",
        "W1 KST Member File-ANON":  "5,361 kinship network members of those mothers. Relationship type, visit frequency, support type.",
        "W2 KST Member File-ANON":  "4,800 kinship network members Wave 2.",
        "W1 Combined Children File-ANON": "972 children from Wave 1 mothers.",
        "W2 Combined Children File-ANON": "952 children from Wave 2 mothers.",
        "W1 Mother Union History File-ANON": "131 relationship history records Wave 1.",
        "W2 Mother Union History File-ANON": "159 relationship history records Wave 2.",
    }

    summary_lines = []
    for name, f in batch.items():
        stats = f.get("stats", {})
        clustering = f.get("clustering")

        lines = [f"\n--- Dataset: {name} ---"]

        # Add human context if available
        ctx = DATASET_CONTEXT.get(name, "")
        if ctx:
            lines.append(f"CONTEXT: {ctx}")

        lines.append(f"Rows: {stats.get('row_count')} | Columns: {stats.get('column_count')}")
        lines.append(f"Missing data: {stats.get('missing_pct')}%")

        corrs = stats.get("top_correlations", [])
        if corrs:
            lines.append("Top correlations:")
            for c in corrs[:3]:
                lines.append(f"  {c['col_a']} ↔ {c['col_b']}: {c['correlation']}")

        num_sum = stats.get("numeric_summary", {})
        if num_sum:
            lines.append("Key numeric summaries:")
            for col, vals in list(num_sum.items())[:5]:
                lines.append(f"  {col}: mean={vals.get('mean')}, std={vals.get('std')}")

        if clustering:
            lines.append(f"Clustering ({clustering['n_clusters']} clusters):")
            for cid, profile in clustering.get("cluster_profiles", {}).items():
                lines.append(f"  {cid} (n={profile['size']}): top features = {profile['top_features'][:3]}")

        summary_lines.extend(lines)

    data_summary = "\n".join(summary_lines)

    return f"""You are a maternal health data scientist analysing real research datasets from Sub-Saharan Africa focused on pregnancy loss, maternal complications, and community health.

This is batch {batch_num} of {total}. Each dataset has a CONTEXT explanation of what it contains.

{data_summary}

Based on the actual clinical and social meaning of these datasets, return a JSON object:
{{
  "datasets_covered": ["list of dataset names"],
  "key_findings": ["5-8 specific clinical or social findings — e.g. what % of women had complications, what predicts referral follow-through, which factors correlate with mortality"],
  "risk_factors": ["clinical and social risk factors identified"],
  "population_notes": "concrete description of who these women are and their circumstances",
  "patterns": "specific patterns — e.g. urban vs rural differences, age correlations, facility delivery predictors"
}}

Be specific and clinical. Reference actual variable meanings, not column codes. Return ONLY valid JSON."""



def build_knowledge(force=False):
    global _knowledge_cache

    if not force and os.path.exists(KNOWLEDGE_PATH):
        print("[SafeMum AI] Dataset knowledge already exists. Use force=True to regenerate.")
        _knowledge_cache = joblib.load(KNOWLEDGE_PATH)
        return _knowledge_cache

    if not os.path.exists(FINDINGS_PATH):
        print("[SafeMum AI] ERROR: unsupervised_findings.joblib not found.")
        print("Run train_unsupervised.py first.")
        return None

    print("[SafeMum AI] Loading unsupervised findings...")
    findings = joblib.load(FINDINGS_PATH)

    try:
        from groq import Groq
        import time
        client = Groq()

        # Split into batches of 3 datasets each
        items = list(findings.items())
        batch_size = 3
        batches = [
            dict(items[i:i+batch_size])
            for i in range(0, len(items), batch_size)
        ]

        print(f"[SafeMum AI] Processing {len(batches)} batches of up to {batch_size} datasets each...")

        batch_results = []
        for idx, batch in enumerate(batches):
            print(f"  Batch {idx+1}/{len(batches)}: {list(batch.keys())}")
            prompt = _build_prompt_for_batch(batch, idx+1, len(batches))

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw = response.choices[0].message.content.strip()

                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    import re
                    match = re.search(r'\{.*\}', raw, re.DOTALL)
                    result = json.loads(match.group()) if match else {"raw": raw}

                batch_results.append(result)
                print(f"    Done.")

            except Exception as e:
                print(f"    Batch {idx+1} failed: {e}")
                batch_results.append({"error": str(e)})

            # Wait between batches to avoid rate limits
            if idx < len(batches) - 1:
                time.sleep(5)

        # Merge all batch results into final knowledge document
        print("\n[SafeMum AI] Merging batch results...")

        all_findings_list = []
        all_risk_factors  = []
        all_population    = []
        all_patterns      = []

        for r in batch_results:
            findings_raw = r.get("key_findings", [])
            if isinstance(findings_raw, list):
                all_findings_list.extend([str(x) for x in findings_raw])
            
            risks_raw = r.get("risk_factors", [])
            if isinstance(risks_raw, list):
                all_risk_factors.extend([str(x) for x in risks_raw])
            
            pop = r.get("population_notes", "")
            if isinstance(pop, list):
                pop = " ".join([str(x) for x in pop])
            if pop:
                all_population.append(str(pop))
            
            pat = r.get("patterns", "")
            if isinstance(pat, list):
                pat = " ".join([str(x) for x in pat])
            if pat:
                all_patterns.append(str(pat))

        # Final synthesis call
        print("[SafeMum AI] Running final synthesis...")
        synthesis_prompt = f"""You are a maternal health AI system for Sub-Saharan Africa.

Based on analysis of 17 research datasets covering 6,500+ women, here are the findings:

Key findings: {json.dumps(all_findings_list[:20])}
Risk factors: {json.dumps(list(set(all_risk_factors))[:15])}
Population notes: {' '.join(all_population[:3])[:500]}
Patterns: {' '.join(all_patterns[:3])[:500]}

Write a final structured JSON knowledge document:
{{
  "summary": "3 sentence overview of what these datasets tell us about maternal health in this region",
  "population": "who these women are",
  "risk_factors": ["deduplicated list of top 10 risk factors"],
  "care_patterns": "how women seek care and what predicts follow-through",
  "chw_insights": "what CHW data reveals about community health worker effectiveness",
  "cultural_notes": "cultural and social factors influencing outcomes",
  "cluster_meanings": "what natural groupings in the data represent",
  "app_guidance": "how a maternal health AI app should use these insights",
  "key_statistics": {{
    "total_patients": "3215",
    "total_facilities": "328",
    "total_chws": "127"
  }}
}}

Return ONLY valid JSON."""

        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )

        final_raw = final_response.choices[0].message.content.strip()
        try:
            knowledge = json.loads(final_raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', final_raw, re.DOTALL)
            knowledge = json.loads(match.group()) if match else {"raw_text": final_raw}

        # Attach raw findings
        knowledge["raw_findings"] = {
            name: {"stats": f.get("stats", {}), "clustering": f.get("clustering")}
            for name, f in findings.items()
        }
        knowledge["batch_results"] = batch_results

        # Save
        joblib.dump(knowledge, KNOWLEDGE_PATH)

        with open(KNOWLEDGE_TXT, "w", encoding="utf-8") as f:
            f.write("SAFEMUM AI — DATASET KNOWLEDGE DOCUMENT\n")
            f.write("="*60 + "\n\n")
            for key in ["summary", "population", "risk_factors", "care_patterns",
                        "chw_insights", "cultural_notes", "cluster_meanings",
                        "app_guidance", "key_statistics"]:
                val = knowledge.get(key, "")
                if not val:
                    continue
                f.write(f"\n{key.upper().replace('_', ' ')}\n")
                f.write("-"*40 + "\n")
                if isinstance(val, list):
                    for item in val:
                        f.write(f"  • {item}\n")
                elif isinstance(val, dict):
                    for k, v in val.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{val}\n")

        print(f"[SafeMum AI] Knowledge saved to {KNOWLEDGE_PATH}")
        print(f"[SafeMum AI] Human-readable version saved to {KNOWLEDGE_TXT}")

        _knowledge_cache = knowledge
        return knowledge

    except Exception as e:
        print(f"[SafeMum AI] Failed: {e}")
        return None

def get_dataset_knowledge() -> dict:
    """
    Get the dataset knowledge for use anywhere in the app.
    Loads from cache if already loaded, otherwise from disk.

    Returns dict with keys:
        summary, population, risk_factors, care_patterns,
        chw_insights, cultural_notes, cluster_meanings,
        app_guidance, key_statistics, raw_findings
    """
    global _knowledge_cache

    if _knowledge_cache is not None:
        return _knowledge_cache

    if os.path.exists(KNOWLEDGE_PATH):
        _knowledge_cache = joblib.load(KNOWLEDGE_PATH)
        return _knowledge_cache

    print("[SafeMum AI] WARNING: dataset_knowledge.joblib not found.")
    print("Run: python train_unsupervised.py then python dataset_interpreter.py")
    return {}


def get_risk_context_for_prompt() -> str:
    """
    Returns a short string suitable for injecting into any Groq prompt
    to give the LLM real data context about this population.
    """
    knowledge = get_dataset_knowledge()
    if not knowledge:
        return ""

    parts = []

    summary = knowledge.get("summary", "")
    if summary:
        parts.append(f"Population context: {summary[:400]}")

    risk_factors = knowledge.get("risk_factors", [])
    if risk_factors:
        parts.append(f"Key risk factors in this population: {'; '.join(risk_factors[:5])}")

    care_patterns = knowledge.get("care_patterns", "")
    if care_patterns:
        parts.append(f"Care-seeking patterns: {care_patterns[:300]}")

    cultural_notes = knowledge.get("cultural_notes", "")
    if cultural_notes:
        parts.append(f"Cultural context: {cultural_notes[:300]}")

    return "\n".join(parts)


# ─── Run directly to build knowledge ──────────────────────────────────────────
if __name__ == "__main__":
    knowledge = build_knowledge(force=True)
    if knowledge:
        print("\n" + "="*60)
        print("KNOWLEDGE DOCUMENT PREVIEW")
        print("="*60)
        for key in ["summary", "population", "risk_factors", "care_patterns"]:
            val = knowledge.get(key, "")
            print(f"\n{key.upper()}:")
            if isinstance(val, list):
                for item in val[:5]:
                    print(f"  • {item}")
            else:
                print(f"  {str(val)[:300]}")