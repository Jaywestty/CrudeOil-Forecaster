import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

AVAILABLE_SCENARIOS = {
    "opec_cut":             "OPEC or major producer cutting oil supply",
    "global_recession":     "Global economic recession reducing demand",
    "rate_hike":            "Central bank interest rate increases",
    "geopolitical_tension": "Geopolitical conflict disrupting supply routes",
    "demand_boom":          "Strong global demand surge"
}

# Default shock magnitudes per scenario
# The LLM can scale these based on what the user describes
DEFAULT_SHOCKS = {
    "opec_cut": {
        "inventory_pct":  -0.05,
        "vix_diff":       -2.0,
        "dollar_return":  -0.002,
        "indpro_return":   0.0,
        "fed_funds_diff":  0.0
    },
    "global_recession": {
        "indpro_return":  -0.02,
        "vix_diff":        8.0,
        "dollar_return":   0.005,
        "inventory_pct":   0.03,
        "fed_funds_diff":  0.0
    },
    "rate_hike": {
        "dollar_return":   0.008,
        "vix_diff":        3.0,
        "indpro_return":  -0.005,
        "fed_funds_diff":  0.75,
        "inventory_pct":   0.0
    },
    "geopolitical_tension": {
        "inventory_pct":  -0.08,
        "vix_diff":       12.0,
        "dollar_return":   0.003,
        "indpro_return":   0.0,
        "fed_funds_diff":  0.0
    },
    "demand_boom": {
        "indpro_return":   0.015,
        "inventory_pct":  -0.04,
        "vix_diff":       -3.0,
        "dollar_return":  -0.003,
        "fed_funds_diff":  0.0
    }
}


class LLMExplainer:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get a free key at console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.3-70b-versatile"
        print("✅ LLM Explainer ready")


    # ─────────────────────────────────────────────────────────────────────
    # ORIGINAL METHOD — single scenario parser
    # Used by /simulate endpoint. Unchanged from your working version.
    # ─────────────────────────────────────────────────────────────────────

    def parse_user_query(self, user_query):
        """
        Parse natural language into structured scenario parameters.

        Returns:
        - scenario_key: which of our 5 scenarios fits best
        - magnitude_modifier: how severe (0.5=mild, 1.0=normal, 2.0=extreme)
        - scenario_context: a plain English restatement of what the user
          actually asked — used so explanations reference the real question,
          not just the generic scenario name
        - specific_entity: e.g. "Nigeria", "Russia" — used in explanation
        """
        system_prompt = f"""
            You are a financial analyst mapping oil market questions to scenario parameters.

            Available scenarios and when to use them:
            - opec_cut:             Any supply reduction — OPEC cuts, country stops exporting,
                                    sanctions on producers, pipeline disruptions
            - global_recession:     Any demand collapse — recession, depression,
                                    economic crisis, financial crash, pandemic demand shock
            - rate_hike:            Any monetary tightening — rate hikes, Fed action,
                                    central bank policy, inflation fighting
            - geopolitical_tension: Any conflict or war — Middle East war, Russia conflict,
                                    trade war, sanctions causing supply panic
            - demand_boom:          Any demand surge — China reopening, emerging market growth,
                                    industrial expansion, economic boom

            Return ONLY valid JSON, no explanation, no markdown:
            {{
              "scenario_key": "one of the five keys above",
              "magnitude_modifier": 1.0,
              "confidence": "high/medium/low",
              "reasoning": "why this scenario fits",
              "scenario_context": "restate what the user asked in 1 sentence,
                                    keeping their specific framing e.g.
                                    'Nigeria halting all oil exports to global markets'
                                    NOT 'an oil supply disruption'",
              "specific_entity": "the country/org/event the user mentioned, or null",
              "forecast_weeks": 12
            }}

            magnitude_modifier:
              2.0 = extreme/total/catastrophic/complete halt
              1.5 = major/severe/significant
              1.0 = standard/moderate (default)
              0.5 = mild/slight/modest

            Important: scenario_context must reflect the USER'S SPECIFIC situation,
            not the generic scenario name. This is used to personalize the explanation.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Query: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=400
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            # Guard: LLM occasionally returns a JSON array instead of object.
            # e.g. ["geopolitical_tension", ...] instead of {"scenario_key": ...}
            # If that happens, route straight to the fallback dict below.
            if isinstance(parsed, list):
                raise ValueError("LLM returned a list instead of a dict")

            # Validate scenario key
            if parsed.get("scenario_key") not in AVAILABLE_SCENARIOS:
                parsed["scenario_key"]    = "geopolitical_tension"
                parsed["confidence"]      = "low"
                parsed["scenario_context"] = user_query

            # Ensure scenario_context always exists
            if not parsed.get("scenario_context"):
                parsed["scenario_context"] = user_query

            return parsed

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # JSONDecodeError -> LLM returned malformed JSON
            # KeyError        -> expected key missing from parsed dict
            # ValueError      -> raised because LLM returned a list not dict
            # TypeError       -> .get() called on non-dict (safety net)
            print("  Parse failed -- using fallback")
            return {
                "scenario_key":       "geopolitical_tension",
                "magnitude_modifier":  1.0,
                "confidence":         "low",
                "reasoning":          "Parse failed -- using default",
                "scenario_context":   user_query,
                "specific_entity":    None,
                "forecast_weeks":     12
            }


    # ─────────────────────────────────────────────────────────────────────
    # NEW METHOD — probabilistic multi-scenario parser
    # Used by /simulate-probabilistic endpoint.
    #
    # KEY DIFFERENCE from parse_user_query():
    #   Instead of picking ONE best-fit scenario, the LLM distributes
    #   probability weights across ALL relevant scenarios.
    #
    #   "What if tensions rise AND demand grows?"
    #   → { geopolitical_tension: 0.55, demand_boom: 0.35, opec_cut: 0.10 }
    #
    #   These weights are then adjusted by real macro signals
    #   (VIX, dollar, inventories) inside scenario_engine.py before
    #   being used in the weighted forecast calculation.
    # ─────────────────────────────────────────────────────────────────────

    def parse_query_probabilistic(self, user_query):
        """
        Parse natural language into probability weights across scenarios.

        Returns a dict with:
        - probabilities: {scenario_key: float} summing to 1.0
        - primary_driver: scenario with the highest weight
        - reasoning: one sentence explaining the distribution
        - scenario_context: user's query restated (for explanation step)
        - specific_entity: country/org mentioned, or null
        """
        system_prompt = """
            You are a senior energy market analyst assigning scenario probabilities.

            Given a user query about oil markets, distribute probability weights
            across the most relevant scenarios. Weights must sum to exactly 1.0.
            Only include scenarios with weight above 0.05 — omit irrelevant ones.

            Available scenarios:
            - opec_cut:             Supply cuts by OPEC or major producers
            - global_recession:     Demand collapse from economic downturn
            - rate_hike:            Central bank tightening, dollar strengthening
            - geopolitical_tension: Conflict, sanctions, supply route disruption
            - demand_boom:          Strong demand surge, industrial growth

            Return ONLY valid JSON, no markdown, no explanation:
            {
              "probabilities": {
                "scenario_key": probability_float
              },
              "primary_driver": "the scenario with highest weight",
              "confidence": "high/medium/low",
              "reasoning": "one sentence explaining the probability distribution",
              "scenario_context": "restate what the user asked in 1 sentence",
              "specific_entity": "country/org/event mentioned, or null"
            }

            Examples:
            Query: "What if OPEC cuts and Middle East tensions also rise?"
            → { "opec_cut": 0.55, "geopolitical_tension": 0.45 }

            Query: "Global recession hits while the Fed keeps raising rates"
            → { "global_recession": 0.60, "rate_hike": 0.40 }

            Query: "China reopens and oil demand surges"
            → { "demand_boom": 1.0 }

            Query: "Saudi Arabia or any country increases oil output or production"
            → This is a SUPPLY INCREASE — bearish for prices.
               Map to: { "global_recession": 0.50, "rate_hike": 0.30, "opec_cut": 0.20 }
               Reasoning: Supply glut suppresses prices similarly to demand collapse.
               Do NOT map supply increases to demand_boom.

            Query: "What if there is a ceasefire or peace deal?"
            → This is a TENSION REDUCTION — map to demand_boom + opec_cut reversal:
               { "demand_boom": 0.55, "rate_hike": 0.25, "global_recession": 0.20 }
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Query: {user_query}"}
                ],
                temperature=0.1,
                max_tokens=400
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            # Validate all scenario keys — remove any the LLM invented
            probs = parsed.get("probabilities", {})
            probs = {k: v for k, v in probs.items() if k in AVAILABLE_SCENARIOS}

            # Fallback if nothing valid came back
            if not probs:
                probs = {"geopolitical_tension": 1.0}

            # Normalize so probabilities ALWAYS sum to exactly 1.0
            # This guards against the LLM returning 0.45 + 0.45 = 0.90
            # or 0.60 + 0.60 = 1.20 etc.
            total = sum(probs.values())
            parsed["probabilities"] = {
                k: round(v / total, 4) for k, v in probs.items()
            }

            # Ensure scenario_context always exists
            if not parsed.get("scenario_context"):
                parsed["scenario_context"] = user_query

            return parsed

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            print("  Probabilistic parse failed -- using fallback")
            return {
                "probabilities":    {"geopolitical_tension": 1.0},
                "primary_driver":   "geopolitical_tension",
                "confidence":       "low",
                "reasoning":        "Parse failed — using default",
                "scenario_context": user_query,
                "specific_entity":  None
            }


    # ─────────────────────────────────────────────────────────────────────
    # ORIGINAL METHOD — single scenario explanation
    # Used by /simulate endpoint. Unchanged from your working version.
    # ─────────────────────────────────────────────────────────────────────

    def explain_results(self, result, parsed_query):
        """
        Generate explanation that directly references what the user asked.
        Passes scenario_context into the prompt so the LLM addresses the
        user's specific question, not just the generic scenario category.
        """
        w1  = result['impact_week1']
        w12 = result['impact_week12']

        user_context    = parsed_query.get('scenario_context', result['scenario_name'])
        specific_entity = parsed_query.get('specific_entity', None)

        entity_note = ""
        if specific_entity:
            entity_note = f"""
                Special context: The user asked specifically about {specific_entity}.
                Your explanation must reference {specific_entity} by name and discuss
                its specific role in global oil markets where relevant.
                For example — if Nigeria: mention it produces ~1.5mb/day, is Africa's
                largest producer, exports mainly to Europe and Asia, etc.
                If Russia: mention its role in OPEC+, Urals crude, sanctions context.
                Adapt your knowledge to the specific entity mentioned.
            """

        numbers = f"""
            User's specific scenario: "{user_context}"

            Current Brent price: ${result['current_price']:.2f}/barrel
            Immediate impact (Week 1):  {w1['formatted']}
            Full impact (Week 12):      {w12['formatted']}
            Baseline Week 12:  ${w12['baseline']:.2f}/barrel
            Scenario Week 12:  ${w12['shocked']:.2f}/barrel

            Shocks applied to macro model:
            {json.dumps(result['shocks'], indent=2)}
        """

        system_prompt = f"""
            You are a senior energy economist explaining oil market forecasts.

            CRITICAL RULE: Your explanation must directly address the user's
            SPECIFIC scenario as they framed it. Do NOT give a generic explanation
            about the scenario category. If they asked about Nigeria stopping exports,
            explain what happens when NIGERIA specifically stops exporting — not
            just "a supply disruption."

            {entity_note}

            Structure your response in 3 short paragraphs:

            Paragraph 1 — Bottom line: State the price impact immediately.
            Reference the user's specific scenario by name.

            Paragraph 2 — The mechanism: Explain WHY this happens through
            the specific economic channels at play. Reference which macro
            variables drove the model result (USD, VIX, inventories etc.)

            Paragraph 3 — Historical analog: Name 1 real historical event
            that is similar to THIS specific scenario. Briefly say what
            happened to oil prices then and how it compares to our forecast.

            Rules:
            - Under 220 words total
            - Never invent numbers beyond what is given
            - Use plain English, no jargon without explanation
            - Address the user's scenario specifically, not generically
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": numbers}
                ],
                temperature=0.5,
                max_tokens=450
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Explanation unavailable: {str(e)}"


    # ─────────────────────────────────────────────────────────────────────
    # NEW METHOD — probabilistic multi-scenario explanation
    # Used by /simulate-probabilistic endpoint.
    #
    # KEY DIFFERENCE from explain_results():
    #   Single scenario explains one outcome.
    #   This explains a DISTRIBUTION — expected price, range, which
    #   scenario dominates and why, and what the range implies.
    #   Written in institutional research note style.
    # ─────────────────────────────────────────────────────────────────────

    def explain_probabilistic_results(self, result, parsed_query):
        """
        Generate explanation for a probability-weighted multi-scenario result.

        Tells the user:
        1. What the expected price is and what range to expect
        2. Which scenario is the primary driver and why
        3. What macro signals shifted the probabilities
        4. What needs to happen for the low vs high end to materialise
        """
        user_context   = parsed_query.get("scenario_context", "the described scenario")
        primary        = result["primary_driver"]
        primary_prob   = result["scenario_probabilities"].get(primary, 0)
        scenario_names = result.get("scenario_names", {})

        # Build a readable text summary of the breakdown
        breakdown_text = "\n".join([
            f"  {scenario_names.get(k, k)}: {v*100:.0f}% weight "
            f"→ Week-12 price ${result['scenario_week12'].get(k, 0):.2f}"
            for k, v in result["scenario_probabilities"].items()
        ])

        numbers = f"""
            User query: "{user_context}"

            Current Brent price: ${result['current_price']:.2f}/barrel
            Baseline (no shock): ${result['baseline_week12']:.2f}/barrel

            Probabilistic forecast result:
              Expected price (Week 12): ${result['price_expected']:.2f}/barrel
              Price range:              ${result['price_low']:.2f} – ${result['price_high']:.2f}/barrel
              Primary driver: {scenario_names.get(primary, primary)} ({primary_prob*100:.0f}% weight)

            Scenario breakdown:
            {breakdown_text}

            Macro signal adjustments that were applied:
            {json.dumps(result.get('macro_adjustments', {}), indent=2)}
        """

        system_prompt = """
            You are a senior energy economist explaining a probabilistic oil price forecast.

            CRITICAL RULES — read before writing a single word:

            1. ACCEPT THE MODEL NUMBERS AS GIVEN. Do not override them with
               economic theory. If the model shows a price DROP during a supply
               disruption, explain why that specific model output occurred —
               do not say prices will rise when the numbers show a drop.

            2. NEVER duplicate the low and high end prices. Paragraph 2 must
               state TWO DIFFERENT prices — the exact low end price AND the
               exact high end price from the data given. Read them carefully
               before writing.

            3. DO NOT say "reduced consumption leads to higher prices" or any
               other internally contradictory logic. If demand falls, prices fall.
               If supply falls, prices may rise OR fall depending on what the
               model computed — always follow the numbers.

            Structure — 3 paragraphs:

            Paragraph 1: State the expected price (exact number from data) and
            which scenario dominates and why it was assigned the highest weight
            given the user's query.

            Paragraph 2: "The low end of $[EXACT LOW PRICE] would materialise if
            [specific condition]. The high end of $[EXACT HIGH PRICE] would
            materialise if [different specific condition]."
            These MUST be two different prices and two different conditions.

            Paragraph 3: How did current macro signals (VIX, dollar, inventories)
            shift the probability weights? If no adjustments were made, say so.

            Tone: institutional research note. Under 240 words. Plain English.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": numbers}
                ],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Explanation unavailable: {str(e)}"


    # ─────────────────────────────────────────────────────────────────────
    # ORIGINAL METHOD — single scenario uncertainty note
    # Used by /simulate endpoint. Unchanged from your working version.
    # ─────────────────────────────────────────────────────────────────────

    def generate_uncertainty_note(self, result):
        """Generate a 2-sentence uncertainty caveat for single scenario results."""
        w12 = result['impact_week12']

        prompt = f"""
            Write a 2-sentence uncertainty note for this specific forecast.

            Scenario: {result['scenario_name']}
            Estimated impact: {w12['formatted']} over 12 weeks

            Sentence 1: What specific factors could make this forecast wrong
                        for THIS scenario in particular.
            Sentence 2: What the model cannot capture about this type of event.

            Be specific, not boilerplate. Under 60 words.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=120
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return "Model assumes historical relationships remain stable. Structural breaks may cause actual outcomes to differ significantly."


    # ─────────────────────────────────────────────────────────────────────
    # NEW METHOD — probabilistic uncertainty note
    # Specific to the /simulate-probabilistic endpoint.
    # Highlights the key limitation: LLM probabilities ≠ market probabilities.
    # ─────────────────────────────────────────────────────────────────────

    def generate_probabilistic_uncertainty_note(self, result):
        """
        Generate a 2-sentence uncertainty note for probabilistic results.

        Deliberately honest about the fact that probabilities are LLM-estimated
        from query language + macro signals, not derived from options market data.
        This transparency is a feature, not a weakness — it shows you understand
        exactly where the system sits on the research-to-production spectrum.
        """
        prompt = f"""
            Write a 2-sentence uncertainty note for a PROBABILISTIC oil price forecast.

            Expected price: ${result['price_expected']:.2f}/barrel
            Range: ${result['price_low']:.2f} – ${result['price_high']:.2f}/barrel
            Primary driver: {result['primary_driver']}

            Sentence 1: What specific conditions could cause the outcome to fall
                        OUTSIDE the stated price range entirely.
            Sentence 2: The key limitation — scenario probabilities are LLM-estimated
                        from query language and macro signals, not derived from options
                        market implied volatility or historical scenario frequency data.
                        In a production system, those would replace the LLM estimates.

            Be specific and honest. Under 65 words.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=130
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return (
                "Scenario probabilities are LLM-estimated and not derived from "
                "options market data. Actual outcomes may fall outside the stated "
                "range if multiple scenarios materialise simultaneously at greater "
                "severity than assumed."
            )