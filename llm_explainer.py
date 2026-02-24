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
        "inventory_pct": -0.05,
        "vix_diff": -2.0,
        "dollar_return": -0.002,
        "indpro_return": 0.0,
        "fed_funds_diff": 0.0
    },
    "global_recession": {
        "indpro_return": -0.02,
        "vix_diff": 8.0,
        "dollar_return": 0.005,
        "inventory_pct": 0.03,
        "fed_funds_diff": 0.0
    },
    "rate_hike": {
        "dollar_return": 0.008,
        "vix_diff": 3.0,
        "indpro_return": -0.005,
        "fed_funds_diff": 0.75,
        "inventory_pct": 0.0
    },
    "geopolitical_tension": {
        "inventory_pct": -0.08,
        "vix_diff": 12.0,
        "dollar_return": 0.003,
        "indpro_return": 0.0,
        "fed_funds_diff": 0.0
    },
    "demand_boom": {
        "indpro_return": 0.015,
        "inventory_pct": -0.04,
        "vix_diff": -3.0,
        "dollar_return": -0.003,
        "fed_funds_diff": 0.0
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
        print(" LLM Explainer ready")


    def parse_user_query(self, user_query):
        """
        Parse natural language into structured scenario parameters.

        Now returns:
        - scenario_key: which of our 5 scenarios fits best
        - magnitude_modifier: how severe (0.5=mild, 1.0=normal, 2.0=extreme)
        - scenario_context: a plain English description of what the user
          ACTUALLY asked — this gets passed to the explainer so it can
          reference the real question, not just the generic scenario name
        - specific_entity: e.g. "Nigeria", "Russia", "China" — used in explanation
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

            # Validate
            if parsed.get("scenario_key") not in AVAILABLE_SCENARIOS:
                parsed["scenario_key"]      = "geopolitical_tension"
                parsed["confidence"]         = "low"
                parsed["scenario_context"]   = user_query

            # Ensure scenario_context exists
            if not parsed.get("scenario_context"):
                parsed["scenario_context"] = user_query

            return parsed

        except (json.JSONDecodeError, KeyError):
            print("  Parse failed — using fallback")
            return {
                "scenario_key":      "geopolitical_tension",
                "magnitude_modifier": 1.0,
                "confidence":        "low",
                "reasoning":         "Parse failed — using default",
                "scenario_context":  user_query,
                "specific_entity":   None,
                "forecast_weeks":    12
            }


    def explain_results(self, result, parsed_query):
        """
        Generate explanation that directly references what the user asked.

        The key fix: we pass scenario_context (the user's actual question)
        into the prompt and REQUIRE the LLM to address it specifically.
        This prevents generic boilerplate answers.
        """

        w1  = result['impact_week1']
        w12 = result['impact_week12']

        # The user's actual question framing
        user_context   = parsed_query.get('scenario_context', 
                                           result['scenario_name'])
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


    def generate_uncertainty_note(self, result):
        """Generate uncertainty caveat referencing the specific scenario."""

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
