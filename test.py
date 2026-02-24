import sys
from scenario_engine import ScenarioEngine, SCENARIOS
from llm_explainer   import LLMExplainer
from utils           import save_output
from datetime        import datetime


def print_banner():
    print("\n" + "="*60)
    print("    OIL PRICE SCENARIO FORECASTING SYSTEM")
    print("   Powered by SARIMAX + LLaMA 3.3 (Groq)")
    print("="*60 + "\n")


def print_full_report(result, explanation, uncertainty):
    """Print a complete formatted report to the terminal."""

    w1  = result['impact_week1']
    w12 = result['impact_week12']

    report = []
    report.append("="*60)
    report.append(f"SCENARIO: {result['scenario_name']}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("="*60)

    report.append(f"\n CURRENT BRENT PRICE: ${result['current_price']:.2f}/barrel")
    report.append(f"\n SCENARIO DESCRIPTION:")
    report.append(f"   {result['scenario_desc']}")

    report.append(f"\n PRICE IMPACT SUMMARY:")
    report.append(f"   Immediate (Week 1):  {w1['formatted']}")
    report.append(f"   Full Impact (Week 12): {w12['formatted']}")

    report.append(f"\n WEEK-BY-WEEK FORECAST:")
    report.append(f"   {'Week':<6} {'Baseline':>10} {'Scenario':>10} {'Δ Price':>10}")
    report.append(f"   {'-'*40}")

    for i, (b, s) in enumerate(zip(
        result['baseline_forecast'],
        result['shocked_forecast']
    ), 1):
        diff = s - b
        sign = "+" if diff >= 0 else ""
        report.append(
            f"   Week {i:<2}  ${b:>8.2f}  ${s:>8.2f}  {sign}${diff:>7.2f}"
        )

    report.append(f"\n ECONOMIC EXPLANATION:")
    report.append(f"{explanation}")

    report.append(f"\n  UNCERTAINTY NOTE:")
    report.append(f"{uncertainty}")

    report.append("\n" + "="*60)

    full_report = "\n".join(report)
    print(full_report)
    return full_report


def run_interactive():
    """Main interactive loop."""

    print_banner()

    # Initialize both engines once at startup
    print("Loading system components...")
    engine   = ScenarioEngine()
    explainer = LLMExplainer()

    print("\n System ready. Type 'quit' to exit, 'list' to see scenarios.\n")

    while True:
        print("-"*60)
        user_input = input(" Enter your scenario query:\n> ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("\nGoodbye! ")
            break

        if user_input.lower() == 'list':
            engine.list_scenarios()
            continue

        print(f"\n  Parsing your query...")

        # Step 1 — LLM parses natural language into structured params
        parsed = explainer.parse_user_query(user_input)

        print(f"   Identified scenario: [{parsed['scenario_key']}]")
        print(f"   Confidence: {parsed['confidence']}")
        print(f"   Reasoning: {parsed['reasoning']}")

        if parsed['confidence'] == 'low':
            print(f"\n  Low confidence match. Proceeding with best guess.")
            confirm = input("   Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                continue

        # Step 2 — Apply magnitude modifier to shocks if user specified intensity
        scenario_key = parsed['scenario_key']
        modifier     = parsed.get('magnitude_modifier', 1.0)

        # Temporarily scale shocks by modifier
        from scenario_engine import SCENARIOS as SCENARIO_DEFS
        import copy
        modified_scenario = copy.deepcopy(SCENARIO_DEFS[scenario_key])
        for var in modified_scenario['shocks']:
            modified_scenario['shocks'][var] *= modifier

        # Inject modified shocks back temporarily
        original_shocks = SCENARIO_DEFS[scenario_key]['shocks'].copy()
        SCENARIO_DEFS[scenario_key]['shocks'] = modified_scenario['shocks']

        # Step 3 — Run the scenario
        forecast_weeks = parsed.get('forecast_weeks', 12)
        result = engine.run_scenario(scenario_key, forecast_weeks)
        engine.print_results(result)

        # Restore original shocks
        SCENARIO_DEFS[scenario_key]['shocks'] = original_shocks

        # Step 4 — LLM explains the results
        print("\n  Generating economic explanation...")
        explanation  = explainer.explain_results(result, parsed)
        uncertainty  = explainer.generate_uncertainty_note(result)

        # Step 5 — Print full report
        full_report = print_full_report(result, explanation, uncertainty)

if __name__ == "__main__":
    run_interactive()