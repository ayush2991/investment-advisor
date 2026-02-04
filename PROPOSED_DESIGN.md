# Proposed Multi-Agent Design: Collaborative Investment Committee (CIC)

## 1. Overview
The current PortfolioPulse architecture uses a "Map-Reduce" pattern: specialized agents extract information in silos, and a synthesis agent combines them. While efficient, this lacks the collaborative depth, cross-verification, and critical thinking found in institutional investment committees.

The **Collaborative Investment Committee (CIC)** design shifts from a linear pipeline to an iterative, collaborative framework where agents can challenge each other and share context dynamically.

---

## 2. New Agent Personas

### A. The Committee Chair (Orchestrator)
- **Role**: Workflow management and quality control.
- **Instructions**: Analyze the initial stock/portfolio data and define the "Research Agenda." If a stock is a growth stock, prioritize valuation and future earnings; if it's a utility, prioritize dividend safety and interest rate sensitivity.
- **Improvement**: Moves away from "one-size-fits-all" analysis to context-aware research.

### B. Macro & Sector Strategist (New)
- **Role**: Top-down analysis.
- **Instructions**: Analyze how broader economic trends (inflation, rates, sector rotation) affect the specific asset.
- **Improvement**: Bridges the gap between company-specific data and the broader market environment.

### C. The Devil's Advocate (Risk & Critique)
- **Role**: Institutional Skepticism.
- **Instructions**: Specifically look for flaws in the findings of the Fundamentals and Technicals agents. Identify "Tail Risks" and "Bear Case" scenarios.
- **Improvement**: Prevents AI confirmation bias and provides a more balanced "Risk/Reward" perspective.

### D. Lead Advisor (Synthesis)
- **Role**: Final Client Briefing.
- **Instructions**: Synthesize the "Specialist Debate" into a final recommendation. Must explicitly address the Devil's Advocate's concerns.
- **Improvement**: Produces more robust, nuanced advice that reflects a range of possible outcomes.

---

## 3. Collaborative Workflow (The "Committee Meeting")

### Stage 1: Discovery & Agenda Setting
The **Committee Chair** reviews raw data and generates a "Research Brief" for the specialists, highlighting specific areas of concern (e.g., "Analyze the sustainability of this 20% revenue growth given the rising debt load").

### Stage 2: Parallel Specialist Analysis (Shared Context)
Specialists (Fundamentals, Technicals, Macro) run in parallel but have access to a **Shared Workspace**.
- *Example*: The Technical Agent sees the Macro Agent's note about an upcoming Fed meeting and adjusts its "Volatility" outlook accordingly.

### Stage 3: The "Challenge" Phase
The **Devil's Advocate** reviews the specialists' outputs and generates a "Critique Report."
- *Example*: "The Fundamentals Agent is optimistic about the new product launch, but news sentiment shows significant supply chain delays that haven't been factored into the earnings projection."

### Stage 4: Synthesis & Final Recommendation
The **Lead Advisor** receives the Specialist Reports AND the Critique Report. The final output is structured as:
1.  **Consensus View** (The Bull Case).
2.  **The Counter-Point** (The Devil's Advocate's view).
3.  **Synthesis & Verdict** (The final balanced judgment).

---

## 4. Technical Implementation Strategy

1.  **State Management**: Use a shared "Thread Context" or "Blackboard" object that is passed and updated between agent calls.
2.  **Iterative Calls**: Utilize the `Runner.run()` capability to allow the Chair to request "Deep Dives" if a specialist's confidence score is low.
3.  **Tool Augmentation**: Allow agents to request additional specific data (e.g., "Get Insider Trading data") via the Chair.

---

## 5. Expected Improvements
- **Higher Accuracy**: Cross-verification between agents reduces "hallucinations" and data misinterpretation.
- **Contextual Nuance**: The Macro agent provides "The Big Picture" that is currently missing.
- **Better Risk Management**: The Devil's Advocate ensures that downside risks are never ignored.
- **Adaptive Research**: The Chair ensures that the analysis is tailored to the specific type of asset being analyzed.
