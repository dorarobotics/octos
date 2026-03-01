# Deep Research: Next-Gen Research Pipeline Design

> Status: Design Phase
> Date: 2026-03-01
> Codename: **Deep Research** (flagship offering)

## Background: Current "Deep Search" (v1)

Our current pipeline (`deep_search` + `synthesize_research`) is a **single-pass search-then-summarize** system:

1. LLM calls `deep_search` tool
2. Tool runs 3 search rounds, fetches ~30 pages, saves full content to disk
3. LLM calls `synthesize_research` which reads all files, does map-reduce (80K char batches), returns merged findings
4. LLM formats answer for user

**Limitations vs Gemini/Kimi:**

| Dimension | Our Deep Search v1 | Gemini Deep Research | Kimi K2.5 Agent Swarm |
|---|---|---|---|
| Search queries | ~10 keywords, 3 rounds | 80-160 queries | 1,500 tool calls |
| Pages read | ~30 | 100+ full pages | 206 URLs (top 3.2% retained) |
| Sub-agents | 0 | Orchestrator + parallel | Up to 100 parallel |
| Reflection/gap detection | None | Search → Reflect → Fill gaps loop | RL-learned iterative refinement |
| Synthesis | Single-pass map-reduce | Multi-pass self-critique | RL-optimized report generation |
| Quality filtering | None (all content equal) | Relevance scoring | Only top 3.2% retained |
| Output detail | ~6K chars (sketchy) | Comprehensive structured report | 10,000+ words with 26 citations |
| Time | ~6 min | 5-60 min | 3-5 min (parallel) |

The core problems:
1. **No reflection loop** — never asks "what's missing?" after searching
2. **No topic-aware decomposition** — single generic query, not specialized per angle/language/region
3. **Synthesis = summarization** — map-reduce extracts then merges, no self-critique or quality filtering

---

## Design: Deep Research (Next-Gen)

### Three-Phase Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: PLANNING                        │
│                                                             │
│  User query → Orchestrator LLM generates research plan:     │
│  - Decompose into N topic angles (5-20, not fixed at 3)     │
│  - Per angle: target languages, regions, site types          │
│  - Present plan to user for review/modification              │
│                                                             │
│  Example: "Iran succession crisis"                          │
│  ├─ Persian/Farsi news (local Iranian portals)              │
│  ├─ Arabic news (Al Jazeera, Al Arabiya)                    │
│  ├─ Israeli analysis (Haaretz, Times of Israel)             │
│  ├─ Western analysis (Reuters, BBC, NYT)                    │
│  ├─ Academic/think tank (RAND, Brookings, IISS)             │
│  └─ Social media / OSINT angle                              │
│                                                             │
│  Example: "2026 World Cup predictions"                      │
│  ├─ Spanish sports press (Marca, AS, Mundo Deportivo)       │
│  ├─ Portuguese sports (A Bola, Record)                      │
│  ├─ English sports (BBC Sport, The Athletic, ESPN)          │
│  ├─ Latin American coverage (Ole, Globo Esporte)            │
│  ├─ Statistical/analytics (FBref, Opta, 538)               │
│  ├─ Betting/prediction markets (Betfair, Polymarket)        │
│  └─ FIFA/official sources                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: COLLECTION (Parallel)                 │
│                                                             │
│  5-20 sub-agents run concurrently, each:                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  Sub-Agent N (specialized angle)    │                    │
│  │                                     │                    │
│  │  loop {                             │                    │
│  │    1. Generate targeted queries     │                    │
│  │       (language-specific, site-     │                    │
│  │        specific, region-aware)      │                    │
│  │    2. Search + browse full pages    │                    │
│  │    3. REFLECT:                      │                    │
│  │       - Score relevance (0-10)      │                    │
│  │       - Discard noise (keep top N%) │                    │
│  │       - Detect gaps: "what's        │                    │
│  │         missing for this angle?"    │                    │
│  │    4. If gaps → generate new        │                    │
│  │       queries → loop               │                    │
│  │    5. If sufficient → write         │                    │
│  │       findings to partial_N.md      │                    │
│  │       with citations                │                    │
│  │  }                                  │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  Wall-time = slowest sub-agent (parallel execution)         │
│  Each sub-agent: 3-5 search-reflect cycles                  │
│  Quality gate: relevance filter before synthesis            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          Phase 3: SYNTHESIS (Fresh Context)                 │
│                                                             │
│  Brand new LLM session (no search history burden):          │
│                                                             │
│  1. Read ALL partial findings from sub-agents               │
│  2. Cross-reference across angles:                          │
│     - Detect contradictions between sources                 │
│     - Find consensus across regions/languages               │
│     - Identify unique insights per angle                    │
│  3. Generate structured draft report                        │
│  4. SELF-CRITIQUE:                                          │
│     - Review draft for weak sections                        │
│     - Check citation coverage                               │
│     - Identify remaining gaps                               │
│     - If critical gaps → can request additional search      │
│  5. Final report: comprehensive, data-rich, cited           │
│                                                             │
│  Output: Canonical Markdown report                          │
│  ├─ Can convert to PPTX (existing skill)                    │
│  ├─ Can convert to DOCX (future)                            │
│  ├─ Can convert to website/infographic (future)             │
│  └─ Stored as artifact for future reference                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Dynamic sub-agent count**: Orchestrator decides how many angles based on query complexity (simple factual → 3, geopolitical analysis → 15+). Not hardcoded.

2. **Language/region-aware queries**: The plan explicitly specifies which languages and regional sources to target. A query about Iran should search Farsi, Arabic, Hebrew, and English sources — not just English.

3. **Reflection loop per sub-agent**: Each sub-agent runs search → reflect → search more if gaps detected. 3-5 cycles. This is what makes output comprehensive vs sketchy.

4. **Quality filtering**: Each sub-agent scores content relevance and discards noise before synthesis. Aim for Kimi's ~3-5% retention rate. 200 pages fetched → 7-10 high-quality pages retained.

5. **Fresh synthesis context**: Synthesis agent starts with zero history. Only receives curated findings from collection phase. Prevents context pollution from search process noise.

6. **Report as artifact**: Markdown is the canonical output format. Downstream conversion to PPTX/DOCX/website is a separate concern, not part of the research pipeline.

7. **Self-critique in synthesis**: The synthesis agent reviews its own draft, identifies weak sections, and can request targeted additional searches before finalizing.

---

## Why RL Training Matters (and Why We Don't Need It Yet)

### What RL Training Does for Gemini/Kimi

Kimi-Researcher achieved its results through **end-to-end reinforcement learning** — the model itself learned agentic behaviors (when to search, what queries to generate, when to stop, what to keep) through reward signals, not prompt engineering.

Their RL reward structure:
- **Format reward**: penalizes invalid tool calls or budget overruns
- **Correctness reward**: compares output against verified ground truth
- **Efficiency reward**: `r * gamma^(T-i)` — encourages shorter trajectories (find answers quickly)

Starting from 8.6% on HLE (Humanity's Last Exam), Kimi reached 26.9% through RL alone. The model learned:
- When to issue follow-up searches (gap detection is *emergent*, not programmed)
- What search queries are most productive (74 keywords avg, learned distribution)
- What content to keep vs discard (3.2% retention rate, learned filter)
- When enough information has been gathered (learned stopping criterion)

### Why RL Matters: Prompt Engineering Has a Ceiling

**Prompt-based approach** (what we can do now):
```
"Search for X. After searching, reflect on what gaps remain.
If gaps exist, search again with targeted queries. Repeat 3-5 times."
```

This *works* but has fundamental limitations:

1. **Search query quality**: A prompted model generates "reasonable" queries. An RL-trained model has optimized over millions of trajectories to find queries that *actually produce useful results*. The difference is like a chess player who knows the rules vs one who has played 10 million games.

2. **Stopping criterion**: When should the agent stop searching? A prompted agent uses a fixed rule ("3-5 cycles") or vague judgment. An RL agent has learned the optimal stopping point — the point where additional searches have diminishing returns. It's been rewarded for efficiency.

3. **Quality filtering**: "Keep only relevant content" in a prompt is subjective. An RL agent has learned *exactly* what "relevant" means for producing high-scoring reports — through thousands of training examples with ground truth.

4. **Exploration strategy**: A prompted agent explores uniformly. An RL agent has learned to allocate more search budget to harder sub-topics and less to easy ones. It has a *learned policy* for resource allocation.

5. **Compounding errors**: In a 23-step trajectory (Kimi's average), each step's quality affects the final output. Prompt engineering optimizes each step independently. RL optimizes the *entire trajectory end-to-end*, accounting for how early decisions affect later steps.

### What This Means in Practice

| Capability | Prompt-Based (Us) | RL-Trained (Kimi/Gemini) |
|---|---|---|
| Query generation | Good, generic | Optimized for information yield |
| Gap detection | Explicit "check for gaps" prompt | Emergent from reward signal |
| When to stop searching | Fixed rules (3-5 cycles) | Learned optimal stopping |
| Content filtering | Heuristic or LLM judgment | Learned relevance model |
| Resource allocation | Uniform across sub-topics | Adaptive (more budget for harder topics) |
| Report quality | Good (well-prompted) | Excellent (optimized for report reward) |
| Error compounding | Accumulates over steps | Minimized via trajectory optimization |

### Our Approach: Architecture First, RL Later

**Phase 1 (Now)**: Build the right *architecture* with prompt-based agents:
- Orchestrator + parallel sub-agents + reflection loops + fresh synthesis
- This alone will be a massive improvement over single-pass search-summarize
- Estimated improvement: from sketchy ~6K char summaries to structured ~10K+ word reports

**Phase 2 (Future)**: Add RL training on top of the architecture:
- Collect trajectories from production usage (search logs, user feedback)
- Define reward: report quality (accuracy, completeness, citation correctness)
- Train the orchestrator and sub-agents via REINFORCE/PPO
- This is where we'd close the remaining gap to Kimi/Gemini

The architecture is the prerequisite. Kimi's RL wouldn't work without the right tool interface and agent loop. We build that first.

---

## Implementation Plan

### New Components

1. **`DeepResearchOrchestrator`** (new tool in `crew-agent/src/tools/`)
   - Takes user query + optional configuration
   - Phase 1: Calls LLM to generate research plan (N angles with metadata)
   - Phase 2: Spawns N sub-agents in parallel (reuses existing `Agent` infrastructure)
   - Phase 3: Collects partials, spawns fresh synthesis agent
   - Returns final report + metadata

2. **`ResearchSubAgent`** (configuration for collection phase agents)
   - Specialized system prompt per angle
   - Tools: `web_search`, `web_fetch`, `browser`, `deep_search`, `read_file`, `write_file`
   - Reflection loop built into system prompt
   - Quality filter: score + discard low-relevance content
   - Max iterations: configurable per angle complexity

3. **`SynthesisAgent`** (configuration for final synthesis)
   - Fresh context (no search history)
   - Long context model preferred (Gemini 2.5 Pro 1M, or largest available)
   - Self-critique loop: draft → review → revise
   - Citation validation pass

### Files to Modify

- `crates/crew-agent/src/tools/deep_research_v2.rs` — new orchestrator
- `crates/crew-agent/src/tools/mod.rs` — register new tool
- `crates/crew-agent/src/prompts/research_planner.txt` — orchestrator planning prompt
- `crates/crew-agent/src/prompts/research_collector.txt` — sub-agent collection prompt
- `crates/crew-agent/src/prompts/research_synthesizer.txt` — synthesis prompt
- `crates/crew-cli/src/commands/gateway.rs` — register tool, update system prompt
- `crates/crew-cli/src/commands/chat.rs` — register tool

### Naming

| Component | Name | Description |
|---|---|---|
| Current pipeline | `deep_search` + `synthesize_research` | Single-pass search + map-reduce synthesis |
| Next-gen pipeline | `deep_research` (tool name) | Multi-agent orchestrated research with reflection |
| User-facing brand | **Deep Research** | Flagship research capability |

The existing `DeepResearchTool` will be retired or renamed to avoid confusion. The new tool takes over the `deep_research` name.

---

## Search Source Registry (Prerequisite)

Before building the orchestrator, we need a **structured source database** that maps topics, languages, and regions to the right search engines, news portals, and specialized sites. Without this, sub-agents all hit the same DuckDuckGo pipe — real multi-source research is impossible.

### Current State (4 backends, all English-centric)

| Provider | Type | Key Required | Coverage |
|---|---|---|---|
| DuckDuckGo | HTML scrape | No | English-biased general |
| Brave Search | REST API | `BRAVE_API_KEY` | English-biased general |
| You.com | REST API | `YDC_API_KEY` | English general |
| Perplexity Sonar | AI-synthesized | `PERPLEXITY_API_KEY` | English general |

### Target: Multi-Source Registry

The registry is a structured data file (`source_registry.toml` or similar) that the orchestrator queries to select which sources each sub-agent should use.

#### 1. General Search Engines (by region)

| Engine | API | Region/Language | Free Tier | Notes |
|---|---|---|---|---|
| Google Custom Search | REST | Global, any language | 100 queries/day | `cx` + `gl`/`lr` params for region/language filtering |
| Google News | RSS/scrape | Global, any language | Free | `hl=ar&gl=AE` for Arabic, `hl=fa&gl=IR` for Farsi, etc. |
| Bing Web Search | REST | Global, any language | 1K/month | `mkt` param for market, `setLang` for language |
| Bing News Search | REST | Global, any language | 1K/month | Separate endpoint, structured news results |
| Baidu | REST/scrape | Chinese | Free (scrape) | Essential for China-focused research |
| Yandex | REST | Russian, CIS | 1K/day | `lr` param for region, good for Russia/Ukraine topics |
| Naver | REST | Korean | Requires app | Essential for Korea-focused research |
| Sogou | Scrape | Chinese | Free | Alternative to Baidu, good for WeChat content |
| DuckDuckGo | HTML scrape | Global (English-biased) | Free | Current default, no API key needed |
| Brave Search | REST | Global (English-biased) | 2K/month | Current supported |
| Searx/SearxNG | Self-hosted | Meta-search (all engines) | Free | Aggregates Google/Bing/DDG/etc., self-hosted privacy |

#### 2. News Portals & Aggregators

| Source | Coverage | Language | Access Method |
|---|---|---|---|
| **Google News** | Global | 40+ languages | RSS feeds with `hl`/`gl` params, or scrape |
| **Bing News** | Global | 20+ languages | REST API with `mkt` param |
| **Reuters** | Global | English | `web_fetch` (no API) |
| **AP News** | Global | English | RSS / `web_fetch` |
| **Al Jazeera** | Middle East | Arabic, English | RSS / `web_fetch` |
| **Al Arabiya** | Middle East | Arabic | RSS / `web_fetch` |
| **Haaretz** | Israel | Hebrew, English | `web_fetch` (paywall) |
| **Times of Israel** | Israel | English | RSS / `web_fetch` |
| **IRNA** | Iran | Farsi, English | RSS / `web_fetch` |
| **Fars News** | Iran | Farsi | `web_fetch` |
| **TASS** | Russia | Russian, English | RSS / `web_fetch` |
| **Xinhua** | China | Chinese, English | RSS / `web_fetch` |
| **NHK** | Japan | Japanese, English | RSS / `web_fetch` |
| **Yonhap** | Korea | Korean, English | RSS / `web_fetch` |
| **Globo** | Brazil | Portuguese | RSS / `web_fetch` |
| **EFE** | Spain/LatAm | Spanish | RSS / `web_fetch` |
| **Deutsche Welle** | Germany | 30 languages | RSS / `web_fetch` |
| **France 24** | France | French, English, Arabic | RSS / `web_fetch` |

#### 3. Vertical / Domain-Specific Sources

**Sports:**
| Source | Focus | Language | Access |
|---|---|---|---|
| ESPN | US/Global sports | English | RSS / `web_fetch` |
| BBC Sport | UK/Global | English | RSS / `web_fetch` |
| Marca | Spanish football | Spanish | RSS / `web_fetch` |
| AS | Spanish football | Spanish | RSS / `web_fetch` |
| Mundo Deportivo | Spanish football | Spanish | RSS / `web_fetch` |
| A Bola | Portuguese football | Portuguese | RSS / `web_fetch` |
| Record (PT) | Portuguese football | Portuguese | RSS / `web_fetch` |
| L'Equipe | French sports | French | RSS / `web_fetch` |
| Gazzetta dello Sport | Italian football | Italian | RSS / `web_fetch` |
| Kicker | German football | German | RSS / `web_fetch` |
| Olé | Argentine football | Spanish | RSS / `web_fetch` |
| Globo Esporte | Brazilian football | Portuguese | RSS / `web_fetch` |
| The Athletic | Premium analysis | English | `web_fetch` (paywall) |
| FBref / StatsBomb | Statistics | English | `web_fetch` / API |
| Transfermarkt | Transfer data | Multi-language | `web_fetch` |
| WhoScored / SofaScore | Match stats | English | `web_fetch` |

**Finance:**
| Source | Focus | Access |
|---|---|---|
| Bloomberg | Global finance | `web_fetch` (limited) |
| Reuters Finance | Global markets | `web_fetch` |
| Yahoo Finance | Stocks/crypto | API (free tier) |
| Seeking Alpha | Analysis | `web_fetch` |
| CoinGecko / CoinMarketCap | Crypto | API (free) |
| 东方财富 (EastMoney) | China markets | `web_fetch` |

**Technology:**
| Source | Focus | Access |
|---|---|---|
| Hacker News | Tech/startups | API (free, no key) |
| Ars Technica | Tech analysis | RSS / `web_fetch` |
| TechCrunch | Startups/VC | RSS / `web_fetch` |
| The Verge | Consumer tech | RSS / `web_fetch` |
| 36Kr | China tech | `web_fetch` |
| ITHome | China tech | `web_fetch` |

**Academic / Research:**
| Source | Focus | Access |
|---|---|---|
| Google Scholar | Academic papers | Scrape (no official API) |
| Semantic Scholar | Academic papers | API (free, no key) |
| arXiv | Preprints | API (free) |
| PubMed | Biomedical | API (free) |
| SSRN | Social sciences | `web_fetch` |

**Government / Think Tanks:**
| Source | Focus | Access |
|---|---|---|
| UN.org | International | `web_fetch` |
| WHO | Health | `web_fetch` |
| World Bank | Economics | API (free) |
| RAND Corporation | Policy analysis | `web_fetch` |
| Brookings | Policy analysis | `web_fetch` |
| IISS | Defense/security | `web_fetch` |
| CFR | Foreign relations | `web_fetch` |
| CSIS | Strategic studies | `web_fetch` |

**Prediction / OSINT:**
| Source | Focus | Access |
|---|---|---|
| Polymarket | Prediction markets | API / `web_fetch` |
| Metaculus | Forecasting | API / `web_fetch` |
| Reddit | Community discussion | API (free, rate-limited) |
| X/Twitter | Real-time social | API (paid, expensive) |

### Registry Data Model

```toml
# source_registry.toml

# --- Search Engines ---

[[search_engines]]
id = "google"
name = "Google Custom Search"
type = "api"
endpoint = "https://www.googleapis.com/customsearch/v1"
key_env = "GOOGLE_CSE_API_KEY"
extra_env = { cx = "GOOGLE_CSE_ID" }
supports_language = true    # gl/lr params
supports_region = true      # gl param
rate_limit = "100/day"
priority = 1                # prefer when available

[[search_engines]]
id = "google_news"
name = "Google News"
type = "rss"
endpoint = "https://news.google.com/rss/search?q={query}&hl={lang}&gl={region}"
key_env = ""                # no key needed
supports_language = true
supports_region = true
rate_limit = "none"
priority = 1

[[search_engines]]
id = "bing"
name = "Bing Web Search"
type = "api"
endpoint = "https://api.bing.microsoft.com/v7.0/search"
key_env = "BING_API_KEY"
supports_language = true    # mkt param
supports_region = true
rate_limit = "1000/month"
priority = 2

[[search_engines]]
id = "bing_news"
name = "Bing News Search"
type = "api"
endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
key_env = "BING_API_KEY"
supports_language = true
supports_region = true
rate_limit = "1000/month"
priority = 2

[[search_engines]]
id = "baidu"
name = "Baidu Search"
type = "scrape"
endpoint = "https://www.baidu.com/s?wd={query}"
key_env = ""
supports_language = false   # Chinese only
supports_region = false
languages = ["zh"]
rate_limit = "none"
priority = 1                # essential for Chinese content

[[search_engines]]
id = "yandex"
name = "Yandex Search"
type = "api"
endpoint = "https://yandex.com/search/xml"
key_env = "YANDEX_API_KEY"
supports_language = true
supports_region = true
languages = ["ru", "uk", "kk", "be"]
rate_limit = "1000/day"
priority = 2

# --- Topic → Source Mapping ---

[[topic_sources]]
topic = "middle_east"
keywords = ["iran", "iraq", "syria", "lebanon", "israel", "palestine", "saudi", "yemen", "khamenei", "hezbollah", "hamas"]
search_engines = ["google", "google_news", "bing_news"]
languages = ["ar", "fa", "he", "en"]
portals = [
    { name = "Al Jazeera", url = "https://www.aljazeera.com", lang = "ar" },
    { name = "Al Jazeera EN", url = "https://www.aljazeera.com/news", lang = "en" },
    { name = "Al Arabiya", url = "https://www.alarabiya.net", lang = "ar" },
    { name = "Times of Israel", url = "https://www.timesofisrael.com", lang = "en" },
    { name = "IRNA", url = "https://www.irna.ir", lang = "fa" },
    { name = "Fars News", url = "https://www.farsnews.ir", lang = "fa" },
]
think_tanks = ["IISS", "RAND", "Brookings", "CFR", "CSIS"]

[[topic_sources]]
topic = "football"
keywords = ["world cup", "FIFA", "soccer", "football", "premier league", "la liga", "champions league", "euros"]
search_engines = ["google", "google_news"]
languages = ["en", "es", "pt", "fr", "de", "it"]
portals = [
    { name = "ESPN FC", url = "https://www.espn.com/soccer/", lang = "en" },
    { name = "BBC Sport", url = "https://www.bbc.com/sport/football", lang = "en" },
    { name = "Marca", url = "https://www.marca.com/futbol.html", lang = "es" },
    { name = "AS", url = "https://as.com/futbol/", lang = "es" },
    { name = "A Bola", url = "https://www.abola.pt", lang = "pt" },
    { name = "Globo Esporte", url = "https://ge.globo.com/futebol/", lang = "pt" },
    { name = "L'Equipe", url = "https://www.lequipe.fr/Football/", lang = "fr" },
    { name = "Gazzetta", url = "https://www.gazzetta.it/Calcio/", lang = "it" },
    { name = "Kicker", url = "https://www.kicker.de", lang = "de" },
    { name = "FBref", url = "https://fbref.com", lang = "en" },
    { name = "Transfermarkt", url = "https://www.transfermarkt.com", lang = "en" },
]
data_sources = ["FBref", "WhoScored", "SofaScore", "Transfermarkt"]

[[topic_sources]]
topic = "china"
keywords = ["china", "beijing", "xi jinping", "CPC", "chinese", "中国", "北京"]
search_engines = ["baidu", "google", "bing"]
languages = ["zh", "en"]
portals = [
    { name = "Xinhua", url = "https://www.xinhuanet.com", lang = "zh" },
    { name = "SCMP", url = "https://www.scmp.com", lang = "en" },
    { name = "36Kr", url = "https://36kr.com", lang = "zh" },
    { name = "ITHome", url = "https://www.ithome.com", lang = "zh" },
]

[[topic_sources]]
topic = "russia_ukraine"
keywords = ["russia", "ukraine", "putin", "zelensky", "moscow", "kyiv", "NATO"]
search_engines = ["yandex", "google", "google_news"]
languages = ["ru", "uk", "en"]
portals = [
    { name = "TASS", url = "https://tass.com", lang = "en" },
    { name = "TASS RU", url = "https://tass.ru", lang = "ru" },
    { name = "Ukrainska Pravda", url = "https://www.pravda.com.ua", lang = "uk" },
    { name = "Kyiv Independent", url = "https://kyivindependent.com", lang = "en" },
]

[[topic_sources]]
topic = "finance"
keywords = ["stock", "market", "GDP", "inflation", "fed", "interest rate", "crypto", "bitcoin", "IPO"]
search_engines = ["google", "bing_news"]
languages = ["en", "zh"]
portals = [
    { name = "Bloomberg", url = "https://www.bloomberg.com", lang = "en" },
    { name = "Reuters Finance", url = "https://www.reuters.com/finance/", lang = "en" },
    { name = "Yahoo Finance", url = "https://finance.yahoo.com", lang = "en" },
    { name = "EastMoney", url = "https://www.eastmoney.com", lang = "zh" },
]
data_sources = ["Yahoo Finance API", "CoinGecko API", "World Bank API"]

[[topic_sources]]
topic = "technology"
keywords = ["AI", "LLM", "GPU", "startup", "tech", "software", "SaaS", "cloud"]
search_engines = ["google", "bing"]
languages = ["en", "zh"]
portals = [
    { name = "Hacker News", url = "https://news.ycombinator.com", lang = "en" },
    { name = "Ars Technica", url = "https://arstechnica.com", lang = "en" },
    { name = "TechCrunch", url = "https://techcrunch.com", lang = "en" },
    { name = "36Kr", url = "https://36kr.com", lang = "zh" },
]
data_sources = ["HN API", "arXiv API", "Semantic Scholar API"]

[[topic_sources]]
topic = "academic"
keywords = ["research", "study", "paper", "journal", "peer-reviewed", "clinical trial"]
search_engines = ["google"]
languages = ["en"]
portals = []
data_sources = ["Google Scholar (scrape)", "Semantic Scholar API", "arXiv API", "PubMed API", "SSRN"]
```

### How the Orchestrator Uses the Registry

```
User: "Who will win the 2026 World Cup?"

1. Orchestrator analyzes query → detects topic: "football" + "prediction"

2. Looks up source_registry:
   - topic "football" → engines: google, google_news
                       → languages: en, es, pt, fr, de, it
                       → portals: ESPN, Marca, A Bola, FBref, etc.
   - topic "prediction" → add: Polymarket, Metaculus

3. Generates sub-agent plan:
   ┌─ Agent 1: English analysis (Google News en, ESPN, BBC Sport, The Athletic)
   ├─ Agent 2: Spanish press (Google News es, Marca, AS, Mundo Deportivo)
   ├─ Agent 3: Portuguese press (Google News pt, A Bola, Globo Esporte)
   ├─ Agent 4: Statistics (FBref, WhoScored, Transfermarkt — data-heavy)
   ├─ Agent 5: Prediction markets (Polymarket, Metaculus, betting odds)
   ├─ Agent 6: FIFA/official + historical analysis (Google Scholar for patterns)
   └─ Agent 7: French/German/Italian press (L'Equipe, Kicker, Gazzetta)

4. Each agent gets:
   - Specific search engines to use (with language params)
   - Specific portals to browse directly via web_fetch
   - Specific data sources to query via API
```

### Implementation Priority

**Phase 0 (Now)**: Build the registry data file + lookup logic
- `source_registry.toml` with topic→source mappings
- Rust struct to parse and query it
- Orchestrator reads registry to generate sub-agent assignments

**Phase 1**: Add search engine backends
- Google Custom Search API (highest priority — global, multi-language)
- Google News RSS (free, no key, multi-language)
- Bing Web + News API (complements Google)
- Baidu scrape (essential for Chinese content)

**Phase 2**: Add vertical APIs
- Semantic Scholar API (academic)
- HN API (tech)
- Yahoo Finance API (finance)
- arXiv API (preprints)

**Phase 3**: Direct portal browsing
- Sub-agents use `web_fetch` to browse portals directly
- RSS feed parsing for news portals
- Handle paywalls gracefully (use free-tier content, note limitation)

### API Keys Required

| Engine | Key | Free Tier | Priority |
|---|---|---|---|
| Google Custom Search | `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_ID` | 100 queries/day | HIGH |
| Bing Search | `BING_API_KEY` | 1K queries/month | HIGH |
| Baidu | None (scrape) | Unlimited | HIGH |
| Yandex | `YANDEX_API_KEY` | 1K/day | MEDIUM |
| Semantic Scholar | None | Unlimited (rate-limited) | MEDIUM |
| Hacker News | None | Unlimited | LOW |
| arXiv | None | Unlimited | LOW |

---

## Success Metrics

- **Report length**: from ~6K chars to 10K+ words
- **Citation count**: from ~0 (inline only) to 20+ traceable citations
- **Source diversity**: from single-language to multi-language/multi-region
- **User satisfaction**: reports should contain specific data points, numbers, dates, quotes — not generic summaries
- **Time budget**: 5-15 minutes (acceptable for comprehensive research)

## References

- [Gemini Deep Research API](https://ai.google.dev/gemini-api/docs/deep-research)
- [Google LangGraph Reference Implementation](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)
- [Kimi-Researcher Technical Report](https://moonshotai.github.io/Kimi-Researcher/)
- [Kimi K2.5 Technical Report](https://arxiv.org/html/2602.02276v1)
- [DeepSearchQA Benchmark](https://www.arxiv.org/pdf/2601.20975)
- [ByteByteGo: How OpenAI, Gemini, and Claude Use Agents for Deep Research](https://blog.bytebytego.com/p/how-openai-gemini-and-claude-use)
