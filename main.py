# ===================== Import Libraries =====================
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated, Literal, List, Optional
from pydantic import BaseModel, Field
import operator
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from enum import Enum
from langgraph.types import Send
from pathlib import Path
from datetime import date, datetime, timedelta
from CONFIG import GROQ_MODEL
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
# ===================== Define LLM =====================
gpt_llm = ChatOpenAI(model='gpt-4o-mini')

# ===================== Pydantic Scemas =====================
class TASK(BaseModel):
    id: int = Field(..., description="put unique ID")
    task_title: str = Field(..., description="task title should be well structured, clean, and to the point")
    target_words: int = Field(..., description="Target word count for this section (120–550).")
    bullets: str = Field(..., min_length=2, max_length=3, description="must be 2-3 bullets")
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class PLAN(BaseModel):
    blog_title: str = Field(..., description="Blog title should be Attractive, Clear and Clean")
    audience: str = Field(..., description="The intended audience for this blog.")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    class ToneEnum(str, Enum):
        PROFESSIONAL = "professional"
        CONVERSATIONAL = "conversational"
        EDUCATIONAL = "educational"
        TECHNICAL = "technical"
        HUMOROUS = "humorous"
    tone: ToneEnum = Field(..., description="Writing style that matches the target audience and content type")
    tasks: List[TASK]
    constraints: List[str]

class Router(BaseModel):
    need_research: bool = Field(..., description="return True if need research else False")
    mode: Literal['closed_book', 'hybrid', 'open_book']
    queries: List[str] = Field(..., default_factory=list)

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(..., default_factory=list)

# ===================== Define State =====================
class state_class(TypedDict):
    topic: str
    plan: PLAN
    need_research: bool
    mode: str
    queries: List[str]
    evidence: List[EvidenceItem]
    sections: Annotated[List[tuple[int, str]], operator.add]
    final_result: str
    recency_days: int
    as_of: str  #ISO date 

# ===================== Router Node =====================
def router_node(state: state_class):
    ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
                        Decide whether web research is needed BEFORE planning.

                        Modes:
                        - closed_book (needs_research=false):
                        Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
                        - hybrid (needs_research=true):
                        Mostly evergreen but needs up-to-date examples/tools/models to be useful.
                        - open_book (needs_research=true):
                        Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

                        If needs_research=true:
                        - Output 3–10 high-signal queries.
                        - Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
                        - If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
                        """

    topic = state['topic']

    decider = gpt_llm.with_structured_output(Router)
    response = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=topic)
        ]
    )

    if response.mode == 'open_book':
        recency_days = 7
    elif response.mode == 'hybrid':
        recency_days = 45
    elif response.mode == 'closed_book':
        recency_days = 3650

    return {
        'need_research': response.need_research,
        'mode': response.mode,
        'queries': response.queries,
        'recency_days': recency_days,
        'as_of': date.today().isoformat()
    }


def route_next(state: state_class):
    return "research" if state["need_research"] else "orchestrator"

# ===================== Research Node =====================
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Uses TavilySearchResults if installed and TAVILY_API_KEY is set.
    Returns list of dict with common fields. Note: published date is often missing.
    """
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

def research_node(state: state_class) -> dict:
    RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
                        Given raw web search results, produce a deduplicated list of EvidenceItem objects.
                        Rules:
                        - Only include items with a non-empty url.
                        - Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
                        - Extract/normalize published_at as ISO (YYYY-MM-DD) if you can infer it from title/snippet.
                        If you can't infer a date reliably, set published_at=null (do NOT guess).
                        - Keep snippets short.
                        - Deduplicate by URL.
                        """
    queries = (state.get("queries", []) or [])[:10]
    max_results = 6

    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = gpt_llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw_results}"
                )
            ),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # HARD RECENCY FILTER for open_book weekly roundup:
    # keep only items with a parseable ISO date and within the window.
    mode = state.get("mode", "closed_book")
    if mode == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        fresh: List[EvidenceItem] = []
        for e in evidence:
            d = _iso_to_date(e.published_at)
            if d and d >= cutoff:
                fresh.append(e)
        evidence = fresh

    return {"evidence": evidence}

# ===================== Orchestrator Node =====================
def orchestrator_node(state: state_class) -> dict:
    ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
                    Your job is to produce a highly actionable outline for a technical blog post.

                    Hard requirements:
                    - Create 5–9 sections (tasks) suitable for the topic and audience.
                    - Each task must include:
                    1) goal (1 sentence)
                    2) 3–6 bullets that are concrete, specific, and non-overlapping
                    3) target word count (120–550)

                    Flexibility:
                    - Do NOT use a fixed taxonomy unless it naturally fits.
                    - You may tag tasks (tags field), but tags are flexible.

                    Quality bar:
                    - Assume the reader is a developer; use correct terminology.
                    - Bullets must be actionable: build/compare/measure/verify/debug.
                    - Ensure the overall plan includes at least 2 of these somewhere:
                    * minimal code sketch / MWE (set requires_code=True for that section)
                    * edge cases / failure modes
                    * performance/cost considerations
                    * security/privacy considerations (if relevant)
                    * debugging/observability tips

                    Grounding rules:
                    - Mode closed_book: keep it evergreen; do not depend on evidence.
                    - Mode hybrid:
                    - Use evidence for up-to-date examples (models/tools/releases) in bullets.
                    - Mark sections using fresh info as requires_research=True and requires_citations=True.
                    - Mode open_book (weekly news roundup):
                    - Set blog_kind = "news_roundup".
                    - Every section is about summarizing events + implications.
                    - DO NOT include tutorial/how-to sections (no scraping/RSS/how to fetch news) unless user explicitly asked for that.
                    - If evidence is empty or insufficient, create a plan that transparently says "insufficient fresh sources"
                        and includes only what can be supported.

                    Output must strictly match the Plan schema.
                    """
    planner = gpt_llm.with_structured_output(PLAN)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    # Force blog_kind for open_book
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}\n\n"
                    f"Instruction: If mode=open_book, your plan must NOT drift into a tutorial."
                )
            ),
        ]
    )

    # Ensure open_book forces the kind even if model forgets
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}

# ===================== Get Workers =====================
def fanout(state: state_class):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# ===================== Worker Node =====================
def worker(payload: dict) -> dict:
    WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
                        Write ONE section of a technical blog post in Markdown.

                        Hard constraints:
                        - Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
                        - Stay close to Target words (±15%).
                        - Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
                        - Start with a '## <Section Title>' heading.

                        Scope guard (prevents mid-blog topic drift):
                        - If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
                        Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
                        Focus on summarizing events and implications.

                        Grounding policy:
                        - If mode == open_book (weekly news):
                        - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
                        - For each event claim, attach a source as a Markdown link: ([Source](URL)).
                        - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
                        - If requires_citations == true (hybrid sections):
                        - For outside-world claims, cite Evidence URLs the same way.
                        - Evergreen reasoning (concepts, intuition) is OK without citations unless requires_citations is true.

                        Code:
                        - If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

                        Style:
                        - Short paragraphs, bullets where helpful, code fences for code.
                        - Avoid fluff/marketing. Be precise and implementation-oriented.
                        """
    
    task = TASK(**payload["task"])
    plan = PLAN(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")
    as_of = payload.get("as_of")
    recency_days = payload.get("recency_days")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # Provide a compact evidence list for citation use
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = gpt_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {as_of} (recency_days={recency_days})\n\n"
                    f"Section title: {task.task_title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    # deterministic ordering
    return {"sections": [(task.id, section_md)]}

# ===================== Reducer Node =====================
def reducer_node(state: state_class):
    title = state['plan'].blog_title
    sorted_sections = sorted(state['sections'], key=lambda x: x[0])
    section_contents = [content for _, content in sorted_sections] # Extract just the content (second element of tuple)
    body = '\n\n'.join(section_contents).strip()
    final_md = f"# {title}\n\n{body}\n"
    
    # Create filename
    filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title)
    filename = filename.strip().lower().replace(" ", "_") + ".md"
    
    # Write to file silently
    Path(filename).write_text(final_md, encoding="utf-8")
    
    print(f"✅ Blog created: {filename}")
    
    return {"final_result": final_md}

# ===================== Define All Nodes =====================
g = StateGraph(state_class)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

app
