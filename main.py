# ===================== Import Libraries =====================
from google import genai
import time
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated, Literal, List, Optional
from pydantic import BaseModel, Field
import operator
from langchain_openai import ChatOpenAI
from enum import Enum
from langgraph.types import Send
from pathlib import Path

load_dotenv()
# ===================== Define LLM =====================
llm = ChatOpenAI(model='gpt-4o-mini')

# ===================== Pydantic Scemas =====================
class TASK(BaseModel):
    id: int = Field(..., description="put unique ID")
    task_title: str = Field(..., description="task title should be well structured, clean, and to the point")
    target_words: int = Field(..., description="Target word count for this section (120–550).")
    bullets: str = Field(..., min_length=2, max_length=3, description="must be 2-3 bullets")
    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )

class PLAN(BaseModel):
    blog_title: str = Field(..., description="Blog title should be Attractive, Clear and Clean")
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    audience: str = Field(..., description="The intended audience for this blog.")

    class ToneEnum(str, Enum):
        PROFESSIONAL = "professional"
        CONVERSATIONAL = "conversational"
        EDUCATIONAL = "educational"
        TECHNICAL = "technical"
        HUMOROUS = "humorous"

    tone: ToneEnum = Field(..., description="Writing style that matches the target audience and content type")
    tasks: List[TASK]

# ===================== Define State =====================
class state_class(TypedDict):
    topic: str
    plan: PLAN
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    final_result: str

# ===================== Orchestrator Node =====================
def orchestrator_node(state: state_class):
    topic = state['topic']
    planner = llm.with_structured_output(PLAN)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5–7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3–5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120–450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                    "trade-offs → testing/observability → conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a practical summary/checklist and next steps.\n\n"
                    "Output must strictly match the Plan schema."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return {"plan": plan}

# ===================== Get Workers =====================
def fanout(state: state_class):
    return [
        Send(
            'worker',
            {
                'task': task,
                'topic': state['topic'],
                'plan': state['plan']
            }
        ) for task in state['plan'].tasks
    ]

# ===================== Worker Node =====================
def worker_node(payload: dict):
    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    section = llm.invoke(
        SystemMessage(
            content=(
                "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n\n"
                "Hard constraints:\n"
                "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
                "- Stay close to the Target words (±15%).\n"
                "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
                "Technical quality bar:\n"
                "- Be precise and implementation-oriented (developers should be able to apply it).\n"
                "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
                "- When relevant, include at least one of:\n"
                "  * a small code snippet (minimal, correct, and idiomatic)\n"
                "  * a tiny example input/output\n"
                "  * a checklist of steps\n"
                "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
                "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
                "- Call out edge cases / failure modes and what to do about them.\n"
                "- If you mention a best practice, add the 'why' in one sentence.\n\n"
                "Markdown style:\n"
                "- Start with a '## <Section Title>' heading.\n"
                "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
                "- Avoid fluff. Avoid marketing language.\n"
                "- If you include code, keep it focused on the bullet being addressed.\n"
                )
                ),
        HumanMessage(
            content=(
                f"blog title is: {plan.blog_title}"
                f"blog goal is: {plan.goal}"
                f"blog audience is: {plan.audience}"
                f"blog tone is: {plan.tone}"
                f"topic is: {topic}\n"
                f"topic is: {task.task_title}\n"
                f"topic is: {task.section_type}\n"
                f"topic is: {task.targer_words}\n"
                f"Bullets : {bullets_text}\n"
                )
                )
                ).content.strip()
    return {'sections': [section]}

# ===================== Reducer Node =====================
def reducer_node(state: state_class):
    title = state['plan'].blog_title
    body = '\n\n'.join(state['sections']).strip()

    final_md = f"# {title}\n\n {body}\n"

    filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title)
    filename = filename.strip().lower().replace(" ", "_") + ".md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}
