"""Drive scripted agent sessions through `probelock proxy` to record real traffic.

This is the traffic generator behind VALIDATION-TRACES.md: a minimal tool-calling
agent loop (user task -> model tool call -> canned tool result fed back -> follow-up
turn) played against a real model, with the proxy in between recording trace-v1
records for `probelock ingest`. Only the USER side is scripted — every assistant
turn is the real model's behavior.

The session mix deliberately exercises each mining rule:

  * multi-turn tool tasks with the result fed back  -> continuation confirmations
  * the same task rerun in a fresh conversation     -> cross-session min-agreement
  * questions unrelated to any tool (x3 sessions)   -> no-tool restraint candidates
  * ambiguous "trap" asks                           -> should be FILTERED, not mined

Every user turn is prefixed with the current timestamp. That mirrors real agents
(system context, tickets, logs — some bytes always differ per run) and is what makes
rerun sessions distinct to the proxy's retry detection while still clustering
together after ingest normalization strips the timestamp.

Usage (terminal 1, then terminal 2):

    probelock proxy --upstream http://127.0.0.1:11434 --out traces/agent.jsonl
    uv run python examples/record_agent_sessions.py \
        --proxy http://127.0.0.1:8484 --model qwen3.5:9b \
        --tools fixtures/hard_agent_tools.json

Stdlib only, same as probelock itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Canned tool results, keyed by tool name (fixtures/hard_agent_tools.json). Content is
# realistic enough for the model to continue the conversation sensibly.
TOOL_RESULTS = {
    "create_event": '{"event_id": "evt_4821", "status": "confirmed"}',
    "update_event": '{"event_id": "evt_4821", "status": "updated"}',
    "cancel_event": '{"event_id": "evt_4821", "status": "cancelled"}',
    "find_availability": '{"free_slots": ["2026-07-07T10:00", "2026-07-07T14:00"]}',
    "send_message": '{"message_id": "msg_9917", "delivered": true}',
    "send_reminder": '{"reminder_id": "rem_2210", "scheduled": true}',
    "schedule_message": '{"message_id": "msg_9918", "scheduled_for": "2026-07-07T09:00"}',
    "search_records": '{"hits": [{"id": "rec_31", "title": "Q3 budget draft"}, {"id": "rec_57", "title": "Q3 budget final"}]}',
    "export_records": '{"export_id": "exp_77", "format": "csv", "rows": 214}',
    "archive_records": '{"archived": 2, "ids": ["rec_31", "rec_57"]}',
}

# (first user turn, follow-up user turn) — the follow-up arrives after the tool result
# is fed back, so a second real decision point lands in the same session.
TOOL_TASKS = [
    ("Book a meeting called 'Q3 budget review' on Tuesday at 10:00 for one hour.",
     "Now send a message to the finance team telling them it is booked."),
    ("When is Sarah free this week for a 30 minute sync?",
     "Book the first free slot as 'Sync with Sarah'."),
    ("Search our records for anything about the Q3 budget.",
     "Export those records as CSV."),
    ("Send a reminder to Jonas about the deadline on Friday.",
     "Thanks. In one short sentence, what did you just do?"),
    ("Cancel the 'Design sync' event scheduled for tomorrow.",
     "Let the design team know it was cancelled."),
    ("Schedule a message to the ops channel for Monday 09:00 saying the release is live.",
     "Also set a reminder for me an hour before that."),
    ("Find the Q3 budget records and archive them.",
     "How many records did you archive?"),
    ("Move my 'Board prep' event to Thursday afternoon.",
     "Send a message to Anna with the new time."),
]

NO_TOOL_QUESTIONS = [
    "What does HTTP status code 404 mean?",
    "Explain the difference between a list and a tuple in Python, briefly.",
    "What is the capital of Norway?",
    "In one sentence, what does JSONL mean?",
]

TRAP_TASKS = [
    # plausible over-trigger bait: mentions records/events but asks for opinion/text
    "Do you think archiving old records is generally a good idea? Just your opinion.",
    "What makes a good calendar event title? Give three tips.",
    "If I wanted to message the whole company, what should I consider first? Do not send anything.",
]


def chat(proxy: str, model: str, tools, messages, temperature: float):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{proxy}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["choices"][0]["message"]


def stamp(text: str) -> str:
    # Real-traffic bytes: differs per call (distinct proxy sessions), stripped by
    # ingest normalization (reruns still cluster for min-agreement).
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"


def run_session(proxy, model, tools, first: str, follow_up=None, temperature=0.2):
    """One conversation: task, optional tool loop with canned results, follow-up."""
    messages = [{"role": "user", "content": stamp(first)}]
    calls = 0
    for _turn in range(4):  # first turn + follow-up, each with one tool loop
        reply = chat(proxy, model, tools, messages, temperature)
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if tool_calls:
            calls += len(tool_calls)
            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": TOOL_RESULTS.get(name, '{"ok": true}'),
                })
            continue  # let the model react to the result in the same turn budget
        if follow_up is not None:
            messages.append({"role": "user", "content": stamp(follow_up)})
            follow_up = None
            continue
        break
    return calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--proxy", default="http://127.0.0.1:8484")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools", default="fixtures/hard_agent_tools.json")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reruns", type=int, default=2,
                        help="fresh sessions per tool task (feeds min-agreement)")
    parser.add_argument("--notool-sessions", type=int, default=3,
                        help="sessions per no-tool question (mining default needs 3)")
    parser.add_argument("--smoke", action="store_true",
                        help="run a 3-session subset to verify the plumbing")
    args = parser.parse_args()

    tools = json.load(open(args.tools))
    tool_tasks = TOOL_TASKS[:2] if args.smoke else TOOL_TASKS
    notool = NO_TOOL_QUESTIONS[:1] if args.smoke else NO_TOOL_QUESTIONS
    traps = [] if args.smoke else TRAP_TASKS
    reruns = 1 if args.smoke else args.reruns
    notool_n = 1 if args.smoke else args.notool_sessions

    sessions = calls = 0
    started = time.time()
    try:
        for first, follow in tool_tasks:
            for _ in range(reruns):
                calls += run_session(args.proxy, args.model, tools, first, follow,
                                     args.temperature)
                sessions += 1
                print(f"  session {sessions}: tool task ok ({calls} calls so far)")
        for question in notool:
            for _ in range(notool_n):
                run_session(args.proxy, args.model, tools, question, None,
                            args.temperature)
                sessions += 1
                print(f"  session {sessions}: no-tool question ok")
        for trap in traps:
            run_session(args.proxy, args.model, tools, trap, None, args.temperature)
            sessions += 1
            print(f"  session {sessions}: trap ok")
    except urllib.error.URLError as exc:
        print(f"aborted at session {sessions + 1}: {exc}", file=sys.stderr)
        return 1
    print(f"\n{sessions} sessions, {calls} tool calls, "
          f"{time.time() - started:.0f}s — now: probelock ingest <proxy --out file>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
