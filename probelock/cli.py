"""probelock command line interface."""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .clients import AnyLlmClient, ClientError, HttpClient, LiteLlmClient, SimulatedClient
from .diff import diff_lockfiles, error_derived_negative_capabilities
from .lockfile import read_lockfile, write_lockfile
from .probes import derive_probes, tools_fingerprint
from .runner import run_probes

app = typer.Typer(
    add_completion=False,
    help="probelock — a capability lockfile for local models. Catch silent "
    "regressions when you swap a model, quant, or runtime.",
)
console = Console()
err_console = Console(stderr=True)


def _err(msg: str) -> None:
    """Print a red error line, escaping dynamic text so brackets in paths or
    messages (e.g. 'probelock[anyllm]') aren't eaten by rich's markup parser."""
    err_console.print(f"[red]{escape(msg)}[/]")


_BAR = 24


def _bar(score: float) -> str:
    filled = round(score * _BAR)
    color = "green" if score >= 0.9 else "yellow" if score >= 0.7 else "red"
    return f"[{color}]{'█' * filled}{'░' * (_BAR - filled)}[/] {score:.2f}"


def _load_tools(path: Path):
    tools = json.loads(Path(path).read_text())
    if not isinstance(tools, list):
        raise ValueError("tools file must be a JSON array of OpenAI-style tools")
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict) or "name" not in (tool.get("function") or {}):
            raise ValueError(
                f"tool #{i} must be an object with a 'function.name' (OpenAI tools format)"
            )
    return tools


def _load_tools_or_exit(path: Path):
    """Load tools, converting any bad-input error to a clean Exit(2) so a typo'd
    --tools path never exits 1 (the regression code) with a raw traceback."""
    try:
        return _load_tools(path)
    except FileNotFoundError:
        _err(f"Tools file not found: {path}")
        raise typer.Exit(2)
    except OSError as exc:
        _err(f"Could not read tools file {path}: {exc}")
        raise typer.Exit(2)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        _err(f"Invalid tools file {path}: {exc}")
        raise typer.Exit(2)


def _load_json_or_exit(path: Path, what: str):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        _err(f"{what} not found: {path}")
        raise typer.Exit(2)
    except OSError as exc:
        _err(f"Could not read {what} {path}: {exc}")
        raise typer.Exit(2)
    except (json.JSONDecodeError, ValueError) as exc:
        _err(f"Invalid {what} {path}: {exc}")
        raise typer.Exit(2)


@app.command()
def derive(
    tools: Path = typer.Option(..., "--tools", "-t", help="OpenAI-style tools JSON file."),
) -> None:
    """Show the probe battery that would be generated from a toolset (transparency)."""
    probes = derive_probes(_load_tools_or_exit(tools))
    table = Table(title=f"{len(probes)} probes derived", expand=True)
    table.add_column("Probe id", no_wrap=True)
    table.add_column("Capability", no_wrap=True)
    table.add_column("Checks")
    for p in probes:
        table.add_row(p.id, p.capability, p.description)
    console.print(table)


@app.command()
def probe(
    tools: Path = typer.Option(..., "--tools", "-t", help="OpenAI-style tools JSON file."),
    simulate: Optional[Path] = typer.Option(
        None, "--simulate", "-s", help="Run against a deterministic profile (no model)."
    ),
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint", help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1)."
    ),
    via: str = typer.Option(
        "", "--via", help="Route through a library instead of --endpoint: anyllm | litellm "
        "(model is 'provider/name', e.g. anthropic/claude-3-5-sonnet)."
    ),
    model: str = typer.Option("", "--model", "-m", help="Model id (or 'provider/name' with --via)."),
    quant: str = typer.Option("", "--quant", help="Quantization tag, recorded in the lockfile."),
    runtime: str = typer.Option("", "--runtime", help="Runtime tag (ollama, llama.cpp, mlx...)."),
    api_key: str = typer.Option("", "--api-key", help="Bearer token, if the endpoint needs one."),
    timeout: float = typer.Option(60.0, "--timeout", help="Per-probe timeout in seconds."),
    samples: int = typer.Option(
        1, "--samples", help="Run each probe N times; the score becomes a pass-rate."
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", help="Sampling temperature (raise it with --samples for variance)."
    ),
    label: Optional[str] = typer.Option(None, "--label", help="Override the lockfile label."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the lockfile here."),
) -> None:
    """Run the probe battery and produce a capability lockfile."""
    tool_list = _load_tools_or_exit(tools)
    probes = derive_probes(tool_list)
    fingerprint = tools_fingerprint(tool_list)

    try:
        if simulate is not None:
            client = SimulatedClient(_load_json_or_exit(simulate, "simulate profile"))
        elif via:
            kind = via.lower()
            if kind in ("anyllm", "any-llm"):
                client = AnyLlmClient(model=model, api_key=api_key, temperature=temperature, quant=quant)
            elif kind in ("litellm", "lite-llm"):
                client = LiteLlmClient(model=model, api_key=api_key, temperature=temperature, quant=quant)
            else:
                _err(f"Unknown --via '{via}' (use anyllm | litellm).")
                raise typer.Exit(2)
        elif endpoint is not None:
            client = HttpClient(
                base_url=endpoint, model=model, api_key=api_key, quant=quant,
                runtime=runtime, timeout=timeout, temperature=temperature,
            )
        else:
            err_console.print("[red]Provide --simulate PROFILE, --endpoint URL, or --via {anyllm,litellm}.[/]")
            raise typer.Exit(2)
    except ClientError as exc:  # e.g. the --via SDK isn't installed
        _err(str(exc))
        raise typer.Exit(2)

    try:
        lock = run_probes(client, probes, fingerprint, __version__, samples=samples)
    except ClientError as exc:
        _err(str(exc))
        raise typer.Exit(2)

    errored = [r for r in lock.results if r.error]
    if errored and len(errored) == lock.n_probes:
        # Every probe failed at the API level -> this is a misconfiguration, not a
        # capability profile. Refuse to write a lockfile that could become a
        # poisoned all-zeros baseline.
        _err(
            f"All {lock.n_probes} probes failed at the API level — refusing to write "
            f"a lockfile. First error: {errored[0].error}"
        )
        raise typer.Exit(2)

    if samples > lock.samples:
        err_console.print(
            f"[yellow]--samples {samples} had no effect: this endpoint is deterministic "
            f"(temperature 0 / simulated), so samples are identical. Recorded samples=1; "
            f"raise --temperature for independent samples.[/]"
        )

    lock.generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    if label:
        lock.label = label

    console.print(
        f"\n[bold]{escape(lock.label)}[/]  ([cyan]{lock.n_probes} probes[/], "
        f"fp {lock.tools_fingerprint})"
    )
    table = Table(expand=True)
    table.add_column("Capability", no_wrap=True)
    table.add_column("Score", ratio=1)
    for cap, sc in lock.capabilities.items():
        table.add_row(cap, _bar(sc))
    console.print(table)

    if errored:
        console.print(
            f"[yellow]{len(errored)} probe(s) errored at the API level.[/] "
            f"e.g. {escape(str(errored[0].error))}"
        )
        for cap, (n_err, total) in error_derived_negative_capabilities(lock).items():
            console.print(
                f"[yellow]{cap}'s {lock.capabilities.get(cap):.2f} includes {n_err}/{total} "
                f"error-derived probe(s)[/] — not genuine model behavior; treat this "
                f"capability as low-confidence if committed as a baseline."
            )

    if out is not None:
        write_lockfile(lock, out)
        console.print(f"[green]wrote[/] {out}")


def _validate_confidence(confidence: Optional[float]) -> None:
    if confidence is not None and not (0.0 < confidence < 1.0):
        err_console.print(
            f"[red]--confidence must be between 0 and 1 (exclusive); got {confidence}.[/]"
        )
        raise typer.Exit(2)


def _read_lock(path: Path):
    """Read a lockfile, converting any malformed-input error into a clean Exit(2)
    so CI can distinguish a broken lockfile from a real regression (exit 1)."""
    try:
        return read_lockfile(path)
    except FileNotFoundError:
        _err(f"Lockfile not found: {path}")
        raise typer.Exit(2)
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        _err(f"Could not read lockfile {path}: {exc}")
        raise typer.Exit(2)


# One source of truth for status display (table rich-markup, markdown text).
_STATUS_LABELS = {
    "ok": ("[green]ok[/]", "✅ ok"),
    "regression": ("[bold red]REGRESSION[/]", "⚠️ REGRESSION"),
    "noisy": ("[yellow]noisy ↓[/]", "〰️ noisy"),
    "improved": ("[cyan]improved[/]", "⬆️ improved"),
    "added": ("[dim]added[/]", "➕ added"),
    "removed": ("[bold red]REMOVED[/]", "⛔ REMOVED"),
}


def _cell(value, signed=False) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _diff_notes(result, b, c):
    """Plain-text comparison caveats, shared by the table and markdown renderers
    so the two can never drift out of sync."""
    notes = []
    if result.tools_changed:
        notes.append("⚠ toolsets differ — comparison may not be apples-to-apples.")
    if b.model and c.model and b.model != c.model:
        notes.append(
            f"⚠ different models ({b.model} → {c.model}) — cross-model comparison, "
            f"not a within-model regression check."
        )
    elif b.model and (b.quant, b.runtime) != (c.quant, c.runtime):
        notes.append(
            f"within-model swap: {b.quant or 'native'}/{b.runtime or '?'} → "
            f"{c.quant or 'native'}/{c.runtime or '?'}"
        )
    if b.samples != c.samples:
        notes.append(
            f"⚠ sample counts differ ({b.samples} vs {c.samples}); --confidence has "
            f"uneven statistical power across the two lockfiles."
        )
    for which, lock in (("baseline", b), ("candidate", c)):
        for cap, (errored, total) in error_derived_negative_capabilities(lock).items():
            notes.append(
                f"⚠ {which}'s {cap} score includes {errored}/{total} probe(s) that errored "
                f"at the API level (scored via the safety fallback, not genuine model "
                f"behavior) — treat this capability as low-confidence."
            )
    return notes


def _render_diff(result, b, c) -> None:
    # escape() externally-sourced text (labels, capability names) so brackets in a
    # model/quant tag aren't eaten by rich markup; the status badge is real markup.
    title = f"{escape(b.label or 'baseline')}  →  {escape(c.label or 'candidate')}"
    table = Table(title=title, expand=True)
    for col, justify in (("Capability", "left"), ("Baseline", "right"),
                         ("Candidate", "right"), ("Δ", "right"), ("Status", "left")):
        table.add_column(col, justify=justify, no_wrap=(col in ("Capability", "Status")))
    for r in result.rows:
        label = _STATUS_LABELS.get(r.status, (r.status, r.status))[0]
        table.add_row(escape(r.capability), _cell(r.baseline), _cell(r.candidate),
                      _cell(r.delta, signed=True), label)
    console.print(table)
    for note in _diff_notes(result, b, c):
        console.print(f"[yellow]{escape(note)}[/]")


def _markdown_diff(result, b, c) -> str:
    lines = [
        f"### probelock: `{b.label or 'baseline'}` → `{c.label or 'candidate'}`",
        "",
        "| Capability | Baseline | Candidate | Δ | Status |",
        "|---|--:|--:|--:|---|",
    ]
    for r in result.rows:
        label = _STATUS_LABELS.get(r.status, (r.status, r.status))[1]
        lines.append(
            f"| `{r.capability}` | {_cell(r.baseline)} | {_cell(r.candidate)} | "
            f"{_cell(r.delta, signed=True)} | {label} |"
        )
    lines.append("")
    if result.regressed:
        names = ", ".join(f"`{r.capability}`" for r in result.regressions)
        lines.append(f"**FAIL** — capabilities regressed or removed: {names}")
    else:
        lines.append("**PASS** — no capability regressed.")
    for note in _diff_notes(result, b, c):
        lines.append(f"\n> {note}")
    return "\n".join(lines)


def _diff_payload(result, b, c) -> dict:
    def meta(lock):
        return {"label": lock.label, "model": lock.model, "quant": lock.quant,
                "runtime": lock.runtime, "samples": lock.samples}
    return {
        "baseline": meta(b),
        "candidate": meta(c),
        "max_drop": result.max_drop,
        "tools_changed": result.tools_changed,
        "regressed": result.regressed,
        "rows": [
            {"capability": r.capability, "baseline": r.baseline, "candidate": r.candidate,
             "delta": r.delta, "status": r.status, "significant": r.significant}
            for r in result.rows
        ],
    }


_HTML_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
max-width:780px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.45}
h1{font-size:1.2rem;margin:0 0 .2rem}.sub{color:#6b7280;font-size:.9rem;margin:0 0 1rem}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:.45rem .6rem;border-bottom:1px solid #eee}th{text-align:left;color:#6b7280;font-weight:600}
td.cap{font-family:ui-monospace,Menlo,monospace}td.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:.5rem;border-radius:3px;background:#eceef1;min-width:90px}
.bar>span{display:block;height:100%;border-radius:3px}
.badge{font-size:.72rem;font-weight:700;padding:.08rem .42rem;border-radius:4px;white-space:nowrap}
.ok,.good{background:#dcfce7;color:#166534}.bad{background:#fee2e2;color:#991b1b}
.warn{background:#fef9c3;color:#854d0e}.dim{background:#f3f4f6;color:#6b7280}
.banner{padding:.6rem .9rem;border-radius:6px;font-weight:700;margin:1rem 0}
.banner.fail{background:#fee2e2;color:#991b1b}.banner.pass{background:#dcfce7;color:#166534}
.note{color:#854d0e;font-size:.85rem;margin:.25rem 0}
footer{color:#9ca3af;font-size:.78rem;margin-top:1.5rem}
"""
_HTML_STATUS = {  # status -> (css class, label)
    "ok": ("ok", "ok"), "regression": ("bad", "REGRESSION"), "noisy": ("warn", "noisy ↓"),
    "improved": ("good", "improved"), "added": ("dim", "added"), "removed": ("bad", "REMOVED"),
}


def _html_diff(result, b, c) -> str:
    def esc(s):
        return _html.escape(str(s))

    rows = []
    for r in result.rows:
        cls, label = _HTML_STATUS.get(r.status, ("dim", r.status))
        pct = int(round((r.candidate or 0.0) * 100))
        color = "#16a34a" if (r.candidate or 0) >= 0.9 else "#ca8a04" if (r.candidate or 0) >= 0.7 else "#dc2626"
        rows.append(
            f"<tr><td class='cap'>{esc(r.capability)}</td>"
            f"<td class='num'>{_cell(r.baseline)}</td>"
            f"<td class='num'>{_cell(r.candidate)}</td>"
            f"<td class='num'>{_cell(r.delta, signed=True)}</td>"
            f"<td><div class='bar'><span style='width:{pct}%;background:{color}'></span></div></td>"
            f"<td><span class='badge {cls}'>{esc(label)}</span></td></tr>"
        )
    banner = (
        f"<div class='banner fail'>FAIL — capabilities regressed or removed: "
        f"{esc(', '.join(r.capability for r in result.regressions))}</div>"
        if result.regressed else
        "<div class='banner pass'>PASS — no capability regressed.</div>"
    )
    notes = "".join(f"<p class='note'>{esc(n)}</p>" for n in _diff_notes(result, b, c))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{_HTML_CSS}</style>"
        "<title>probelock report</title></head><body>"
        f"<h1>probelock capability report</h1>"
        f"<p class='sub'>{esc(b.label or 'baseline')} &nbsp;→&nbsp; {esc(c.label or 'candidate')}</p>"
        f"{banner}"
        "<table><thead><tr><th>Capability</th><th>Baseline</th><th>Candidate</th>"
        "<th>Δ</th><th></th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{notes}"
        "<footer>Generated by probelock — deterministic, no LLM judge.</footer>"
        "</body></html>"
    )


@app.command()
def diff(
    baseline: Path = typer.Argument(..., help="Baseline lockfile."),
    candidate: Path = typer.Argument(..., help="Candidate lockfile."),
    max_drop: float = typer.Option(0.05, "--max-drop", help="Regression threshold."),
    confidence: Optional[float] = typer.Option(
        None, "--confidence", help="If set (e.g. 0.95), mark sub-significant drops 'noisy'."
    ),
    fmt: str = typer.Option("table", "--format", help="table | markdown | json | html"),
) -> None:
    """Show within-model capability deltas between two lockfiles (informational)."""
    _validate_confidence(confidence)
    b, c = _read_lock(baseline), _read_lock(candidate)
    result = diff_lockfiles(b, c, max_drop, confidence)
    if fmt == "markdown":
        print(_markdown_diff(result, b, c))
    elif fmt == "json":
        print(json.dumps(_diff_payload(result, b, c), indent=2))
    elif fmt == "html":
        print(_html_diff(result, b, c))
    elif fmt == "table":
        _render_diff(result, b, c)
    else:
        _err(f"Unknown --format '{fmt}' (use table | markdown | json | html).")
        raise typer.Exit(2)


@app.command()
def gate(
    baseline: Path = typer.Option(..., "--baseline", "-b", help="Baseline (committed) lockfile."),
    candidate: Path = typer.Option(..., "--candidate", "-c", help="Candidate lockfile."),
    max_drop: float = typer.Option(0.05, "--max-drop", help="Regression threshold."),
    require_same_model: bool = typer.Option(
        False, "--require-same-model", help="Fail if baseline and candidate are different models."
    ),
    confidence: Optional[float] = typer.Option(
        None, "--confidence", help="Only fail on drops significant at this confidence (e.g. 0.95)."
    ),
) -> None:
    """CI gate: exit 1 if any capability regressed (or was dropped) beyond --max-drop.

    With --confidence, a drop past --max-drop that isn't statistically significant
    for the recorded sample count is reported as 'noisy' and does NOT fail the gate.
    Exit 2 is reserved for invalid input (bad lockfile, or a cross-model
    comparison under --require-same-model), so CI can tell the two apart.
    """
    _validate_confidence(confidence)
    b, c = _read_lock(baseline), _read_lock(candidate)
    result = diff_lockfiles(b, c, max_drop, confidence)
    _render_diff(result, b, c)

    noisy = [r for r in result.rows if r.status == "noisy"]
    if noisy:
        console.print(
            f"[yellow]{len(noisy)} drop(s) past --max-drop are below the {confidence} "
            f"confidence bar (noisy ↓) — raise --samples to confirm or clear them.[/]"
        )

    if require_same_model and b.model and c.model and b.model != c.model:
        console.print(
            f"\n[bold red]INVALID[/] — --require-same-model set but models differ "
            f"({escape(b.model)} vs {escape(c.model)})."
        )
        raise typer.Exit(2)
    if result.regressed:
        names = escape(", ".join(r.capability for r in result.regressions))
        console.print(f"\n[bold red]FAIL[/] — capabilities regressed or removed: {names}")
        raise typer.Exit(1)
    console.print(f"\n[bold green]PASS[/] — no capability regressed beyond {max_drop:.2f}.")


_TEMPLATE_TOOLS = """\
[
  {
    "type": "function",
    "function": {
      "name": "create_event",
      "description": "Create a calendar event",
      "parameters": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "start": {"type": "string", "description": "ISO 8601 datetime"},
          "visibility": {"type": "string", "enum": ["public", "private"]}
        },
        "required": ["title", "start"]
      }
    }
  }
]
"""

_TEMPLATE_WORKFLOW = """\
name: probelock
on: [pull_request]

jobs:
  capabilities:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      # Point --endpoint at your model server. CI needs network access to it
      # (a hosted endpoint, or a self-hosted runner with Ollama/llama.cpp).
      # Uses the published `probelock` from PyPI. To pin an unreleased revision,
      # replace `uvx probelock` with:
      #   uvx --from git+https://github.com/kelkalot/probelock probelock ...
      - name: Probe candidate
        run: uvx probelock probe --tools probelock.tools.json
             --endpoint "$LLM_ENDPOINT" --model "$LLM_MODEL"
             --samples 5 --temperature 0.7 -o candidate.lock
        env:
          LLM_ENDPOINT: ${{ secrets.LLM_ENDPOINT }}
          LLM_MODEL: ${{ vars.LLM_MODEL }}
      - name: Gate on regression
        run: uvx probelock gate --baseline probelock.lock --candidate candidate.lock
             --max-drop 0.05 --confidence 0.95
"""


@app.command()
def init(
    path: Path = typer.Option(Path("."), "--path", help="Directory to scaffold into."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Scaffold a tools file and a CI workflow to get started."""
    targets = [
        (path / "probelock.tools.json", _TEMPLATE_TOOLS),
        (path / ".github" / "workflows" / "probelock.yml", _TEMPLATE_WORKFLOW),
    ]
    for target, content in targets:
        if target.exists() and not force:
            console.print(f"[yellow]exists, skipped[/] {target} (use --force)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        console.print(f"[green]created[/] {target}")
    console.print(
        "\nNext: probe your model and commit the baseline, then gate candidates in CI:\n"
        "  probelock probe --tools probelock.tools.json "
        "--endpoint http://localhost:11434/v1 --model <model> -o probelock.lock\n"
        "  git add probelock.lock   # this is your committed baseline"
    )


@app.command()
def version() -> None:
    """Print the probelock version."""
    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
