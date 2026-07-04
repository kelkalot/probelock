"""probelock command line interface."""

from __future__ import annotations

import datetime as _dt
import hashlib
import html as _html
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .clients import AnyLlmClient, ClientError, HttpClient, LiteLlmClient, SimulatedClient
from .diff import diff_lockfiles, error_derived_negative_capabilities
from .ingest import FORMATS, REDACT_PATTERNS, MiningConfig, ingest_files
from .lockfile import read_lockfile, write_lockfile
from .mined import (
    AUTO_ACCEPT_SAFE,
    CATEGORIES,
    NEVER_AUTO_ACCEPT,
    accepted_probes,
    auto_accept as batch_accept,
    edit_expected_tool,
    load_mined,
    mined_fingerprint,
    save_mined,
    to_probe,
)
from .probes import derive_probes, tools_fingerprint
from .runner import run_probes
from .traces import derive_traced_probes, load_trace_records, traces_fingerprint

app = typer.Typer(
    add_completion=False,
    help="probelock — a capability lockfile for local models. Catch silent "
    "regressions when you swap a model, quant, or runtime.",
)
traces_app = typer.Typer(add_completion=False, help="Work with trace-mined probes.")
app.add_typer(traces_app, name="traces")
console = Console()
err_console = Console(stderr=True)


def _err(msg: str) -> None:
    """Print a red error line, escaping dynamic text so brackets in paths or
    messages (e.g. 'probelock[anyllm]') aren't eaten by rich's markup parser."""
    err_console.print(f"[red]{escape(msg)}[/]")


_BAR = 24

_TRACES_OPTION = typer.Option(
    None, "--traces", help="Optional trace-export JSON to add real, replayed decision "
    "points to the battery (see README: Deriving probes from real traces)."
)

_MINED_OPTION = typer.Option(
    None, "--mined", help="Optional mined probes file from `probelock ingest` — only "
    "accepted (reviewed) probes join the battery."
)


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


def _load_traces_or_exit(path: Path):
    """Load a trace-export file, converting any bad-input error to a clean Exit(2) —
    same convention as _load_tools_or_exit/_load_json_or_exit."""
    try:
        return load_trace_records(path)
    except FileNotFoundError:
        _err(f"Traces file not found: {path}")
        raise typer.Exit(2)
    except OSError as exc:
        _err(f"Could not read traces file {path}: {exc}")
        raise typer.Exit(2)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        _err(f"Invalid traces file {path}: {exc}")
        raise typer.Exit(2)


def _load_mined_or_exit(path: Path):
    try:
        return load_mined(path)
    except FileNotFoundError:
        _err(f"Mined probes file not found: {path}")
        raise typer.Exit(2)
    except OSError as exc:
        _err(f"Could not read mined probes file {path}: {exc}")
        raise typer.Exit(2)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        _err(f"Invalid mined probes file {path}: {exc}")
        raise typer.Exit(2)


def _mined_battery(path: Path):
    """Load a mined probes file and split it into the replayable battery slice.
    Returns (accepted MinedProbes, converted Probes); reports pending/rejected counts
    so a user who forgot to review isn't left wondering where their probes went."""
    all_mined = _load_mined_or_exit(path)
    accepted = accepted_probes(all_mined)
    pending = sum(1 for p in all_mined if p.status == "pending")
    if pending:
        err_console.print(
            f"[yellow]{pending} mined probe(s) still pending review — run "
            f"`probelock traces review {path}` to activate them.[/]"
        )
    if not accepted:
        err_console.print(f"[yellow]0 accepted probe(s) in {path}; none added.[/]")
    return accepted, [to_probe(p) for p in accepted]


_TOOLS_OPTION = typer.Option(
    None, "--tools", "-t", help="OpenAI-style tools JSON file (optional when --traces "
    "or --mined supply probes — those carry their own embedded tool definitions)."
)


def _require_probe_source(tools, traces, mined) -> None:
    if tools is None and traces is None and mined is None:
        _err("Provide at least one probe source: --tools, --traces, or --mined.")
        raise typer.Exit(2)


@app.command()
def derive(
    tools: Optional[Path] = _TOOLS_OPTION,
    traces: Optional[Path] = _TRACES_OPTION,
    mined: Optional[Path] = _MINED_OPTION,
) -> None:
    """Show the probe battery that would be generated from a toolset (transparency)."""
    _require_probe_source(tools, traces, mined)
    probes = derive_probes(_load_tools_or_exit(tools)) if tools is not None else []
    if traces is not None:
        probes = probes + derive_traced_probes(_load_traces_or_exit(traces))
    if mined is not None:
        probes = probes + _mined_battery(mined)[1]
    table = Table(title=f"{len(probes)} probes derived", expand=True)
    table.add_column("Probe id", no_wrap=True)
    table.add_column("Capability", no_wrap=True)
    table.add_column("Checks")
    for p in probes:
        table.add_row(p.id, p.capability, p.description)
    console.print(table)


@app.command()
def probe(
    tools: Optional[Path] = _TOOLS_OPTION,
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
    traces: Optional[Path] = _TRACES_OPTION,
    mined: Optional[Path] = _MINED_OPTION,
    allow_sensitive: bool = typer.Option(
        False, "--allow-sensitive", help="Let sensitive mined probes (verbatim real "
        "conversation content) into a written lockfile run."
    ),
) -> None:
    """Run the probe battery and produce a capability lockfile."""
    _require_probe_source(tools, traces, mined)
    if tools is not None:
        tool_list = _load_tools_or_exit(tools)
        probes = derive_probes(tool_list)
        fingerprint = tools_fingerprint(tool_list)
    else:
        # Trace-only run: traced/mined probes replay their own embedded tool
        # definitions, so there is no schema battery and no toolset to fingerprint.
        # The empty fingerprint is stable, so two trace-only lockfiles diff cleanly,
        # while a trace-only vs schema-derived pair still flags tools_changed.
        probes = []
        fingerprint = ""

    traces_fp = None
    if traces is not None:
        records = _load_traces_or_exit(traces)
        traced_probes = derive_traced_probes(records)
        probes = probes + traced_probes
        traces_fp = traces_fingerprint(records)
        err_console.print(
            f"[yellow]{len(traced_probes)} probe(s) derived from real trace content in "
            f"{traces} — this may contain real user data; review before committing the "
            f"traces file or the resulting lockfile.[/]"
        )

    mined_fp = None
    if mined is not None:
        accepted, mined_battery = _mined_battery(mined)
        sensitive = [p for p in accepted if p.sensitive]
        if sensitive and out is not None and not allow_sensitive:
            # A lockfile is meant to be committed; probes frozen from verbatim real
            # conversations are not — refuse to bind the one to the other unless the
            # user explicitly opts in (see README: mining probes from raw agent logs).
            err_console.print(
                f"[yellow]{len(sensitive)} sensitive mined probe(s) excluded from this "
                f"lockfile run — their frozen contexts hold verbatim real conversation "
                f"content. Pass --allow-sensitive to include them anyway, or re-ingest "
                f"with --redact-patterns for committable probes.[/]"
            )
            accepted = [p for p in accepted if not p.sensitive]
            mined_battery = [to_probe(p) for p in accepted]
        if accepted:
            probes = probes + mined_battery
            mined_fp = mined_fingerprint(accepted)
            if any(p.sensitive for p in accepted):
                err_console.print(
                    f"[yellow]{len(mined_battery)} mined probe(s) added from real trace "
                    f"content in {mined} — this may contain real user data.[/]"
                )

    # One trace-input fingerprint in the lockfile, covering whichever real-trace sources
    # fed this battery — diff's traces_changed note stays a single, reliable signal.
    fps = [fp for fp in (traces_fp, mined_fp) if fp]
    if len(fps) == 2:
        traces_fp = hashlib.sha256("+".join(fps).encode()).hexdigest()[:16]
    elif fps:
        traces_fp = fps[0]

    if not probes:
        # Possible without --tools: a traces file that derives nothing, or a mined
        # file with no accepted probes. A zero-probe lockfile can gate nothing.
        _err("The probe battery is empty — nothing derived from the given sources.")
        raise typer.Exit(2)

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
        lock = run_probes(
            client, probes, fingerprint, __version__, samples=samples,
            traces_fingerprint=traces_fp,
        )
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


@app.command()
def ingest(
    logs: List[Path] = typer.Argument(
        ..., help="JSONL log(s) of raw agent traffic (recording-proxy output — pass "
        "rotated segments together so sessions spanning a rotation still stitch — or "
        "your own request/response logging)."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Write the mined probes file here."),
    fmt: str = typer.Option("auto", "--format", help="auto | trace-v1 | openai-jsonl"),
    min_agreement: int = typer.Option(
        2, "--min-agreement", help="Distinct sessions that must agree to confirm a "
        "tool-selection probe when no continuation evidence exists."
    ),
    min_agreement_notool: int = typer.Option(
        3, "--min-agreement-notool", help="Distinct sessions required for a no-tool "
        "probe (stricter: a mislabeled one freezes a model mistake as expected)."
    ),
    per_capability: int = typer.Option(
        8, "--per-capability", help="Max probes kept per (tool, category) pair."
    ),
    max_context_tokens: int = typer.Option(
        8192, "--max-context-tokens", help="Skip exchanges whose frozen context exceeds "
        "this (estimated) — bounds replay cost."
    ),
    redact_patterns: str = typer.Option(
        "", "--redact-patterns", help="Comma-separated scrubbers applied to message "
        "text: emails,phones,paths. Using this marks probes non-sensitive (committable)."
    ),
) -> None:
    """Mine probes from a raw agent traffic log.

    Everything lands with status "pending" — run `probelock traces review` to activate
    probes, then replay them with `probelock probe --mined`.
    """
    if fmt not in FORMATS:
        _err(f"Unknown --format '{fmt}' (use {' | '.join(FORMATS)}).")
        raise typer.Exit(2)
    patterns = tuple(p.strip().lower() for p in redact_patterns.split(",") if p.strip())
    for name in patterns:
        if name not in REDACT_PATTERNS:
            _err(
                f"Unknown redact pattern '{name}' (use {', '.join(sorted(REDACT_PATTERNS))})."
            )
            raise typer.Exit(2)

    config = MiningConfig(
        min_agreement=min_agreement,
        min_agreement_notool=min_agreement_notool,
        per_capability=per_capability,
        max_context_tokens=max_context_tokens,
        redact_patterns=patterns,
    )
    try:
        probes, summary = ingest_files(logs, fmt, config)
    except FileNotFoundError as exc:
        _err(f"Log file not found: {exc.filename or logs[0]}")
        raise typer.Exit(2)
    except OSError as exc:
        _err(f"Could not read log file: {exc}")
        raise typer.Exit(2)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        _err(f"Invalid log file: {exc}")
        raise typer.Exit(2)

    save_mined(probes, out, header={
        "generated_by": f"probelock {__version__}",
        "source": ", ".join(str(p) for p in logs),
    })

    console.print(
        f"\n[bold]{len(probes)} probe(s) mined[/] from {summary.records} record(s) — "
        f"{summary.sessions} session(s), {summary.clusters} distinct context(s)"
    )
    if summary.emitted:
        table = Table(expand=True)
        table.add_column("Category", no_wrap=True)
        table.add_column("Mined", justify="right")
        table.add_column("Candidates", justify="right")
        for cat in CATEGORIES:
            if summary.candidates.get(cat) or summary.emitted.get(cat):
                table.add_row(
                    cat, str(summary.emitted.get(cat, 0)), str(summary.candidates.get(cat, 0))
                )
        console.print(table)
    # No silent drops: everything the pipeline discarded, and why.
    for reason, n in sorted(summary.skipped.items()):
        console.print(f"[yellow]skipped {n} record(s): {reason}[/]")
    if summary.ambiguous_tool_selection:
        console.print(
            f"[yellow]{summary.ambiguous_tool_selection} context(s) had conflicting "
            f"cross-session tool agreement — not mined for tool selection.[/]"
        )
    if summary.unconfirmed_tool_clusters:
        console.print(
            f"[dim]{summary.unconfirmed_tool_clusters} tool-calling context(s) could not "
            f"be confirmed good — eligible for schema_validity only.[/]"
        )
    console.print(f"[green]wrote[/] {out}")
    if probes:
        sensitive_n = sum(1 for p in probes if p.sensitive)
        if sensitive_n:
            console.print(
                f"[yellow]{sensitive_n} probe(s) carry verbatim conversation content "
                f"(sensitive: true) — don't commit {out}; re-ingest with "
                f"--redact-patterns if you want committable probes.[/]"
            )
        console.print(f"\nNext: probelock traces review {out}")


@app.command(name="proxy")
def proxy_cmd(
    upstream: str = typer.Option(
        ..., "--upstream", help="OpenAI-compatible upstream base URL "
        "(e.g. http://127.0.0.1:8080 for llama.cpp, http://127.0.0.1:11434 for Ollama)."
    ),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Append trace-v1 JSONL records here. Created 0600: "
        "this file holds verbatim conversation content — treat it as sensitive."
    ),
    listen: str = typer.Option("127.0.0.1:8484", "--listen", help="host:port to listen on."),
    timeout: float = typer.Option(
        300.0, "--timeout", help="Upstream read timeout in seconds (local generations are slow)."
    ),
    connect_timeout: float = typer.Option(
        10.0, "--connect-timeout", help="Upstream connect timeout in seconds (fail fast when down)."
    ),
    max_size: float = typer.Option(
        100.0, "--max-size", help="Rotate the log past this many MB (0 = never)."
    ),
    max_age: float = typer.Option(
        0.0, "--max-age", help="Rotate the log after this many minutes (0 = never)."
    ),
) -> None:
    """Record real agent traffic for `probelock ingest`.

    Point your agent's base_url at this proxy; requests are forwarded to --upstream
    unchanged and each completed chat-completions exchange is appended asynchronously
    as one trace-v1 record. Recording never blocks or fails a request.
    """
    import signal
    import threading

    from .proxy import ProxyConfig, start_proxy, stop_proxy

    host, _, port_text = listen.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        port = -1
    if not host or not (0 <= port <= 65535):
        _err(f"Invalid --listen '{listen}' (use host:port).")
        raise typer.Exit(2)

    config = ProxyConfig(
        upstream=upstream, out=out, listen_host=host, listen_port=port,
        timeout=timeout, connect_timeout=connect_timeout,
        max_size_mb=max_size, max_age_min=max_age,
    )
    try:
        server = start_proxy(config)
    except ValueError as exc:
        _err(str(exc))
        raise typer.Exit(2)
    except OSError as exc:  # bind failure, unwritable --out parent, ...
        _err(f"Could not start proxy: {exc}")
        raise typer.Exit(2)

    # The OS-assigned address, not the flag value — with --listen host:0 the bound
    # port is exactly the piece of information the banner exists to provide.
    host, port = server.server_address[0], server.server_address[1]
    console.print(f"[bold]probelock proxy[/] listening on [cyan]http://{host}:{port}[/] "
                  f"→ {escape(upstream)}")
    console.print(f"Point your agent at [cyan]base_url=\"http://{host}:{port}/v1\"[/]")
    console.print(
        f"Recording chat completions to [bold]{escape(str(out))}[/] — verbatim "
        f"conversation content; keep it out of version control (redaction happens "
        f"later, at `probelock ingest`)."
    )
    console.print("[dim]Ctrl-C to stop.[/]")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())  # supervisors send TERM, not INT
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stop_proxy(server)

    writer = server.writer
    console.print(
        f"\n[bold]{writer.written}[/] record(s) written"
        + (f", [yellow]{writer.dropped} dropped[/]" if writer.dropped else "")
        + (f", {writer.rotated} rotation(s)" if writer.rotated else "")
        + (f", [yellow]{server.capture_failures} capture failure(s)[/]"
           if server.capture_failures else "")
    )
    if writer.written or writer.rotated:
        # After rotation, one directory-qualified glob covers the live file AND every
        # rotated segment (agent.jsonl, agent-<stamp>.jsonl) — and keeps working even
        # if the live file itself was just rotated away.
        source = out.with_name(f"{out.stem}*{out.suffix}") if writer.rotated else out
        console.print(f"\nNext: probelock ingest {source} --out probes/mined.json")


def _final_user_turn(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return "(no user turn)"


def _render_mined_probe(position: str, mp) -> None:
    console.rule(f"[bold]{escape(mp.id)}[/]  {position}")
    turn = _final_user_turn(mp.messages)
    if len(turn) > 400:
        turn = turn[:400] + "…"
    ref = mp.reference or {}
    if mp.category == "no_tool":
        recorded = "answered in text (no tool call)"
    elif ref.get("tool"):
        recorded = f"called {ref['tool']} {json.dumps(ref.get('valid_args', {}))}"
    else:
        recorded = "(not recorded)"
    prov = mp.provenance
    console.print(f"[bold]category[/]   {escape(mp.category)}")
    console.print(f"[bold]check[/]      {escape(json.dumps(mp.check()))}")
    console.print(f"[bold]user turn[/]  {escape(turn)}")
    console.print(f"[bold]recorded[/]   {escape(recorded)}")
    console.print(
        f"[bold]provenance[/] {prov.get('sessions', '?')} session(s) via "
        f"{escape(str(prov.get('rule', '?')))} — model {escape(str(prov.get('model') or '?'))}, "
        f"{len(mp.messages)} message(s)"
        + (", [yellow]sensitive[/]" if mp.sensitive else "")
    )


@traces_app.command()
def review(
    mined_file: Path = typer.Argument(..., help="Mined probes file from `probelock ingest`."),
    auto_accept: List[str] = typer.Option(
        [], "--auto-accept", help="Batch-accept a category without prompting. Only "
        "schema_validity qualifies (its mining needed no correctness inference)."
    ),
    auto_accept_all: bool = typer.Option(
        False, "--auto-accept-all", help="Batch-accept every pending category except "
        "no_tool. Requires --i-know-what-im-doing."
    ),
    i_know_what_im_doing: bool = typer.Option(
        False, "--i-know-what-im-doing", help="Acknowledge that batch-accepting "
        "inferred probes can freeze model mistakes into the baseline."
    ),
) -> None:
    """Review pending mined probes: y accept, n reject, e edit expected tool,
    a accept all in category, s skip, q quit (progress is saved)."""
    probes = _load_mined_or_exit(mined_file)
    # Keep whatever header (source, generated_by, ...) ingest wrote alongside the probes.
    raw = _load_json_or_exit(mined_file, "mined probes file")
    header = {k: v for k, v in raw.items() if k not in ("probes", "version")}

    cats = frozenset(c.strip().lower().replace("-", "_") for c in auto_accept if c.strip())
    unknown = cats - set(CATEGORIES)
    if unknown:
        _err(f"Unknown --auto-accept category: {', '.join(sorted(unknown))} "
             f"(use {', '.join(CATEGORIES)}).")
        raise typer.Exit(2)
    unsafe = cats - AUTO_ACCEPT_SAFE
    if unsafe:
        _err(
            f"--auto-accept only covers {', '.join(sorted(AUTO_ACCEPT_SAFE))}; "
            f"{', '.join(sorted(unsafe))} was inferred from traces and needs review "
            f"(or the explicit --auto-accept-all --i-know-what-im-doing)."
        )
        raise typer.Exit(2)
    if auto_accept_all and not i_know_what_im_doing:
        _err("--auto-accept-all requires --i-know-what-im-doing.")
        raise typer.Exit(2)

    if cats or auto_accept_all:
        counts = batch_accept(probes, cats, accept_all=auto_accept_all)
        save_mined(probes, mined_file, header=header)
        skipped_no_tool = counts.pop("no_tool_skipped", 0)
        total = sum(counts.values())
        console.print(f"[green]auto-accepted {total} probe(s)[/] "
                      f"({', '.join(f'{c}: {n}' for c, n in sorted(counts.items())) or 'none'})")
        if skipped_no_tool:
            console.print(
                f"[yellow]{skipped_no_tool} no_tool probe(s) left pending — no_tool has "
                f"no auto-accept path (a mislabeled probe freezes a model mistake as "
                f"expected behavior); accept them individually.[/]"
            )
        _review_summary(probes, mined_file)
        return

    pending = [p for p in probes if p.status == "pending"]
    if not pending:
        console.print("[green]nothing pending[/]")
        _review_summary(probes, mined_file)
        return

    console.print(
        f"{len(pending)} pending probe(s). "
        f"[bold]y[/] accept · [bold]n[/] reject · [bold]e[/] edit expected tool · "
        f"[bold]a[/] accept all in category · [bold]s[/] skip · [bold]q[/] quit"
    )
    stopped = False
    for i, mp in enumerate(pending, 1):
        if stopped or mp.status != "pending":  # 'a' may have accepted it already
            continue
        _render_mined_probe(f"({i}/{len(pending)})", mp)
        while True:
            try:
                choice = typer.prompt("y/n/e/a/s/q", default="s").strip().lower()
            except typer.Abort:  # EOF / Ctrl-C: keep the decisions made so far
                stopped = True
                break
            if choice in ("y", "yes"):
                mp.status, mp.provenance["review"] = "accepted", "accepted"
                break
            if choice in ("n", "no"):
                mp.status, mp.provenance["review"] = "rejected", "rejected"
                break
            if choice == "s":
                break
            if choice == "q":
                stopped = True
                break
            if choice == "e":
                try:
                    edit_expected_tool(mp, typer.prompt("expected tool").strip())
                    break
                except typer.Abort:
                    stopped = True
                    break
                except ValueError as exc:
                    console.print(f"[yellow]{escape(str(exc))}[/]")
                    continue
            if choice == "a":
                if mp.category in NEVER_AUTO_ACCEPT:
                    console.print(
                        "[yellow]no_tool probes have no accept-all path — a mislabeled "
                        "one freezes a model mistake as expected behavior. Accept them "
                        "individually.[/]"
                    )
                    continue
                n = 0
                for p in pending:
                    if p.status == "pending" and p.category == mp.category:
                        p.status, p.provenance["review"] = "accepted", "accepted"
                        n += 1
                console.print(f"[green]accepted {n} {escape(mp.category)} probe(s)[/]")
                break
            console.print("[yellow]y / n / e / a / s / q[/]")

    save_mined(probes, mined_file, header=header)
    if stopped:
        console.print("[yellow]review stopped — progress saved.[/]")
    _review_summary(probes, mined_file)


def _review_summary(probes, mined_file: Path) -> None:
    by_status = {}
    for p in probes:
        by_status[p.status] = by_status.get(p.status, 0) + 1
    console.print(
        "  ".join(f"{status}: {by_status.get(status, 0)}"
                  for status in ("accepted", "rejected", "pending"))
    )
    if by_status.get("accepted"):
        console.print(f"\nNext: probelock probe --tools <tools.json> --mined {mined_file} ...")


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
    if result.traces_changed:
        notes.append("⚠ trace inputs differ — comparison may not be apples-to-apples.")
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
        "traces_changed": result.traces_changed,
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
