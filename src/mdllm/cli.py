"""MultiDeviceLLM CLI — `mdllm`.

Ergonomic control layer over the exo distributed inference engine:

    mdllm doctor      # check prerequisites
    mdllm bootstrap   # clone + build the exo engine
    mdllm up          # start this device as a cluster node
    mdllm status      # show the live cluster (devices + pooled memory)
    mdllm models      # list / search available models
    mdllm fit MODEL   # can the cluster run this model? which split is best?
    mdllm run MODEL   # load a model across the cluster and chat with it
"""

from __future__ import annotations

import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__, bootstrap, config
from .client import ExoClient, ExoNotRunning, Placement
from .planner import best_placement, human_bytes, rank_placements, valid_placements

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run one LLM across your phone, laptop, and more by pooling their memory.",
)
console = Console()


def _client() -> ExoClient:
    return ExoClient()


def _run_shell(commands: list[str], dry_run: bool) -> int:
    for cmd in commands:
        console.print(f"[dim]$ {cmd}[/dim]")
        if dry_run:
            continue
        rc = subprocess.call(cmd, shell=True)
        if rc != 0:
            console.print(f"[red]Command failed (exit {rc}):[/red] {cmd}")
            return rc
    return 0


@app.command()
def version() -> None:
    """Print the MultiDeviceLLM version."""
    console.print(f"MultiDeviceLLM {__version__}")


@app.command()
def doctor() -> None:
    """Check that all prerequisites for running the exo engine are installed."""
    table = Table(title="MultiDeviceLLM • prerequisites", show_lines=False)
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail / fix")

    all_ok = True
    for c in bootstrap.doctor():
        required = "optional" not in c.name
        if not c.ok and required:
            all_ok = False
        mark = "[green]OK[/green]" if c.ok else "[red]MISSING[/red]"
        detail = c.detail if c.ok else f"[yellow]{c.hint}[/yellow]"
        table.add_row(c.name, mark, detail)

    console.print(table)
    if all_ok:
        console.print("\n[green]All required tools present.[/green] "
                      "Next: [bold]mdllm bootstrap[/bold]")
    else:
        console.print("\n[red]Some required tools are missing.[/red] "
                      "Install them, then re-run [bold]mdllm doctor[/bold].")
        raise typer.Exit(1)


@app.command(name="bootstrap")
def bootstrap_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the commands without running them."
    ),
) -> None:
    """Clone and build the exo engine into ./vendor/exo."""
    console.print(Panel(f"exo will be installed to:\n[bold]{config.EXO_DIR}[/bold]",
                        title="bootstrap"))
    cmds = bootstrap.clone_or_update_exo() + bootstrap.build_commands()
    rc = _run_shell(cmds, dry_run)
    if rc != 0:
        raise typer.Exit(rc)
    if not dry_run:
        console.print("\n[green]Engine ready.[/green] Start it with [bold]mdllm up[/bold]")


@app.command()
def up(
    worker: bool = typer.Option(
        True, "--worker/--no-worker",
        help="Run inference here (--worker) or act as a coordinator only.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Start this device as a node in the cluster (runs exo in the foreground)."""
    if not config.EXO_DIR.exists():
        console.print("[red]exo is not installed.[/red] Run [bold]mdllm bootstrap[/bold] first.")
        raise typer.Exit(1)
    cmd = bootstrap.run_command(worker=worker)
    console.print(Panel(
        "Starting exo. Other devices on the same network will auto-discover "
        "this node.\nDashboard + API: [bold]http://localhost:52415[/bold]\n"
        "Press Ctrl-C to leave the cluster.",
        title="mdllm up",
    ))
    if dry_run:
        console.print(f"[dim]$ {cmd}[/dim]")
        return
    try:
        subprocess.call(cmd, shell=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Left the cluster.[/yellow]")


@app.command()
def status() -> None:
    """Show the live cluster: connected devices and pooled memory."""
    client = _client()
    try:
        state = client.state()
    except ExoNotRunning as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    nodes = _extract_nodes(state)
    table = Table(title="MultiDeviceLLM • cluster")
    table.add_column("Device")
    table.add_column("Model / chip")
    table.add_column("Memory", justify="right")

    total_mem = 0
    for n in nodes:
        mem = n.get("memory", 0) or 0
        total_mem += mem
        table.add_row(
            str(n.get("id", "?"))[:20],
            str(n.get("model", n.get("chip", ""))),
            human_bytes(mem) if mem else "-",
        )
    console.print(table)
    console.print(
        f"\n[bold]{len(nodes)}[/bold] device(s), "
        f"[bold]{human_bytes(total_mem)}[/bold] pooled memory."
    )

    instances = _extract_instances(state)
    if instances:
        console.print("\n[bold]Loaded models:[/bold]")
        for inst in instances:
            console.print(
                f"  • {inst.get('model_id', '?')}  "
                f"[dim]id={inst.get('id', '?')}[/dim]"
            )
    else:
        console.print("\n[dim]No models loaded. Try: mdllm run llama-3.2-1b[/dim]")


def _extract_nodes(state: dict) -> list[dict]:
    """Be liberal in what we accept — exo's state schema evolves."""
    for key in ("nodes", "topology", "devices"):
        val = state.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return list(val.values())
    return []


def _extract_instances(state: dict) -> list[dict]:
    for key in ("instances", "deployments"):
        val = state.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, dict):
            return list(val.values())
    return []


@app.command()
def models(
    search: str = typer.Option(None, "--search", "-s", help="Search HuggingFace."),
    downloaded: bool = typer.Option(False, "--downloaded", help="Only downloaded."),
) -> None:
    """List available models (or search HuggingFace)."""
    client = _client()
    try:
        data = (
            client.search_models(search)
            if search
            else client.models(status="downloaded" if downloaded else None)
        )
    except ExoNotRunning as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    items = data if isinstance(data, list) else data.get("models", data.get("data", []))
    table = Table(title="Models" + (f" matching '{search}'" if search else ""))
    table.add_column("Model ID")
    table.add_column("Status / info")
    for m in items[:50]:
        if isinstance(m, str):
            table.add_row(m, "")
        else:
            table.add_row(
                str(m.get("id", m.get("model_id", "?"))),
                str(m.get("status", m.get("downloaded", ""))),
            )
    console.print(table)


def _placement_table(placements: list[Placement], title: str) -> Table:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Split")
    table.add_column("Devices", justify="right")
    table.add_column("Per-device memory")
    for i, p in enumerate(rank_placements(placements)):
        per_node = ", ".join(
            f"{k[:8]}={human_bytes(v)}" for k, v in p.memory_delta_by_node.items()
        )
        table.add_row(str(i), p.sharding, str(len(p.nodes)), per_node)
    return table


@app.command()
def fit(model: str = typer.Argument(..., help="Model id, e.g. llama-3.2-1b")) -> None:
    """Check whether the cluster can run a model, and show the best split."""
    client = _client()
    try:
        placements = client.previews(model)
    except ExoNotRunning as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    valid = valid_placements(placements)
    if not valid:
        console.print(
            f"[red]'{model}' does not fit on the current cluster.[/red]\n"
            "Add another device (mdllm up on your phone/laptop) or pick a "
            "smaller / more-quantized model."
        )
        if placements:
            errs = {p.error for p in placements if p.error}
            for e in list(errs)[:3]:
                console.print(f"  [dim]{e}[/dim]")
        raise typer.Exit(1)

    console.print(_placement_table(valid, f"'{model}' fits — {len(valid)} placement(s)"))
    best = best_placement(valid)
    if best:
        console.print(
            f"\n[green]Recommended:[/green] {best.sharding} across "
            f"{len(best.nodes)} device(s)."
        )


@app.command()
def load(
    model: str = typer.Argument(..., help="Model id, e.g. llama-3.2-1b"),
    choice: int = typer.Option(
        None, "--choice", "-c", help="Placement index from `mdllm fit` (default: best)."
    ),
    timeout: float = typer.Option(600, help="Seconds to wait for readiness."),
) -> str:
    """Load a model across the cluster (preview -> create -> await ready)."""
    client = _client()
    try:
        placements = client.previews(model)
    except ExoNotRunning as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    ranked = rank_placements(placements)
    if not ranked:
        console.print(f"[red]'{model}' does not fit on the current cluster.[/red]")
        raise typer.Exit(1)

    placement = ranked[choice] if choice is not None else ranked[0]
    console.print(
        f"Loading [bold]{placement.model_id}[/bold] "
        f"({placement.sharding} across {len(placement.nodes)} device(s))…"
    )
    client.create_instance(placement.instance)

    with console.status("Waiting for all devices to load their shard…"):
        evt = client.await_ready(placement.model_id, timeout_seconds=timeout)

    if evt.get("type") != "ready":
        console.print(f"[red]Model did not become ready:[/red] {evt.get('type')}")
        raise typer.Exit(1)
    console.print("[green]Model is ready.[/green]")
    return placement.model_id


@app.command()
def chat(
    model: str = typer.Argument(..., help="Full model id, e.g. mlx-community/Llama-3.2-1B-Instruct-4bit"),
) -> None:
    """Interactive chat with a model that is already loaded."""
    _chat_loop(_client(), model)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model id, e.g. llama-3.2-1b"),
    choice: int = typer.Option(None, "--choice", "-c"),
) -> None:
    """Load a model across the cluster and immediately start chatting."""
    model_id = load(model, choice=choice, timeout=600)
    _chat_loop(_client(), model_id)


def _chat_loop(client: ExoClient, model: str) -> None:
    console.print(Panel(
        f"Chatting with [bold]{model}[/bold] running across your cluster.\n"
        "Type your message and press Enter. Ctrl-C or 'exit' to quit.",
        title="mdllm chat",
    ))
    history: list[dict[str, str]] = []
    try:
        while True:
            try:
                user = console.input("[bold cyan]you › [/bold cyan]").strip()
            except EOFError:
                break
            if user.lower() in {"exit", "quit", ":q"}:
                break
            if not user:
                continue
            history.append({"role": "user", "content": user})
            console.print("[bold magenta]llm › [/bold magenta]", end="")
            reply = ""
            try:
                for delta in client.chat_stream(model, history):
                    reply += delta
                    console.print(delta, end="")
                    sys.stdout.flush()
            except ExoNotRunning as e:
                console.print(f"\n[red]{e}[/red]")
                return
            console.print()
            history.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        console.print("\n[yellow]Bye.[/yellow]")


if __name__ == "__main__":
    app()
