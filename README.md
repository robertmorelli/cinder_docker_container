# cinder-docker-container

Local workspace for running `de_typer.py`, `de_type2.py`, and `de_typer_boxunbox.py` against `static-python-perf` inside the Docker image.

## Detyper Limitations

Current guardrails in `de_typer_boxunbox.py` intentionally skip or force-typed certain cases:

- Class members: body detyping does not rewrite `self.<field>` declarations, reads, or writes.
- Top-level body annassigns: module `AnnAssign` body detyping is forced typed.
- `body_cast_all`: `Optional[...]` and unions are skipped (no flow narrowing).
- `nogo_types`: annotations rooted at iterator/generator/protocol/callable types are never detyped.
  - `Iterator`, `Iterable`, `Generator`, `AsyncIterator`, `AsyncGenerator`, `Coroutine`, `Protocol`, `Callable`
- Ambiguous method names across unrelated class hierarchies are no-go for unresolved attribute-call boundary rewrites.
- Pure function calls (`foo(...)`) are still eligible for normal boundary rewriting.
