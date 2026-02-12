## General Principles

- Communicate intent precisely.
- Edge cases matter.
- Favor reading code over writing code.
- Runtime crashes are better than bugs.
- Compile errors are better than runtime crashes.
- Incremental improvements.
- Avoid local maximums.
- Reduce the amount one must remember.
- Focus on code rather than style.

## Do Not Go Rogue (Hard Rule)

- Unrequested edits are **unhygienic, disgusting, revolting, and unacceptable**.
- If the user asks to investigate, reproduce, fuzz, explain, compare, or show evidence, that is **read-only mode**.
- In read-only mode, do not modify files and do not run write commands (`apply_patch`, `sed -i`, redirections, autofix formatters, etc.).
- Do not “helpfully” patch behavior during analysis. That is out-of-scope, destructive and unhelpful.
- Only edit when the user explicitly asks for a change.
- Before first edit in any turn, state:
  - exact files to modify
  - exact requested change being implemented
- If an unrequested edit happens:
  - stop immediately
  - list every modified file
  - apologize plainly
  - ask whether to revert or keep
  - take zero further actions until user confirms
- Scope discipline is mandatory: do exactly what was asked, nothing extra.

