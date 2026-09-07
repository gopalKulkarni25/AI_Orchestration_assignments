# Architecture Guidelines

All generated Python code must satisfy every rule below.
The Reviewer agent checks each rule and issues `VERDICT: REJECTED` if any are violated.

## Rules

1. **snake_case naming** — all function names, variable names, and parameter names must use snake_case.
2. **Type hints** — every parameter and every return value must have a type annotation.
3. **Docstrings** — every public function must have a docstring describing what it does, its parameters, its return value, and any exceptions raised.
4. **No global mutable state** — do not declare module-level mutable variables (lists, dicts, sets) that functions write to.
5. **Function body ≤ 30 lines** — no single function may exceed 30 lines (blank lines and comments excluded).
6. **No bare `except`** — every `except` clause must name the exception type(s) being caught.
7. **Imports at the top** — all `import` and `from … import` statements must appear at the top of the file, not inside functions or conditionals.
