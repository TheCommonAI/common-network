"""HumanEval-style scoring: execute the completion against the item's unit tests."""
from bench.sandbox import run_program


def extract_code(model_output: str, entry_point: str) -> str:
    """Best-effort extraction of a code completion from a model's raw text output."""
    text = model_output.strip()
    if "```" in text:
        parts = text.split("```")
        fenced = parts[1::2]
        candidates = []
        for part in fenced:
            candidate = part
            if candidate.lstrip().lower().startswith("python"):
                candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
            candidates.append(candidate)
        for candidate in candidates:
            if entry_point in candidate:
                return candidate
        if candidates:
            return candidates[0]
    return text


def score(item: dict, model_output: str, *, allow_unsafe_exec: bool = False) -> tuple[bool, str]:
    """item needs prompt, entry_point, test. Returns (passed, error_detail)."""
    completion = extract_code(model_output, item["entry_point"])

    # Chat models often return the whole function rather than continuing the
    # prompt's signature -- handle both shapes.
    if f"def {item['entry_point']}" in completion:
        program = completion
    else:
        program = item["prompt"] + completion

    full_source = program + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    return run_program(full_source, allow_unsafe_exec=allow_unsafe_exec)
