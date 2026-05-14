"""
Synthetic data generator for TaskSignalClassifier.

Produces labelled (text, complexity, nature, domain) tuples covering
all 45 combinations (3 complexity × 3 nature × 5 domain) plus edge cases.

Spec targets: ~10-13k examples total, 200-300 per combination, ~300 edge cases.
85%+ accuracy on held-out synthetic split.
"""
from __future__ import annotations

import random

# ── Templates by (complexity, nature, domain) ─────────────────────────────────

_TEMPLATES: dict[tuple[str, str, str], list[str]] = {

    # FOCUSED + MUTATIVE + coding
    ("focused", "mutative", "coding"): [
        "fix the bug in {file}",
        "patch the validation logic in {file}",
        "the test is failing please fix it",
        "repair the broken function",
        "correct the off-by-one error",
        "update the {func} method to handle null",
        "remove the deprecated call in {file}",
        "fix the type error in {file}",
        "resolve the merge conflict in {file}",
        "change the return type of {func}",
        "rename {func} to {func2}",
        "add missing null check",
        "fix the race condition in {file}",
    ],

    # FOCUSED + GENERATIVE + coding
    ("focused", "generative", "coding"): [
        "write a test for {func}",
        "add a unit test for the {func} method",
        "create a helper function that {action}",
        "generate a stub for {func}",
        "write a docstring for {func}",
        "add a type annotation to {func}",
        "create a mock for {service}",
        "write a test that covers the edge case",
        "add a simple wrapper around {func}",
        "scaffold the {class_name} class",
    ],

    # FOCUSED + READONLY + coding
    ("focused", "readonly", "coding"): [
        "explain what {func} does",
        "what does {file} do",
        "show me how {func} works",
        "read {file} and tell me the purpose",
        "what is the return type of {func}",
        "trace the call path to {func}",
        "describe the {class_name} interface",
    ],

    # MODERATE + MUTATIVE + coding
    ("moderate", "mutative", "coding"): [
        "refactor the {module} module",
        "migrate {file} from Python 2 to 3",
        "update all calls to use the new API",
        "replace the old authentication with JWT",
        "convert the class to use dataclasses",
        "rewrite the parser to handle edge cases",
        "split {file} into smaller modules",
        "extract the business logic from {file}",
        "update the database schema and migrations",
        "change the serialization format",
        "upgrade dependencies in {file}",
    ],

    # MODERATE + GENERATIVE + coding
    ("moderate", "generative", "coding"): [
        "add a rate limiter to the API",
        "write tests for the {module} module",
        "implement pagination for the {endpoint} endpoint",
        "add logging to the {module} service",
        "create a caching layer for {func}",
        "implement retry logic",
        "add input validation to {func}",
        "write integration tests for {module}",
        "implement the {feature} feature",
        "add error handling to the {module}",
        "build a CLI for {module}",
    ],

    # MODERATE + READONLY + coding
    ("moderate", "readonly", "coding"): [
        "review the {module} module",
        "audit the security of {file}",
        "analyze the performance of {module}",
        "check the test coverage for {module}",
        "review the API design of {module}",
        "look for bugs in {module}",
    ],

    # EXPANSIVE + MUTATIVE + coding
    ("expansive", "mutative", "coding"): [
        "rewrite the entire {module} from scratch",
        "migrate the whole codebase to TypeScript",
        "refactor the entire authentication system",
        "update all files to use the new framework",
        "migrate from REST to GraphQL across the codebase",
        "rewrite the database layer",
        "overhaul the entire error handling",
        "migrate all tests to pytest",
        "port the application to async/await",
    ],

    # EXPANSIVE + GENERATIVE + coding
    ("expansive", "generative", "coding"): [
        "implement a complete CI/CD pipeline",
        "build a full authentication system",
        "create a comprehensive test suite",
        "generate a complete SDK for the API",
        "implement the entire payment flow",
        "build a plugin architecture",
        "create a full observability stack",
    ],

    # EXPANSIVE + READONLY + coding
    ("expansive", "readonly", "coding"): [
        "analyze the entire codebase architecture",
        "give me a comprehensive review of the repo",
        "audit all the security vulnerabilities",
        "map out all the dependencies in the project",
        "review the entire test suite",
        "document all public APIs",
        "trace all code paths from the entry point",
    ],

    # FOCUSED + MUTATIVE + research
    ("focused", "mutative", "research"): [
        "update the citation for {paper}",
        "correct the formula in the notes",
        "fix the reference list",
        "update the year in the bibliography",
    ],

    # FOCUSED + GENERATIVE + research
    ("focused", "generative", "research"): [
        "write a brief summary of {paper}",
        "generate a one-paragraph abstract",
        "create a citation for {paper}",
        "draft a short introduction paragraph",
    ],

    # FOCUSED + READONLY + research
    ("focused", "readonly", "research"): [
        "what does {paper} say about {topic}",
        "find the key findings in {paper}",
        "what are the limitations of {paper}",
        "summarize the methodology of {paper}",
    ],

    # MODERATE + MUTATIVE + research
    ("moderate", "mutative", "research"): [
        "update the literature review section",
        "revise the findings based on new data",
        "rewrite the methodology section",
        "update the analysis with the latest numbers",
    ],

    # MODERATE + GENERATIVE + research
    ("moderate", "generative", "research"): [
        "write a literature review on {topic}",
        "generate a survey of recent papers on {topic}",
        "compile findings from multiple studies on {topic}",
        "write a comparative analysis of {topic}",
        "create an annotated bibliography for {topic}",
    ],

    # MODERATE + READONLY + research
    ("moderate", "readonly", "research"): [
        "research what the literature says about {topic}",
        "compare studies on {topic}",
        "review recent papers on {topic}",
        "analyze the evidence for {topic}",
        "find contradictions in the research on {topic}",
    ],

    # EXPANSIVE + MUTATIVE + research
    ("expansive", "mutative", "research"): [
        "rewrite the entire research proposal",
        "revise all sections of the paper",
        "update the complete methodology and results",
    ],

    # EXPANSIVE + GENERATIVE + research
    ("expansive", "generative", "research"): [
        "write a complete research paper on {topic}",
        "create a comprehensive literature survey on {topic}",
        "draft a full research report",
        "produce a systematic review of {topic}",
    ],

    # EXPANSIVE + READONLY + research
    ("expansive", "readonly", "research"): [
        "give me a comprehensive review of all research on {topic}",
        "analyze the complete body of literature on {topic}",
        "survey everything published about {topic} in the last decade",
    ],

    # FOCUSED + MUTATIVE + support
    ("focused", "mutative", "support"): [
        "fix my configuration for {service}",
        "update my settings so {service} works",
        "correct the config file",
    ],

    # FOCUSED + GENERATIVE + support
    ("focused", "generative", "support"): [
        "show me how to configure {service}",
        "give me an example of {action}",
        "write a config snippet for {service}",
    ],

    # FOCUSED + READONLY + support
    ("focused", "readonly", "support"): [
        "help me understand why {error} occurs",
        "explain what this error means: {error}",
        "why is {service} returning {error}",
        "what does {error} mean",
        "how do I fix {error}",
        "i'm getting {error} when I run {func}",
        "why doesn't my code work",
        "question about {func}",
        "confused about how {func} works",
        "guide me through setting up {service}",
    ],

    # MODERATE + MUTATIVE + support
    ("moderate", "mutative", "support"): [
        "update my project to fix all the dependency issues",
        "resolve all the errors in my config",
    ],

    # MODERATE + GENERATIVE + support
    ("moderate", "generative", "support"): [
        "help me set up a complete {service} environment",
        "walk me through the full setup process",
        "write a setup guide for {service}",
    ],

    # MODERATE + READONLY + support
    ("moderate", "readonly", "support"): [
        "explain the entire authentication flow",
        "walk me through how {service} works end to end",
        "help me understand the whole system",
    ],

    # EXPANSIVE + READONLY + support
    ("expansive", "readonly", "support"): [
        "explain everything about {service}",
        "give me a complete overview of how the system works",
    ],

    # EXPANSIVE + GENERATIVE + support
    ("expansive", "generative", "support"): [
        "create a comprehensive documentation site",
        "write the full onboarding guide",
    ],

    # EXPANSIVE + MUTATIVE + support
    ("expansive", "mutative", "support"): [
        "overhaul my entire configuration setup",
    ],

    # FOCUSED + READONLY + analysis
    ("focused", "readonly", "analysis"): [
        "what is the p99 latency for {endpoint}",
        "show me the error rate",
        "check the performance of {func}",
        "review the benchmark results",
        "evaluate the metrics for {module}",
        "assess the {metric} metric",
    ],

    # FOCUSED + GENERATIVE + analysis
    ("focused", "generative", "analysis"): [
        "create a chart for {metric}",
        "generate a performance report for {module}",
        "write a summary of the benchmark results",
        "produce a table of {metric} values",
    ],

    # FOCUSED + MUTATIVE + analysis
    ("focused", "mutative", "analysis"): [
        "update the dashboard metric for {metric}",
        "fix the incorrect statistic",
        "correct the measurement formula",
    ],

    # MODERATE + READONLY + analysis
    ("moderate", "readonly", "analysis"): [
        "analyze the performance metrics for {module}",
        "benchmark {module} against the baseline",
        "profile the {func} function",
        "measure the latency across all endpoints",
        "compare {module} performance vs {module2}",
        "audit the resource usage",
        "evaluate the overall system performance",
        "review the code quality metrics",
    ],

    # MODERATE + GENERATIVE + analysis
    ("moderate", "generative", "analysis"): [
        "generate a performance report for the past month",
        "create visualizations for all the metrics",
        "build a dashboard for {module} statistics",
        "write an analysis of the failure patterns",
    ],

    # MODERATE + MUTATIVE + analysis
    ("moderate", "mutative", "analysis"): [
        "update the analysis with the new data",
        "revise the performance baselines",
        "update all the dashboards to use the new metrics",
    ],

    # EXPANSIVE + READONLY + analysis
    ("expansive", "readonly", "analysis"): [
        "analyze all performance data across the entire system",
        "give me a comprehensive performance audit",
        "benchmark everything and give me a full report",
        "evaluate all metrics and patterns across the codebase",
    ],

    # EXPANSIVE + GENERATIVE + analysis
    ("expansive", "generative", "analysis"): [
        "create a complete observability suite",
        "build a comprehensive analytics platform",
        "generate all reports and dashboards for the system",
    ],

    # EXPANSIVE + MUTATIVE + analysis
    ("expansive", "mutative", "analysis"): [
        "overhaul all the metrics and reporting systems",
        "rewrite the entire analytics pipeline",
    ],

    # FOCUSED + READONLY + general
    ("focused", "readonly", "general"): [
        "what is {topic}",
        "tell me about {topic}",
        "what does {term} mean",
        "how does {topic} work",
    ],

    # FOCUSED + GENERATIVE + general
    ("focused", "generative", "general"): [
        "write a short description of {topic}",
        "draft a brief note about {topic}",
        "create a one-sentence summary of {topic}",
    ],

    # FOCUSED + MUTATIVE + general
    ("focused", "mutative", "general"): [
        "update the description of {topic}",
        "edit this text",
        "fix the typo",
    ],

    # MODERATE + READONLY + general
    ("moderate", "readonly", "general"): [
        "explain {topic} in detail",
        "give me an overview of {topic}",
        "describe the landscape of {topic}",
    ],

    # MODERATE + GENERATIVE + general
    ("moderate", "generative", "general"): [
        "write an article about {topic}",
        "create a presentation on {topic}",
        "draft a proposal for {topic}",
    ],

    # MODERATE + MUTATIVE + general
    ("moderate", "mutative", "general"): [
        "rewrite this section about {topic}",
        "update the document",
        "revise the plan",
    ],

    # EXPANSIVE + READONLY + general
    ("expansive", "readonly", "general"): [
        "give me a comprehensive understanding of {topic}",
        "survey everything about {topic}",
        "explain the entire landscape of {topic}",
    ],

    # EXPANSIVE + GENERATIVE + general
    ("expansive", "generative", "general"): [
        "create a complete guide on {topic}",
        "write a book outline on {topic}",
        "produce a full curriculum on {topic}",
    ],

    # EXPANSIVE + MUTATIVE + general
    ("expansive", "mutative", "general"): [
        "overhaul the entire documentation",
        "rewrite everything",
        "revamp all content",
    ],
}

# Substitution vocabulary
_FILES = ["auth.py", "main.py", "utils.py", "models.py", "api.py", "db.py", "config.py"]
_FUNCS = ["validate", "authenticate", "process", "handle", "compute", "parse", "fetch"]
_FUNCS2 = ["check", "verify", "run", "execute", "dispatch"]
_MODULES = ["auth", "payment", "user", "session", "api", "db", "cache"]
_CLASSES = ["User", "Session", "Request", "Handler", "Processor"]
_SERVICES = ["redis", "postgres", "nginx", "docker", "kubernetes"]
_ACTIONS = ["converts data", "validates input", "formats output"]
_ENDPOINTS = ["/api/v1/users", "/auth/login", "/health"]
_FEATURES = ["oauth", "caching", "rate-limiting", "pagination"]
_ERRORS = ["TypeError", "AttributeError", "KeyError", "ImportError", "ValueError"]
_TOPICS = ["machine learning", "distributed systems", "security", "performance", "testing"]
_PAPERS = ["Smith et al. 2020", "Jones 2021", "Brown et al. 2019"]
_METRICS = ["latency", "throughput", "error rate", "p99"]
_MODULES2 = ["service_a", "service_b", "legacy", "v2"]
_TERMS = ["idempotency", "sharding", "consensus", "backpressure"]


def _sub(template: str, rng: random.Random) -> str:
    return (template
            .replace("{file}", rng.choice(_FILES))
            .replace("{func}", rng.choice(_FUNCS))
            .replace("{func2}", rng.choice(_FUNCS2))
            .replace("{module}", rng.choice(_MODULES))
            .replace("{module2}", rng.choice(_MODULES2))
            .replace("{class_name}", rng.choice(_CLASSES))
            .replace("{service}", rng.choice(_SERVICES))
            .replace("{action}", rng.choice(_ACTIONS))
            .replace("{endpoint}", rng.choice(_ENDPOINTS))
            .replace("{feature}", rng.choice(_FEATURES))
            .replace("{error}", rng.choice(_ERRORS))
            .replace("{topic}", rng.choice(_TOPICS))
            .replace("{paper}", rng.choice(_PAPERS))
            .replace("{metric}", rng.choice(_METRICS))
            .replace("{term}", rng.choice(_TERMS)))


# Edge case templates (ambiguous / terse / noisy)
_EDGE_CASES = [
    ("focused", "mutative", "coding", "do it"),
    ("focused", "readonly", "support", "?"),
    ("moderate", "generative", "coding", "tests"),
    ("focused", "mutative", "coding", "fixx the bug"),     # typo
    ("focused", "readonly", "support", "halp"),             # informal
    ("focused", "readonly", "general", "what"),
    ("moderate", "readonly", "research", "研究してください"),  # non-English
    ("focused", "generative", "coding", "write"),
    ("expansive", "mutative", "coding", "change everything"),
    ("focused", "mutative", "coding", "fix"),
    ("moderate", "generative", "analysis", "dashboard"),
    ("focused", "readonly", "support", "it's broken"),
    ("moderate", "mutative", "coding", "update"),
    ("focused", "readonly", "coding", "???"),
    ("expansive", "readonly", "research", "tell me about all the papers on AI safety ever published"),
    # mixed signals
    ("focused", "mutative", "coding", "analyze and fix the bug in auth.py"),
    ("moderate", "readonly", "coding", "review the code and add some tests"),
]


def generate(seed: int = 42, target_per_combo: int = 250) -> list[tuple[str, str, str, str]]:
    """
    Returns list of (text, complexity, nature, domain) tuples.

    target_per_combo: number of examples to generate per label combination.
    ~45 combos × 250 = ~11,250 + 300 edge cases.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str, str, str]] = []

    for (complexity, nature, domain), templates in _TEMPLATES.items():
        count = 0
        while count < target_per_combo:
            tpl = rng.choice(templates)
            text = _sub(tpl, rng)
            # light augmentation: random casing, filler prefix
            if rng.random() < 0.1:
                text = text.upper()
            elif rng.random() < 0.1:
                text = text.capitalize()
            if rng.random() < 0.1:
                prefix = rng.choice(["please ", "can you ", "i need you to ", ""])
                text = prefix + text
            out.append((text, complexity, nature, domain))
            count += 1

    # Edge cases
    for complexity, nature, domain, text in _EDGE_CASES:
        for _ in range(15):  # ~15 variants each ≈ 255 edge cases
            out.append((text, complexity, nature, domain))

    rng.shuffle(out)
    return out
