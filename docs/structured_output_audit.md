# Structured Output Audit — LLM Calls in Parlant

> **Status:** audit complete, fixes in progress (see [PR plan](#pr-plan) below).
> **Audited at:** June 2026, against `develop` (verified identical to the audited tree for
> `src/parlant/adapters/nlp/` — except `openrouter_service.py` embedder error-handling — and
> byte-identical for `src/parlant/core/nlp/`).
> **Method:** 43 agents — per-provider research of the latest official provider docs (June 2026),
> per-adapter code audit, per-provider gap analysis, cross-cutting critique. Findings below were
> spot-verified against the source.

## Executive summary

**Parlant does not use native structured output on any default production path.**
Every default path either uses legacy `json_object` mode or pure prompt-scraping
(`jsonfinder` scavenging JSON out of free text), while every major provider now offers
guaranteed schema-valid output via constrained decoding.

Three groups of adapters:

| Group | Adapters | Reality |
|---|---|---|
| Native path exists but is **dead code** | OpenAI, Azure, Snowflake | Gated behind `hints["strict"]` — **no engine call site ever passes `strict=True`**. Default = `json_object` + jsonfinder scraping. |
| **No schema sent to the API at all** | Anthropic, Bedrock, Vertex-Claude, Mistral, Together, DeepSeek, Qwen, ModelScope, GLM, Zhipu, OpenRouter, LiteLLM, Emcie | `json_object` or pure prompt-scraping. |
| Correct (mostly) | Fireworks, Cerebras, Ollama | Schema transmitted unconditionally — but Fireworks/Cerebras have runtime bugs (below), and Gemini uses a fragile tool-forcing workaround. |

Additionally, the audit found **confirmed runtime bugs** (verified by hand, not just by agents):

1. **Fireworks: missing `await`** on `AsyncFireworks().chat.completions.create(...)` — the adapter
   cannot work at all (`fireworks_service.py:131`; the `# type: ignore` comments downstream are
   suppressing exactly this).
2. **Fireworks + Cerebras: `record_llm_metrics()` called without the required `schema_name`
   kwarg → `TypeError`** on every successful generation (`fireworks_service.py:152`,
   `cerebras_service.py:152`).
3. **Dotted-string `getattr` bug**: `getattr(response, "usage.prompt_cache_hit_tokens", 0)` —
   `getattr` does not traverse dots, so cached-token metrics are silently always 0 in
   `deepseek_service.py:174`, `qwen_service.py:275`, `glm_service.py:255`, `litellm_service.py:170`
   (each twice: metric call + UsageInfo extra).
4. **ModelScope overrides `generate` instead of `do_generate`** — bypasses the base-class
   tracing/metrics pipeline entirely (`modelscope_service.py:120`).
5. **Zhipu uses the synchronous `ZhipuAI` client from async code** — blocks the event loop
   (`zhipu_service.py:136`).
6. **`max_tokens` hardcoded** (4096/5000/8192) in anthropic, aws, litellm, qwen, glm — JSON gets
   truncated mid-object on long outputs; in qwen/glm a caller-supplied `max_tokens` hint causes
   `TypeError: got multiple values for keyword argument`.
7. **`common.py` metrics-counter singleton**: module-global `_COUNTERS_INITIALIZED` means the
   first Meter instance wins; all later adapters silently write metrics to the wrong Meter.
8. **AWS `verify_environment` checks `ANTHROPIC_API_KEY`** instead of AWS credentials
   (`aws_service.py:213`).

## Cross-cutting findings

- Dotted-string getattr bug: getattr(response, 'usage.prompt_cache_hit_tokens', 0) appears in deepseek_service.py:172, qwen_service.py:273, litellm_service.py:168, glm_service.py:253, openrouter_service.py:292 (extra field only, not top-level call). Python's getattr does not traverse dots — it looks for a literal attribute named 'usage.prompt_cache_hit_tokens', which never exists. The default 0 is always returned. Correct form is getattr(response.usage, 'prompt_cache_hit_tokens', 0). This silently zeros cached_input_tokens metrics across five adapters.
- Universal absence of ValidationError retry: every single adapter (all 20 SchematicGenerator implementations) logs the error and re-raises without retry. The @policy([retry(...)]) decorator on do_generate retries only network/rate errors (APIConnectionError, RateLimitError, etc.). pydantic.ValidationError and json.JSONDecodeError are excluded from every retry policy. This is a codebase-wide gap, not a per-provider issue — one bad LLM response always crashes the entire batch.
- json_object vs json_schema divergence: adapters split into three groups. Group A (OpenAI, Azure, Snowflake) gated: json_schema path exists but requires hints={'strict':True}, which no engine call site ever passes. Group B (Mistral, Together, DeepSeek, Qwen, ModelScope, GLM, Zhipu, OpenRouter, LiteLLM, Anthropic, AWS): hardcoded json_object or prompt-scraping only. Group C (Fireworks, Cerebras, Ollama): hardcoded json_schema/constrained-decoding unconditionally. No provider routes based on what the engine actually needs — the 'strict' hint is effectively dead code across the whole codebase.
- tiktoken gpt-4o-2024-08-06 proxy tokenizer used for non-OpenAI models across 12 adapters: aws_service.py, together_service.py, fireworks_service.py, ollama_service.py, deepseek_service.py, modelscope_service.py, glm_service.py, qwen_service.py, mistral_service.py, litellm_service.py, cerebras_service.py, openrouter_service.py, vertex_service.py. This silently underestimates token counts for Chinese-language models (GLM, Qwen, Zhipu, ModelScope) and BPE-incompatible models (Mistral, Llama), affecting context-window management and billing.
- normalize_json_output + jsonfinder applied universally even where constrained decoding already guarantees valid JSON: jsonfinder is imported by all 20 adapter files (including Fireworks and Cerebras which use json_schema mode, and Ollama which uses constrained format=). The fallback pipeline is never conditioned on whether the API already enforced schema compliance. It is wasted compute and a latent source of silent data corruption (jsonfinder can return a different JSON object than intended if output contains multiple JSON fragments).
- common.py counter singleton breaks multi-tenant / multi-service setups: _COUNTERS_INITIALIZED is module-global. The first call from any adapter wins and creates counters on that adapter's Meter. All subsequent calls from different adapters (with different Meters) silently write to the first Meter's counters. Every adapter calls record_llm_metrics with self.meter, but once _COUNTERS_INITIALIZED=True that meter is ignored. Token metrics are misattributed from the second adapter onward.
- No engine-level schema embedding in prompts: all 20+ engine batch classes build hand-written JSON examples with string placeholders. No call site invokes model_json_schema() to embed the actual Pydantic schema in the prompt. The LLM never sees field types, constraints, or allowed values — only a natural-language description of the expected format. This makes structured output entirely dependent on the model's ability to infer schema from prose, rather than from a machine-readable specification.
- Dead 'strict' hint declared in six adapters (OpenAI, Azure, DeepSeek, ModelScope, GLM, LiteLLM, Snowflake) but consumed in only three (OpenAI, Azure, Snowflake). In DeepSeek, ModelScope, and GLM the hint is listed in supported_hints but no code path reads hints.get('strict'), making it a false contract. No engine call site passes strict=True in any case, so even the working adapters never activate their native structured-output path in production.

## Core engine wiring (how adapters are driven)

### The `strict` hint

No engine call site passes strict=True. The "strict" hint is declared in supported_hints of OpenAISchematicGenerator, AzureSchematicGenerator, and other adapters, but no code in src/parlant/core/engines/alpha/ (guideline matching, tool calling, message_generator, canned_response_generator) passes {"strict": True} to .generate(). All engine calls pass only {"temperature": <float>} or {"type": <str>}, leaving hints.get("strict", False) as False at every call site. The only adapters that hardcode strict=True unconditionally are Fireworks (fireworks_service.py:139) and Cerebras (cerebras_service.py:120), which embed it inside a response_format json_schema dict — not as a hint but as a protocol-level field sent to those APIs. sdk.py also never passes strict=True. Consequently, in a default OpenAI-backed Parlant run every schematic generation goes through the else branch in OpenAISchematicGenerator._do_generate (openai_service.py:247-309): chat.completions.create with response_format={"type": "json_object"}, followed by JSON parse + model_validate. The beta.chat.completions.parse (native structured outputs) path is dead code for OpenAI unless a caller explicitly passes {"strict": True}.

### Effective default path

json_object mode: chat.completions.create(..., response_format={"type": "json_object"}), then json.loads + pydantic model_validate. See src/parlant/adapters/nlp/openai_service.py:247-309. The .parse() / native structured-output path (openai_service.py:197-245) is only reachable if a caller passes hints={"strict": True}, which no engine module currently does.

### PromptBuilder JSON instructions

Hand-written JSON examples only — no Pydantic schema embedding. Each batch/generator builds a PromptBuilder and adds an "output format" section containing a manually crafted JSON example with placeholder strings (e.g. "<BOOL>", "<Explanation…>"). Key locations: (1) message_generator.py:571-725 — _get_output_format() builds a multi-field JSON template string with inline commentary and embeds it in an "OUTPUT FORMAT" section. (2) guideline_actionable_batch.py:291-327 — _format_of_guideline_check_json_description() calls json.dumps() on a hand-built dict with string placeholders and embeds it as a fenced JSON block. (3) single_tool_batch.py output-format section — similar hand-built JSON block. No call site invokes pydantic's model_json_schema() to embed the Pydantic JSON Schema in the prompt; the schema is only used for model_validate() after the response arrives (json_object path) or as the response_format arg on .parse() (strict path).

### DefaultBaseModel config

DefaultBaseModel (src/parlant/core/common.py:68-76) uses ConfigDict(validate_default=True, model_title_generator=_without_dto_suffix). No extra="forbid" or extra="ignore" is set, so Pydantic defaults to extra="ignore" — extra fields in LLM output are silently dropped. populate_by_name is not set (defaults to False). validate_default=True means fields with defaults are also validated on construction. No use_enum_values, no arbitrary_types_allowed. The model_title_generator strips a "DTO" suffix from class names in the JSON schema title, which is cosmetic only.

### Engine-level retry

No retry-with-feedback loop at the engine level. When an adapter raises ValidationError (pydantic) or JSONDecodeError, the exception propagates unmodified up through BaseSchematicGenerator.generate() (generation.py:278-317) to the caller. There is no catch-and-retry-with-corrected-prompt logic anywhere in src/parlant/core/engines/alpha/. The @policy([retry(...)]) decorator on do_generate in OpenAISchematicGenerator and AzureSchematicGenerator (openai_service.py:150-163, azure_service.py:110-121) retries only on APIConnectionError, APITimeoutError, ConflictError, RateLimitError, APIResponseValidationError, and InternalServerError — not on pydantic.ValidationError or json.JSONDecodeError. Those two error types are logged and re-raised (openai_service.py:305-309, azure_service.py:269-273). FallbackSchematicGenerator (generation.py:320-351) provides a generator-level fallback chain (used by Vertex/Gemini services), but it catches any Exception indiscriminately — it is not wired for OpenAI. In the guideline_actionable_batch.py engine loop (lines 109-113) there is a 3-attempt retry over temperature variations, but that loop does not catch JSON errors; it only retries if the check list is empty. In summary: a bad-JSON or schema-mismatch response will crash the entire batch with no automatic recovery on OpenAI.

### Test coverage

Tests for structured-output adapter behavior are sparse. tests/core/stable/nlp/test_generation.py covers FallbackSchematicGenerator fallback/retry mechanics and policy decorator behavior using mocks — no actual JSON parsing, strict mode, or schema validation is tested. tests/adapters/nlp/test_azure_service.py tests AzureService initialization, authentication, model selection, and that supported_hints includes 'strict' (line 237) — but does not test the strict=True code path end-to-end (no mock of beta.chat.completions.parse). There are no test files for OpenAI adapter structured-output behavior (no tests/adapters/nlp/test_openai_service.py). No tests exercise the json_object fallback parsing (jsonfinder.only_json) or the ValidationError propagation path. tests/adapters/nlp/test_openrouter_service.py, test_litellm_service.py, test_zhipu_service.py, test_qwen_service.py exist but none cover structured-output mode switching.

### Schema complexity notes (relevant for strict json_schema modes)

- SingleToolBatchSchema (single_tool_batch.py:118-124): Deeply nested — SingleToolBatchSchema > list[SingleToolBatchToolCallEvaluation] > Optional[list[SingleToolBatchArgumentEvaluation]]. All top-level fields are Optional[str|bool]=None. SingleToolBatchToolCallEvaluation has ~11 Optional fields (str, bool, list). SingleToolBatchArgumentEvaluation has a custom validator (stringify_values) on args: Optional[dict[str, str | None]]. No field descriptions (no Field(description=...)). The dict[str, str|None] and union types (str|None) are challenging for strict structured-output APIs.
- GenericActionableGuidelineMatchesSchema (guideline_actionable_batch.py:59-61): Shallow — one field checks: Sequence[GenericActionableBatch]. GenericActionableBatch has guideline_id: str, condition: str, rationale: str, applies: bool — all required, no Optional, no unions. Clean for structured outputs.
- MessageSchema (message_generator.py:115-123): Nearly all Optional. Contains Optional[list[Revision]] where Revision itself has ~12 Optional fields including Optional[list[FactualInformationEvaluation]] and Optional[list[OfferedServiceEvaluation]]. No field descriptions. The heavy Optional nesting means JSON schema has many oneOf:[null, T] which strict mode on OpenAI requires all-or-nothing.
- CannedResponseDraftSchema (canned_response_generator.py:107-112): Shallow, mostly Optional. last_message_of_user: Optional[str] (no default — will be required in JSON schema), insights: Optional[list[str]] = None, response_preamble_that_was_already_sent: Optional[str] = None. No field descriptions. FollowUpCannedResponseSelectionSchema has unsatisfied_guidelines: Optional[str | list[str]] — a union of str and list[str] which strict-mode APIs typically reject.
- CannedResponseFieldExtractionSchema (canned_response_generator.py:304-308): Not shown above but is referenced by GenerativeFieldExtraction. Schema definition is minimal (single str field extracted_value).

### Key code refs (core)

- `src/parlant/adapters/nlp/openai_service.py:197-245 (strict=True → beta.chat.completions.parse path)`
- `src/parlant/adapters/nlp/openai_service.py:247-309 (default json_object path, ValidationError re-raise)`
- `src/parlant/adapters/nlp/openai_service.py:119 (supported_hints declaration)`
- `src/parlant/adapters/nlp/fireworks_service.py:134-143 (hardcoded strict:true in json_schema response_format)`
- `src/parlant/adapters/nlp/cerebras_service.py:115-124 (hardcoded strict:true in json_schema response_format)`
- `src/parlant/core/nlp/generation.py:320-351 (FallbackSchematicGenerator — no ValidationError special handling)`
- `src/parlant/core/nlp/generation.py:279-317 (BaseSchematicGenerator.generate — no retry on JSON errors)`
- `src/parlant/core/nlp/policies.py:35-76 (RetryPolicy — only network/rate errors)`
- `src/parlant/core/common.py:68-76 (DefaultBaseModel config — validate_default=True, no extra policy set)`
- `src/parlant/core/engines/alpha/message_generator.py:571-725 (_get_output_format — hand-written JSON example)`
- `src/parlant/core/engines/alpha/message_generator.py:733-736 (generate call — only temperature hint)`
- `src/parlant/core/engines/alpha/guideline_matching/generic/guideline_actionable_batch.py:291-327 (hand-written JSON example, no schema dump)`
- `src/parlant/core/engines/alpha/guideline_matching/generic/guideline_actionable_batch.py:110-112 (generate call — only temperature hint)`
- `src/parlant/core/engines/alpha/tool_calling/single_tool_batch.py:118-124 (SingleToolBatchSchema — deeply nested Optional fields)`
- `src/parlant/core/engines/alpha/canned_response_generator.py:107-128 (CannedResponseDraftSchema, FollowUpCannedResponseSelectionSchema with str|list[str] union)`
- `tests/core/stable/nlp/test_generation.py (FallbackSchematicGenerator and retry policy unit tests)`
- `tests/adapters/nlp/test_azure_service.py:237 (only test touching 'strict' hint — checks list membership only)`

---

# Per-provider deep dive

Each section: how the provider says structured output SHOULD be done (researched against the
latest official docs, June 2026), how this repo actually does it, and the concrete gaps.


## OpenAI — verdict: **suboptimal**

The adapter has working native structured output on the strict=True path (beta.chat.completions.parse with Pydantic model), but it is disabled by default. The non-strict default path uses legacy json_object mode with manual scraping fallbacks — the opposite of what OpenAI recommends. Four additional reliability issues (assert crashes on missing usage fields, no refusal check, deprecated beta namespace, unreachable match/case branches) compound the core routing problem.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** Constrained decoding via Context-Free Grammar (CFG) enforcing schema adherence. Two API surfaces: (1) Responses API uses text.format with type json_schema; (2) Chat Completions API uses response_format with type json_schema. Both require strict:true to activate guaranteed schema adherence.
- **Exact API fields:** Responses API: text={ format={ type='json_schema', name='<schema_name>', schema={...JSON Schema...}, strict=true } }

Chat Completions API: response_format={ type='json_schema', json_schema={ name='<schema_name>', schema={...JSON Schema...}, strict=true } }

Python SDK (Responses API, Pydantic): client.responses.parse(model=..., input=[...], text_format=MyPydanticModel) → response.output_parsed

Python SDK (Chat Completions, Pydantic): client.chat.completions.parse(model=..., messages=[...], response_format=MyPydanticModel) → response.choices[0].message.parsed
- **SDK helpers:** Python: client.responses.parse(text_format=PydanticModel) for Responses API; client.chat.completions.parse(response_format=PydanticModel) for Chat Completions. Both accept Pydantic BaseModel subclasses directly — SDK auto-converts to JSON Schema. Responses API returns response.output_parsed (validated Pydantic instance); Chat Completions returns message.parsed. JavaScript: client.chat.completions.parse() with zodResponseFormat() helper; Responses API uses zodTextFormat().
- **Weaker JSON-mode fallback:** response_format={ type='json_object' } — guarantees valid JSON but NOT schema adherence. Works on older models (gpt-3.5-turbo, gpt-4-*). OpenAI explicitly recommends always preferring Structured Outputs (json_schema strict) over json_object mode when the target model supports it.
- **Deprecated patterns:**
  - beta.chat.completions.parse — was the original beta namespace for the parse helper; has been promoted out of beta to client.chat.completions.parse (stable). The beta.* prefix is no longer required or recommended.
  - json_object mode (response_format={type:'json_object'}) — not formally deprecated but treated as legacy; OpenAI docs say 'always use Structured Outputs instead of JSON mode when possible'.
  - Assistants API — deprecated August 26 2025, sunset August 26 2026; replace with Responses API + Conversations API.
  - functions / function_call parameters in Chat Completions — superseded by tools / tool_choice with json_schema strict schemas.
  - Older chat model snapshots (gpt-3.5-turbo-0125, gpt-4-0613, gpt-4-turbo variants) — deprecated October 23 2026; migrate to gpt-5.4-mini / gpt-5.5.
- **Limitations:**
  - All object properties must be listed in the required array — there are no truly optional keys in strict mode; use type ['T', 'null'] (nullable union) to represent optional values.
  - additionalProperties must be set to false on every object in the schema.
  - anyOf / oneOf unions have limited support; workaround is enum for string variants or nullable types for optional fields.
  - Recursive schemas are supported via $ref / model_rebuild() but depth is bounded; nesting deeper than ~5 levels and schemas exceeding ~100 total properties are constrained.
  - First request with a new schema incurs extra latency (schema compilation); subsequent requests with the identical schema are fast.
  - Safety refusals bypass the schema — check message.refusal (Chat Completions) or response.output[].refusal / incomplete_details.reason (Responses API) before accessing parsed output.
  - Reasoning models (o3, o4-mini, GPT-5): reasoning tokens are internal and not cached across turns; multistep conversations cannot achieve full cache hits because reasoning items from prior turns are stripped before re-submission. Use Responses API with encrypted_content / reasoning item persistence to improve cache utilization (reported ~80% vs ~40% on Chat Completions).
  - reasoning_effort knob should not be combined with explicit chain-of-thought prompting — doing so can degrade output quality on reasoning models.
  - Structured Outputs not available on models older than gpt-4o-2024-08-06.
- **Best practices:**
  - Prefer Responses API over Chat Completions for new development — it is the recommended agentic interface, has better cache utilization with reasoning models, and exposes response.output_parsed directly.
  - Use client.responses.parse(text_format=MyModel) with a Pydantic BaseModel — the SDK handles schema generation, serialization, and deserialization automatically.
  - Set strict=True always; json_object mode is legacy and provides no schema enforcement.
  - Declare all fields in required and use Optional[X] → field: X | None = None in Pydantic to generate nullable unions for optional values.
  - Set additionalProperties=False explicitly on every nested object (Pydantic BaseModel does this automatically via the SDK).
  - Check response.output_parsed is not None and inspect refusal / incomplete_details before using parsed data.
  - Cache schemas across requests (identical schema = no re-compilation latency); do not regenerate schema objects per call.
  - For reasoning models, use the Responses API with reasoning item persistence (encrypted_content) to maximise prompt cache hit rate — cached tokens for o4-mini are 75% cheaper.
  - Do not prompt reasoning models to 'think step-by-step' or reason more extensively — it can hurt structured output quality; set reasoning_effort instead.
  - Validate schemas offline first — schemas exceeding 100 properties or 5 nesting levels may be silently rejected or produce errors at runtime.
- **Sources:** <https://platform.openai.com/docs/guides/structured-outputs>, <https://platform.openai.com/docs/guides/migrate-to-responses>, <https://developers.openai.com/api/docs/guides/structured-outputs>, <https://developers.openai.com/api/docs/guides/reasoning>, <https://developers.openai.com/api/docs/changelog>, <https://platform.openai.com/docs/deprecations>, <https://cookbook.openai.com/examples/responses_api/reasoning_items>, <https://developers.openai.com/cookbook/examples/o-series/o3o4-mini_prompting_guide>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/openai_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** Two branches in _do_generate. When hints["strict"] is True: uses beta.chat.completions.parse() with the Pydantic schema class as response_format — OpenAI SDK handles schema serialization and deserialization natively, returning a fully-validated model object via response.choices[0].message.parsed. When hints["strict"] is False (default): uses chat.completions.create() with response_format={"type": "json_object"}, then manually parses the string content with json.loads(normalize_json_output(...)) and validates with self.schema.model_validate(json_content).
- **Uses native structured output:** yes (beta.chat.completions.parse (strict=True path) with Pydantic model as response_format; response_format={"type": "json_object"} with manual model_validate (strict=False default path))
- **Schema sent to API:** yes
- **Gating:** hints["strict"] — default is False, so native structured output (beta.parse) is NOT used by default; callers must pass strict=True explicitly
- **Parsing fallbacks:** normalize_json_output() strips ```json fences (common.py:19-32); jsonfinder.only_json() scavenges JSON from arbitrary text when json.loads fails after normalize (openai_service.py:270)
- **Retries on ValidationError:** no
- **max_tokens handling:** max_tokens is passed through from hints (listed in supported_openai_params); no hardcoded default — if caller omits it from hints, it is not sent to the API. The max_tokens property on each model class sets context window size for the tokenizer/estimator, not the API call.
- **Notable issues:**
  - ValidationError (pydantic) is caught, logged, and immediately re-raised — no retry (openai_service.py:305-309). The @policy retry decorator only covers transport/rate-limit errors, not schema validation failures.
  - Single-role prompt architecture: all prompts are sent as a single message with role='developer'. No system/user role separation.
  - In the strict=True path, response.usage.prompt_tokens_details is assert-checked (openai_service.py:219) — if the API omits prompt_tokens_details, this crashes with AssertionError rather than a graceful fallback. Same issue in the non-strict path (openai_service.py:276-277).
  - get_schematic_generator() can fall through all match/case branches without returning a value in the NANO and MINI cases if model_generation is an unrecognized string — implicit None return that MyPy should flag but may not catch at runtime.
  - GPT_4o_Mini, GPT_4_1_Mini, GPT_4_1_Nano, GPT_5_1, GPT_5_Mini, GPT_5_Nano each redundantly create a self._token_estimator attribute but the base class already assigns self._tokenizer from the same constructor arg — the field is never read (dead assignment).
  - SingleToolBatchSchema is still routed to GPT_4o (gpt-4o-2024-11-20) in AUTO mode while most other schemas use GPT_4_1, suggesting a leftover routing that was not updated.
  - common.py module-level Counter globals are initialized lazily with a mutable global flag (_COUNTERS_INITIALIZED) — not thread/async safe; a race on first call could initialize counters twice.
- **Key code refs:**
  - `openai_service.py:197 — hints.get('strict', False) branch gate`
  - `openai_service.py:200 — beta.chat.completions.parse() call (strict path)`
  - `openai_service.py:204 — response_format=self.schema passed to parse()`
  - `openai_service.py:215 — response.choices[0].message.parsed extraction`
  - `openai_service.py:250 — chat.completions.create() with json_object response_format (non-strict path)`
  - `openai_service.py:267 — json.loads(normalize_json_output(raw_content))`
  - `openai_service.py:270 — jsonfinder.only_json fallback on JSONDecodeError`
  - `openai_service.py:274 — self.schema.model_validate(json_content)`
  - `openai_service.py:305-309 — ValidationError caught, logged, re-raised (no retry)`
  - `openai_service.py:150-162 — @policy retry decorator covering transport/rate-limit exceptions only`
  - `openai_service.py:118-119 — supported_hints list including 'strict'`
  - `openai_service.py:173-185 — _list_arguments() filtering hints to supported OpenAI params`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] Native structured output (constrained decoding) is opt-in, not the default

`src/parlant/adapters/nlp/openai_service.py:197`

hints.get('strict', False) means every call that does not explicitly pass strict=True falls through to json_object mode — a legacy path with no schema enforcement, manual JSON scraping, and possible ValidationError. OpenAI's docs say 'always use Structured Outputs instead of JSON mode when possible'. All models in the adapter (gpt-4o-2024-08-06 and newer) support strict Structured Outputs. The default should be the safe path.

**Recommendation:** Flip the default: change hints.get('strict', False) to hints.get('strict', True) at openai_service.py:197. Alternatively, remove the branch entirely and make client.chat.completions.parse(response_format=self.schema) the sole code path, keeping the json_object path only as an explicit opt-out for legacy model_names older than gpt-4o-2024-08-06.

#### [MEDIUM] beta.chat.completions.parse is the deprecated beta-namespace surface

`src/parlant/adapters/nlp/openai_service.py:200`

The strict=True path calls self._client.beta.chat.completions.parse(). OpenAI promoted this helper out of beta — the stable path is self._client.chat.completions.parse(). The beta.* prefix is no longer required or recommended and may be removed in a future SDK release.

**Recommendation:** Replace self._client.beta.chat.completions.parse(...) with self._client.chat.completions.parse(...) at openai_service.py:200. The parameters and return shape are identical.

#### [HIGH] No refusal check before accessing parsed output

`src/parlant/adapters/nlp/openai_service.py:215`

After client.chat.completions.parse() returns, the code does assert parsed_object (line 216) but never inspects response.choices[0].message.refusal. When the model issues a safety refusal the parsed field is None and refusal holds the reason string — the bare assert raises AssertionError with no diagnostic. This masks refusal events as crashes rather than surfacing them as a distinct, handleable condition.

**Recommendation:** After line 215, add: refusal = response.choices[0].message.refusal; if refusal: raise RuntimeError(f'OpenAI refused to generate structured output: {refusal}'). Only then assert/use parsed_object.

#### [HIGH] assert on prompt_tokens_details crashes when field is absent

`src/parlant/adapters/nlp/openai_service.py:219`

Lines 219 and 276-277 use assert response.usage.prompt_tokens_details. This field is None when the API omits it (e.g. on some older model responses or during outages). An AssertionError here kills the request rather than gracefully falling back to cached_tokens=0. The same fragility appears in both the strict and non-strict code paths.

**Recommendation:** Replace the assert with a safe access: cached = (response.usage.prompt_tokens_details.cached_tokens or 0) if response.usage.prompt_tokens_details else 0. Remove the assert lines entirely (openai_service.py:219 and 276-277) and use the guarded expression wherever cached_tokens is referenced.

#### [MEDIUM] json_object fallback path retains manual scraping that the API guarantees away

`src/parlant/adapters/nlp/openai_service.py:250`

The non-strict path uses response_format={'type': 'json_object'} plus normalize_json_output() (fence stripping) and jsonfinder.only_json() (content scavenging). json_object mode guarantees valid JSON — fence-wrapping and mid-text JSON cannot appear. These scraping layers add dead-code complexity and indicate the path was designed around older, non-constrained behaviour. More importantly, json_object mode provides no schema adherence, so model_validate can still raise ValidationError with no recovery path.

**Recommendation:** Remove the non-strict path entirely and route all calls through client.chat.completions.parse(response_format=self.schema). If a true legacy fallback must be kept, at minimum drop the normalize_json_output and jsonfinder calls — they are not needed with json_object mode. The ValidationError catch at line 305-309 should also either retry (see next gap) or be removed once the strict path is the default.

#### [MEDIUM] No retry on ValidationError / schema-validation failure

`src/parlant/adapters/nlp/openai_service.py:305`

The @policy retry decorator at line 150-162 covers transport exceptions (APIConnectionError, RateLimitError, etc.) but not pydantic.ValidationError. On the non-strict path a ValidationError is caught, logged, and immediately re-raised (lines 305-309) with no retry. Because json_object mode does not constrain the schema, transient model variance can produce a parseable-but-invalid payload — a single retry often succeeds.

**Recommendation:** Either (a) eliminate the non-strict path so ValidationError cannot occur from schema mismatch, or (b) add pydantic.ValidationError to the retry exception list in the @policy decorator so it gets the same retry treatment as transport errors. Option (a) is preferred.

#### [MEDIUM] get_schematic_generator can fall through NANO/MINI match arms and return None

`src/parlant/adapters/nlp/openai_service.py:767`

In the ModelSize.NANO and ModelSize.MINI cases (lines 766-793), the inner match on model_generation has an explicit 'auto'|'stable' arm and a 'latest' arm but no wildcard _. If model_generation is any other string (e.g. 'experimental'), both inner matches fall through without hitting a return statement. Python returns None implicitly, which violates the declared return type OpenAISchematicGenerator[T] and will crash at the call site.

**Recommendation:** Add a default arm to both inner match blocks: case _: return GPT_4_1_Nano[t](...) (for NANO) and GPT_4_1_Mini[t](...) (for MINI), consistent with the 'auto'|'stable' behaviour. This closes the implicit-None gap that MyPy strict mode may miss at runtime.

#### [LOW] SingleToolBatchSchema still routed to deprecated gpt-4o-2024-11-20 in AUTO mode

`src/parlant/adapters/nlp/openai_service.py:755`

The AUTO routing table at line 754-765 maps SingleToolBatchSchema to GPT_4o (gpt-4o-2024-11-20) while every other schema uses GPT_4_1 or GPT_4_1_Mini. gpt-4o-2024-11-20 is an older snapshot targeted for deprecation October 2026. This is likely a missed update when GPT_4_1 was introduced.

**Recommendation:** Change the SingleToolBatchSchema entry in the AUTO dict from GPT_4o[SingleToolBatchSchema] to GPT_4_1[SingleToolBatchSchema] to align with the rest of the routing table.

#### [LOW] Dead _token_estimator assignments on model subclasses

`src/parlant/adapters/nlp/openai_service.py:351`

GPT_4o_Mini, GPT_4_1_Mini, GPT_4_1_Nano, GPT_5_1, GPT_5_Mini, and GPT_5_Nano each set self._token_estimator = OpenAIEstimatingTokenizer(...) in __init__ (e.g. line 351). The base class OpenAISchematicGenerator already creates self._tokenizer from the same constructor arg at line 136-138. self._token_estimator is never read anywhere; it is a dead assignment that also instantiates a redundant tokenizer object per generator instance.

**Recommendation:** Remove the self._token_estimator = ... lines from all six subclass __init__ methods. No behaviour changes; reduces per-instance memory and eliminates confusion about which attribute is authoritative.


## Azure OpenAI — verdict: **suboptimal**

The strict path (beta.chat.completions.parse with Pydantic class) correctly uses native structured output. However, it is gated behind a non-default hint, so the default code path uses json_object mode (no schema enforcement) with brittle post-processing fallbacks. Additional gaps: no refusal check before accessing .parsed, metrics never recorded on the default path, deprecated api-version default, no retry on ValidationError, and a synchronous DefaultAzureCredential call in verify_environment using deprecated asyncio.get_event_loop().

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** response_format with type=json_schema and strict=true inside the json_schema object; SDK parse helper via client.beta.chat.completions.parse() accepts a Pydantic BaseModel directly as response_format
- **Exact API fields:** response_format={"type": "json_schema", "json_schema": {"name": "<schema_name>", "strict": true, "schema": {"type": "object", "properties": {...}, "required": [...], "additionalProperties": false}}}
- **SDK helpers:** Python openai SDK (>=1.42.0) with pydantic >=2.8.2: client.beta.chat.completions.parse(model=DEPLOYMENT, messages=[...], response_format=MyPydanticModel). Access result via completion.choices[0].message.parsed. Check completion.choices[0].message.refusal for refusals. For async: AsyncAzureOpenAI (or standard OpenAI client with v1 base_url) with the same beta.chat.completions.parse interface.
- **Weaker JSON-mode fallback:** response_format={\"type\": \"json_object\"} — guarantees valid JSON but NOT schema adherence. Requires the word \"JSON\" to appear in the system/user messages or the request returns a 400 error.
- **Deprecated patterns:**
  - json_object mode (JSON mode) is not formally deprecated but is superseded by json_schema strict mode for any use case requiring schema conformance
  - Dated api-version query parameters (e.g. api-version=2024-08-01-preview) are superseded by the v1 GA API (base_url ending in /openai/v1/) launched August 2025, which eliminates the need for monthly api-version bumps
- **Limitations:**
  - Maximum 100 object properties total across the entire schema
  - Maximum 5 nesting levels
  - All fields must be marked required; use ["string", "null"] union types for optional fields — no true optional fields
  - additionalProperties must be false on every object in the schema
  - Unsupported JSON Schema keywords: minLength, maxLength, pattern, minimum, maximum, minItems, maxItems, uniqueItems, exclusiveMinimum, exclusiveMaximum
  - anyOf only supported for nested schemas; root-level objects cannot use anyOf
  - Not supported with: parallel function calls (set parallel_tool_calls=false), bring-your-own-data (BYOD) deployments, Assistants API, Foundry Agents Service, audio preview models
  - Recursive schemas are supported via $ref definitions
- **Best practices:**
  - Use the v1 GA API endpoint (https://YOUR-RESOURCE.openai.azure.com/openai/v1/) instead of dated api-version strings to stay current automatically
  - Prefer client.beta.chat.completions.parse() with a Pydantic BaseModel over manually constructing json_schema dicts — the SDK auto-generates the schema and handles parsing/validation
  - Always check completion.choices[0].message.refusal before accessing .parsed — the model may refuse to generate for policy reasons
  - Use Microsoft Entra ID (DefaultAzureCredential + get_bearer_token_provider) over API keys for production deployments
  - When using structured outputs with function/tool calling, set parallel_tool_calls=False to avoid unsupported behavior
  - Models that support strict structured outputs: gpt-4o (2024-08-06+), gpt-4o-mini (2024-07-18+), o1 (2024-12-17+), gpt-4.1, and all gpt-5 family models; verify your deployed model version supports the feature
  - For the v1 API with Entra ID, initialize with standard openai.OpenAI (not AzureOpenAI) pointing base_url at the v1 endpoint, passing token_provider as api_key
- **Sources:** <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs>, <https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs>, <https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle>, <https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode>, <https://learn.microsoft.com/en-us/azure/developer/ai/how-to/extract-entities-using-structured-outputs>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/azure_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** Two paths in _do_generate based on hints['strict']. Non-strict (default): calls client.chat.completions.create with response_format={"type":"json_object"}, parses raw text via normalize_json_output + json.loads, falls back to jsonfinder.only_json, then validates with self.schema.model_validate. Strict path: calls client.beta.chat.completions.parse with response_format=self.schema (Pydantic class), reads response.choices[0].message.parsed directly.
- **Uses native structured output:** yes (Non-strict: response_format json_object (JSON mode, no schema enforcement). Strict: beta.chat.completions.parse with Pydantic class as response_format — OpenAI structured outputs via the beta parse endpoint.)
- **Schema sent to API:** yes
- **Gating:** hints['strict'] — default is False (non-strict json_object mode is the default path)
- **Parsing fallbacks:** normalize_json_output strips ```json fences (azure_service.py:238); jsonfinder.only_json scavenging on JSONDecodeError (azure_service.py:241)
- **Retries on ValidationError:** no
- **max_tokens handling:** max_tokens is passed through hints via _list_arguments (supported_azure_params includes 'max_tokens'), so it is caller-supplied. The model classes define a max_tokens property (GPT_4o: 128*1024, GPT_4o_Mini: 128*1024, CustomAzureSchematicGenerator: int(os.environ.get('AZURE_GENERATIVE_MODEL_WINDOW', 4096))) but this property is NOT wired into the API call — it is a declared capacity, not an enforced limit sent to the API unless the caller includes max_tokens in hints.
- **Notable issues:**
  - Only a single user role message is sent — no system role — the entire prompt is stuffed into role='user' (azure_service.py:145, 211). This prevents system-level instruction separation.
  - record_llm_metrics is called only in the strict path (azure_service.py:174-183); the non-strict path does NOT call it, so token metrics are never recorded for the default code path.
  - ValidationError in the non-strict path logs and re-raises without retry (azure_service.py:269-273); there is no retry on schema mismatch.
  - The max_tokens property on model classes is declared but never automatically injected into API calls — callers must supply it via hints manually.
  - api_version defaults to '2024-08-01-preview' (azure_service.py:285, 315) — a preview version string that may be outdated.
  - GPT_4o_Mini constructor creates a redundant AzureEstimatingTokenizer assigned to self._token_estimator (azure_service.py:376) — the parent already creates one as self._tokenizer. Dead field.
  - unsupported_params_by_model only excludes 'temperature' for 'gpt-5' (azure_service.py:71-73) — likely incomplete as other newer models also drop temperature.
  - AzureService.verify_environment uses synchronous DefaultAzureCredential (not .aio) and asyncio.get_event_loop() which is deprecated in Python 3.10+ (azure_service.py:562-588).
- **Key code refs:**
  - `azure_service.py:141 — hints.get('strict', False) gate`
  - `azure_service.py:144 — client.beta.chat.completions.parse (strict path)`
  - `azure_service.py:147 — response_format=self.schema (Pydantic class sent to API)`
  - `azure_service.py:210 — client.chat.completions.create (non-strict path)`
  - `azure_service.py:213 — response_format={'type':'json_object'} (no schema)`
  - `azure_service.py:238 — normalize_json_output call`
  - `azure_service.py:241 — jsonfinder.only_json fallback`
  - `azure_service.py:245 — self.schema.model_validate(json_content)`
  - `azure_service.py:269 — ValidationError caught, logged, re-raised (no retry)`
  - `azure_service.py:110-121 — @policy retry decorator (APIConnectionError, APITimeoutError, RateLimitError, APIResponseValidationError, InternalServerError)`
  - `azure_service.py:174 — record_llm_metrics only in strict path`
  - `azure_service.py:285 — api_version default '2024-08-01-preview'`
  - `azure_service.py:344-345 — CustomAzureSchematicGenerator.max_tokens from env with hardcoded fallback 4096`
  - `azure_service.py:376 — dead self._token_estimator in GPT_4o_Mini`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] Structured output is opt-in via non-default hint — json_object mode is the default

`src/parlant/adapters/nlp/azure_service.py:141`

hints.get('strict', False) at azure_service.py:141 means every caller that does not explicitly pass strict=True gets response_format={'type':'json_object'} (line 213), which guarantees valid JSON but provides zero schema enforcement. The model can (and does) return JSON that does not match the Pydantic schema, causing ValidationError at runtime with no recovery path. The native constrained-decoding path exists but is almost never exercised.

**Recommendation:** Flip the default: use beta.chat.completions.parse with response_format=self.schema for all calls. If a caller genuinely needs the looser json_object mode, let them opt OUT with hints.get('strict', True). Alternatively, rename the hint to 'json_object_only' and default it False. The call signature to use is: await self._client.beta.chat.completions.parse(messages=[{"role": "user", "content": prompt}], model=self.model_name, response_format=self.schema, **azure_api_arguments)

#### [HIGH] No refusal check before accessing .parsed on the strict path

`src/parlant/adapters/nlp/azure_service.py:169`

Line 169 does `parsed_object = response.choices[0].message.parsed` and immediately asserts it is truthy (line 170). Per the Azure OpenAI docs, when the model refuses to generate for policy reasons, .parsed is None and .refusal contains the refusal string. The bare assert raises AssertionError with no diagnostic context, masking the real failure.

**Recommendation:** Add an explicit refusal check before the assert:
  msg = response.choices[0].message
  if msg.refusal:
      raise RuntimeError(f"Model refused to generate structured output: {msg.refusal}")
  parsed_object = msg.parsed
  assert parsed_object  # schema mismatch guard

#### [HIGH] record_llm_metrics never called on the default (non-strict) code path

`src/parlant/adapters/nlp/azure_service.py:174`

record_llm_metrics is called only inside the strict branch (lines 174-183). The non-strict path (lines 206-273) builds a GenerationInfo with token counts but never calls record_llm_metrics, so input/output/cached token counters are permanently zero for the default execution path. This silently under-reports all token consumption.

**Recommendation:** Add the same record_llm_metrics call after line 247 (assert response.usage) in the non-strict branch:
  await record_llm_metrics(
      self.meter,
      self.model_name,
      schema_name=self.schema.__name__,
      input_tokens=response.usage.prompt_tokens,
      output_tokens=response.usage.completion_tokens,
      cached_input_tokens=response.usage.prompt_tokens_details.cached_tokens or 0
          if response.usage.prompt_tokens_details else 0,
  )

#### [MEDIUM] Deprecated api-version default ('2024-08-01-preview')

`src/parlant/adapters/nlp/azure_service.py:285`

create_azure_client() defaults api_version to '2024-08-01-preview' (lines 285, 315). The Azure OpenAI v1 GA API launched August 2025 (base_url ending in /openai/v1/) eliminates monthly api-version strings entirely. Staying on the preview version string risks feature gaps and eventual breaking changes as Microsoft retires old api-version paths.

**Recommendation:** Bump the default to the latest stable GA string (currently '2025-04-01-preview' or the latest GA). Better yet, for new deployments document (or auto-detect) use of the v1 endpoint pattern: AsyncAzureOpenAI(base_url='https://<resource>.openai.azure.com/openai/v1/', api_key=...) which needs no api_version at all. At minimum change the env-var fallback:
  api_version=os.environ.get('AZURE_API_VERSION', '2025-01-01-preview')

#### [MEDIUM] No retry on ValidationError in the non-strict path

`src/parlant/adapters/nlp/azure_service.py:269`

When self.schema.model_validate(json_content) raises ValidationError (line 245), the error is logged and immediately re-raised (line 269-273). The @policy retry decorator (lines 110-121) only covers network/API errors (APIConnectionError, APITimeoutError, RateLimitError, APIResponseValidationError, InternalServerError) — it does not cover Pydantic ValidationError. Because json_object mode provides no schema guarantee, schema mismatches are real and recoverable by re-prompting.

**Recommendation:** Either (a) move to the strict path by default (see gap 1, which eliminates this entirely), or (b) add ValidationError to the retry decorator exceptions list and include it in the @policy block:
  retry(exceptions=(APIConnectionError, APITimeoutError, RateLimitError, APIResponseValidationError, InternalServerError, ValidationError))
Option (a) is strongly preferred — constrained decoding makes validation retries unnecessary.

#### [MEDIUM] Prompt-scraping fallbacks (normalize_json_output + jsonfinder) doing work the API already guarantees

`src/parlant/adapters/nlp/azure_service.py:238`

Lines 238-242 strip ```json fences and scavenge JSON from free-form text. This is only necessary because json_object mode does not prevent the model from wrapping output in markdown fences or adding prose. The strict parse path (beta.chat.completions.parse) returns a fully validated Python object — no text scraping needed. Maintaining these fallbacks adds fragility and dead code paths if the default switches to strict.

**Recommendation:** If the default is changed to beta.chat.completions.parse (gap 1), delete normalize_json_output and jsonfinder usage from _do_generate entirely. If json_object mode is retained as a fallback, document clearly why these scraping steps exist and add a test that exercises them.

#### [MEDIUM] verify_environment uses synchronous DefaultAzureCredential and deprecated asyncio.get_event_loop()

`src/parlant/adapters/nlp/azure_service.py:562`

Lines 561-588 import the synchronous azure.identity.DefaultAzureCredential (not the .aio variant), call credential.get_token() synchronously inside an async function, and use asyncio.get_event_loop() which is deprecated since Python 3.10 (raises DeprecationWarning and will eventually raise RuntimeError in future Python versions). The rest of the module correctly uses azure.identity.aio.DefaultAzureCredential (line 25).

**Recommendation:** Replace the synchronous credential with azure.identity.aio.DefaultAzureCredential and use asyncio.run() or simply await the token check directly. Remove the asyncio.get_event_loop() call. If verify_environment must be synchronous, use the synchronous azure.identity.DefaultAzureCredential (already imported at line 561) and call credential.get_token() synchronously without wrapping in an async function.

#### [LOW] Dead field: GPT_4o_Mini creates a redundant AzureEstimatingTokenizer as self._token_estimator

`src/parlant/adapters/nlp/azure_service.py:376`

Line 376 assigns self._token_estimator = AzureEstimatingTokenizer(model_name=self.model_name). The parent AzureSchematicGenerator.__init__ already creates self._tokenizer (line 86) with the same arguments and exposes it via the tokenizer property (line 93). self._token_estimator is never read anywhere — it is an unreferenced duplicate that wastes a tiktoken encoding load on every GPT_4o_Mini construction.

**Recommendation:** Delete line 376. The tokenizer property already provides access to the correct tokenizer via self._tokenizer.

#### [LOW] unsupported_params_by_model exclusion list is likely incomplete for newer models

`src/parlant/adapters/nlp/azure_service.py:71`

Lines 71-73 only exclude 'temperature' for model names starting with 'gpt-5'. Azure's newer reasoning models (o1, o3, o4-mini) and gpt-4.1 also do not support temperature. Any CustomAzureSchematicGenerator deployment targeting these models would silently pass temperature to the API, which returns a 400 error.

**Recommendation:** Expand the exclusion map:
  unsupported_params_by_model: dict[str, list[str]] = {
      "gpt-5": ["temperature"],
      "o1": ["temperature"],
      "o3": ["temperature"],
      "o4": ["temperature"],
      "gpt-4.1": [],  # temperature supported, but verify
  }
Consider reading the deployment model metadata at startup to detect unsupported parameters dynamically.


## Anthropic Claude — verdict: **suboptimal**

The Anthropic adapter uses prompt-only JSON scraping with no schema transmitted to the API. Native constrained decoding (GA since January 2026) is entirely unused. Seven concrete gaps span correctness, robustness, and cost — ranging from JSON that can silently fail validation to prompt-cache metrics that are always zero.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** Grammar-constrained sampling (constrained decoding) via output_config.format with type json_schema. Compiles JSON schema into a grammar that restricts token generation during inference, guaranteeing schema-valid output. Also available as strict tool use (strict: true on tool definitions) for constraining tool input parameters.
- **Exact API fields:** GA (no beta header needed): output_config={"format": {"type": "json_schema", "schema": {"type": "object", "properties": {...}, "required": [...], "additionalProperties": false}}}. For strict tool use: tools=[{"name": "...", "description": "...", "input_schema": {...}, "strict": true}]. Python SDK helper: client.messages.parse(model=..., max_tokens=..., output_format=MyPydanticModel, messages=[...]); the SDK translates output_format to output_config.format internally and returns response.parsed_output.
- **SDK helpers:** Python: client.messages.parse() accepts a Pydantic BaseModel class as output_format parameter; returns .parsed_output with typed result. Also: transform_schema() helper for manual schema customization. TypeScript: zodOutputFormat() for Zod schemas, jsonSchemaOutputFormat() for raw JSON Schema literals (requires as const); both used with client.messages.parse(). Ruby, PHP, Java support native type definitions. C#, Go, CLI use raw JSON schemas.
- **Weaker JSON-mode fallback:** Legacy approach: prompt engineering with system prompt instructions to return JSON, combined with assistant prefill of { to bias output. This is not guaranteed and has been superseded. Still works but not recommended. For older models not supporting structured outputs, tool use with a single tool definition (without strict: true) can coerce JSON-shaped output but without grammar-constrained guarantees.
- **Deprecated patterns:**
  - output_format parameter (top-level) — deprecated, moved to output_config.format at GA (January 2026); still accepted during transition period
  - Beta header anthropic-beta: structured-outputs-2025-11-13 — no longer required after GA launch (January 29, 2026); still accepted during transition
  - Prompt engineering / prefill { technique for JSON coercion — superseded by native structured outputs with constrained decoding
  - Tool-forcing (tool_choice: {type: tool, name: ...}) as a schema enforcement workaround — superseded by strict: true on tool definitions
- **Limitations:**
  - Recursive schemas not supported
  - External $ref not supported
  - Numerical constraints (minimum, maximum) not supported
  - String constraints (minLength, maxLength) not supported
  - Array constraints beyond minItems 0 or 1 not supported
  - Backreference and lookahead regex patterns not supported
  - Complex enum types not supported
  - Maximum 20 strict tools per request
  - Maximum 24 optional parameters total per request
  - Maximum 16 union-type parameters per request
  - 180-second schema compilation timeout
  - Changing output_config.format invalidates prompt cache for that conversation thread
  - Do not include PHI in tool schemas — schemas are cached for up to 24 hours server-side
  - On refusal or max_tokens reached, output may not match schema; always check stop_reason
  - Structured outputs remain in public beta on Amazon Bedrock and Microsoft Foundry (GA only on direct Claude API as of Feb 2026)
  - Supported models: Claude Opus 4.8, Mythos Preview, Opus 4.7, Opus 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5, Haiku 4.5; availability varies by platform
- **Best practices:**
  - Use client.messages.parse() with a Pydantic model as output_format for cleanest Python integration — SDK handles schema translation and response parsing automatically
  - Use output_config.format (not the deprecated top-level output_format) for raw API calls
  - Use strict: true on tool definitions (strict tool use) to enforce schema on tool inputs — complementary to JSON output mode, not a replacement
  - Always check stop_reason before trusting parsed_output — refusals or max_tokens truncation can yield non-conforming output
  - Set additionalProperties: false in schemas to avoid unexpected fields
  - Schema is cached server-side for 24 hours — avoid changing schemas unnecessarily to benefit from compilation cache
  - Do not put PHI in tool input_schema definitions since they are cached; PHI belongs in message content only
  - For JSON output mode and strict tool use together: JSON output controls response format; strict tool use controls function call parameters — use both independently as needed
  - No beta header required for GA usage on direct Claude API
- **Sources:** <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>, <https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use>, <https://platform.claude.com/docs/en/release-notes/overview>, <https://claude.com/blog/structured-outputs-on-the-claude-developer-platform>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/anthropic_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** _do_generate builds a single-turn user message request via AsyncAnthropic().messages.create(). The prompt (string or PromptBuilder) becomes the sole `user` message. Only hints that match `supported_hints = ["temperature"]` are forwarded to the API. max_tokens is hardcoded to 4096. The model is expected to return JSON in its text response; the raw text is then parsed client-side via normalize_json_output + jsonfinder.only_json.
- **Uses native structured output:** no (none (prompt-only) — no response_format, no tool-forcing, no json_schema parameter is sent. The schema is only used client-side for Pydantic validation after scraping JSON from the text response.)
- **Schema sent to API:** no
- **Gating:** No gating at all. There is no "strict" hint or any conditional path that enables native structured output. supported_hints = ["temperature"] only.
- **Parsing fallbacks:** normalize_json_output strips ```json fences (common.py:19-32); jsonfinder.only_json scavenges the first valid JSON object from the stripped text (anthropic_service.py:168)
- **Retries on ValidationError:** no
- **max_tokens handling:** Hardcoded to 4096 at the messages.create() call site (anthropic_service.py:140). The `max_tokens` property on each model subclass (200*1024) is defined but is NOT used in do_generate — it appears to be dead/unused for the actual API call.
- **Notable issues:**
  - max_tokens hardcoded to 4096 in messages.create() (line 140) while each model class advertises max_tokens=200*1024 via the property — the property is never read during generation, so it's dead code
  - No system message role used — the entire prompt is sent as a single `user` message (line 138-139); Anthropic best practice uses a system prompt for instructions
  - No schema or JSON instruction is added to the prompt automatically — the caller must embed JSON formatting instructions in the prompt string
  - cached_input_tokens not tracked for Anthropic calls: record_llm_metrics is called with default cached_input_tokens=0 (line 178) even though response.usage may carry cache_creation_input_tokens / cache_read_input_tokens for prompt-caching users
  - ValidationError on pydantic schema mismatch logs an error and immediately re-raises — no retry (line 198-202)
  - Claude_Sonnet_3_5 (claude-3-5-sonnet-20241022) is defined but never returned by get_schematic_generator — dead class
  - AnthropicEstimatingTokenizer.estimate_token_count sends the prompt as `assistant` role (line 73), which is semantically wrong for estimating user-turn input tokens
- **Key code refs:**
  - `anthropic_service.py:79 — AnthropicAISchematicGenerator class definition, supported_hints=["temperature"]`
  - `anthropic_service.py:103-115 — @policy retry decorator: retries on APIConnectionError, APITimeoutError, RateLimitError, APIResponseValidationError; InternalServerError up to 2 times with waits (1.0, 5.0)`
  - `anthropic_service.py:137-142 — messages.create() call: single user message, model, max_tokens=4096 hardcoded`
  - `anthropic_service.py:164 — raw_content = response.content[0].text`
  - `anthropic_service.py:167-168 — normalize_json_output then jsonfinder.only_json parsing`
  - `anthropic_service.py:176 — self.schema.model_validate(json_object) Pydantic validation`
  - `anthropic_service.py:178-184 — record_llm_metrics called with cached_input_tokens=0 (not tracked)`
  - `anthropic_service.py:198-202 — ValidationError: logs error and re-raises, no retry`
  - `anthropic_service.py:205-217 — Claude_Sonnet_3_5 (claude-3-5-sonnet-20241022), max_tokens property=200*1024, never selected by get_schematic_generator`
  - `anthropic_service.py:220-232 — Claude_Sonnet_4 (claude-sonnet-4-20250514), default model`
  - `anthropic_service.py:235-247 — Claude_Opus_4_1 (claude-opus-4-1-20250805), used for 3 specific schemas`
  - `anthropic_service.py:285-291 — get_schematic_generator: Opus for JourneyBacktrackNodeSelectionSchema, DisambiguationGuidelineMatchesSchema, CannedResponseSelectionSchema; Sonnet 4 for everything else`
  - `anthropic_service.py:69-75 — AnthropicEstimatingTokenizer: sends prompt as assistant role (semantically wrong)`
  - `common.py:19-32 — normalize_json_output strips ```json fences`
  - `common.py:41-83 — record_llm_metrics increments input/output/cached token counters`

### Gaps

#### [HIGH] Native structured output not used — schema never sent to API

`anthropic_service.py:137`

messages.create() at line 140 sends no output_config parameter. The Pydantic schema (self.schema) is available but is only used client-side for validation after scraping text. Grammar-constrained decoding (GA January 2026) guarantees schema-valid tokens; the current approach can produce malformed or mis-shaped JSON that passes jsonfinder but fails Pydantic validation with no recovery path.

**Recommendation:** Replace the messages.create() call with client.messages.parse() and pass the Pydantic model directly: `response = await self._client.messages.parse(model=self.model_name, max_tokens=..., output_format=self.schema, messages=[{"role": "user", "content": prompt}], **anthropic_api_arguments)`. Then use `response.parsed_output` (already a typed instance of self.schema) instead of the normalize_json_output + jsonfinder + model_validate chain. No beta header is required.

#### [HIGH] max_tokens hardcoded to 4096 — model property never read, truncation risk

`anthropic_service.py:140`

messages.create() passes max_tokens=4096 unconditionally (line 140). Each model subclass defines `max_tokens = 200 * 1024` (lines 216, 231, 246) but BaseSchematicGenerator.generate() never reads that property before delegating to do_generate(), so it is dead code. Structured JSON output for large schemas (e.g. DisambiguationGuidelineMatchesSchema with many guidelines) can exceed 4096 output tokens, causing mid-stream truncation; stop_reason will be 'max_tokens', not 'end_turn', and the truncated JSON will silently fail parsing or validation.

**Recommendation:** Use `self.max_tokens` (the property already defined on each subclass) as the max_tokens argument: `max_tokens=self.max_tokens`. If 200*1024 is too large for typical calls, cap it at a reasonable value (e.g. 16384) but read from the property so it stays aligned with the model class definition. Also add a stop_reason check after the call: `if response.stop_reason != 'end_turn': raise RuntimeError(f'Generation stopped with reason: {response.stop_reason}')`.

#### [HIGH] Prompt-scraping fallbacks doing work the API now guarantees

`anthropic_service.py:167`

normalize_json_output (common.py:19-32) strips markdown fences, and jsonfinder.only_json (line 168) scavenges the first valid JSON object from free text. Both exist solely to compensate for the model returning prose around the JSON. With output_config.format / messages.parse(), the API constrains token generation so the response IS the JSON object — no fences, no prose. The fallback chain adds latency and a silent failure mode: jsonfinder can extract a partial or wrong JSON object that then passes Pydantic with structurally-valid but semantically wrong data.

**Recommendation:** After migrating to client.messages.parse(output_format=self.schema), remove the normalize_json_output and jsonfinder.only_json calls entirely and use response.parsed_output directly. Keep normalize_json_output only if it is shared with other adapters that still use prompt-only mode.

#### [MEDIUM] No retry on ValidationError — single failure is terminal

`anthropic_service.py:198`

The @policy retry decorator (lines 103-115) covers network/transport errors only. A ValidationError at line 198 is logged and immediately re-raised with no retry. With prompt-only mode this is particularly risky because the model can return structurally valid JSON that does not match the schema, and a single retry often succeeds. Even after migrating to native structured output, refusals (stop_reason='end_turn' with a refusal text) or max_tokens truncation can still yield non-conforming output.

**Recommendation:** Add ValidationError to the existing retry tuple, or wrap the model_validate call in a separate retry policy. If using client.messages.parse(), also check stop_reason before trusting parsed_output: if stop_reason is not 'end_turn', raise an exception so the @policy retry decorator handles it. Example: `if response.stop_reason != 'end_turn': raise APIResponseValidationError(...)`.

#### [MEDIUM] Prompt-cache token metrics always zero

`anthropic_service.py:178`

record_llm_metrics is called with the default cached_input_tokens=0 (line 178-184). The Anthropic API returns cache_creation_input_tokens and cache_read_input_tokens on response.usage when prompt caching is active. These are silently discarded, so the _CACHED_TOKENS_COUNTER metric in common.py is always zero, making cache hit-rate monitoring impossible.

**Recommendation:** After the response, read both fields: `cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0`. Pass that to record_llm_metrics: `cached_input_tokens=cached`. Optionally also track cache_creation_input_tokens separately for cache-write cost visibility.

#### [MEDIUM] Tokenizer sends prompt as 'assistant' role — semantically wrong

`anthropic_service.py:72`

AnthropicEstimatingTokenizer.estimate_token_count sends messages=[{"role": "assistant", "content": prompt}] (line 72). The prompt being counted is a user-turn input, so the role should be 'user'. The assistant role triggers different tokenization overhead in the Anthropic count_tokens API (e.g. turn delimiters differ), meaning estimates will be slightly off — relevant when the tokenizer is used to gate context-window decisions.

**Recommendation:** Change line 72 to `messages=[{"role": "user", "content": prompt}]`.

#### [LOW] No system prompt — full prompt sent as user message only

`anthropic_service.py:138`

The entire prompt string (instructions + schema description + examples) is sent as a single user message (line 138-139). Anthropic's best practice separates system-level instructions (system parameter) from the per-turn user message. More importantly for structured output: when using output_config.format, Anthropic recommends putting schema-agnostic instructions in the system prompt to maximise prompt-cache reuse — the schema is cached server-side for 24 hours, and a stable system prompt increases cache hit rate.

**Recommendation:** Split PromptBuilder output into system and user parts if the builder supports it, or at minimum pass a static system prompt containing the JSON formatting instruction: `system='Respond only with valid JSON matching the provided schema.'`, with the per-call content in the user message. This also aligns with the documented best practice of not changing output_config.format unnecessarily to preserve the server-side schema compilation cache.


## Google Gemini — verdict: **suboptimal**

The adapter achieves schema-constrained output through tool/function-calling forced mode, which works but is a workaround rather than the native structured-output path. The canonical mechanism since SDK v2.0 (May 2026) is `response_format={'text': {'mime_type': 'application/json', 'schema': ...}}` with constrained decoding, which is not used. Several additional gaps affect reliability (stale model name, no validation retry), cost (thinking tokens untracked on Pro), and correctness (class mutation in schema cache).

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** Constrained decoding via response_format field (since SDK v2.0/May 2026) with nested mime_type=application/json and schema; older response_mime_type + response_schema/response_json_schema pattern still works on pre-breaking-change SDK versions. Output tokens are filtered at each step so only schema-valid continuations are considered.
- **Exact API fields:** Post-May-2026 (SDK >=2.0.0): config={'response_format': {'text': {'mime_type': 'application/json', 'schema': MyModel.model_json_schema()}}}. Pre-2.0 / still accepted: types.GenerateContentConfig(response_mime_type='application/json', response_schema=MyPydanticModel) or response_json_schema=json_schema_dict. Enum responses: response_mime_type='text/x.enum' with response_schema listing enum values.
- **SDK helpers:** Pass a Pydantic BaseModel subclass as response_schema in GenerateContentConfig; SDK auto-converts to JSON Schema via 5-stage pipeline (model_json_schema → normalize → process_schema → handle_null_fields → backend serialization). response.parsed lazy property returns validated Pydantic instance via model_validate(). response.text returns raw JSON string. For plain dict schema use response_json_schema= (full JSON Schema, Gemini 2.5+) or response_schema= (OpenAPI 3.0 subset, all models).
- **Weaker JSON-mode fallback:** Set response_mime_type='application/json' (or response_format.text.mime_type in SDK>=2.0) without a schema to get unstructured JSON — model is encouraged but not guaranteed to return valid JSON. No constrained decoding without a schema.
- **Deprecated patterns:**
  - response_mime_type as a top-level GenerateContentConfig field is REMOVED as of June 8, 2026 (legacy schema permanently removed). Migrate to response_format={'text': {'mime_type': ..., 'schema': ...}}.
  - Separate response_schema and response_json_schema top-level config fields are superseded by response_format.text.schema in SDK>=2.0.0 (May 2026 breaking change). Pre-2.0 fields still accepted in SDK<2.0.
  - response_schema with OpenAPI 3.0 Schema object (non-JSON-Schema) is the legacy approach; response_json_schema (full JSON Schema) is preferred for Gemini 2.5+ models.
  - image_config in generation_config is removed — use response_format with type=image instead.
- **Limitations:**
  - Syntactic validity is guaranteed, not semantic correctness — always validate values in application code.
  - Not all JSON Schema specification features are supported; unsupported properties are silently ignored.
  - API may reject very large or deeply nested schemas.
  - Gemini API does not support additionalProperties in schemas (added Nov 2025 but still has restrictions via SDK transformation).
  - Only string literals are permitted in Pydantic Literal types — non-string Literal raises ValueError.
  - Integer enums are auto-converted to string enums.
  - Pydantic fields with Field(default=...) may be rejected by the API (known issue).
  - Gemini 2.0 models require explicit propertyOrdering list in the schema to control output key order; Gemini 2.5+ preserves schema key order implicitly.
  - anyOf, $ref (recursive structures), minimum/maximum, prefixItems are supported as of Nov 2025 structured outputs update — not available on older models/API versions.
  - Context caching compatibility with structured output is not officially documented; assume schema config is not part of the cached context.
  - $defs/$ref references in Pydantic-generated schemas are automatically inlined by the SDK before sending to the API.
- **Best practices:**
  - Use SDK>=2.0.0 and response_format={'text': {'mime_type': 'application/json', 'schema': ...}} for the current canonical pattern.
  - Prefer passing a Pydantic BaseModel directly to response_schema (pre-2.0) or as .model_json_schema() to response_format.text.schema (post-2.0) for type safety and IDE support.
  - Access response.parsed to get an auto-validated Pydantic instance instead of manually calling model_validate_json(response.text).
  - For Gemini 2.0 models, always include propertyOrdering in the schema to guarantee output key order.
  - Use response_json_schema (full JSON Schema) over response_schema (OpenAPI subset) when targeting Gemini 2.5+ — broader keyword support.
  - Keep schemas shallow and property names short to avoid API rejection of large/nested schemas.
  - Use text/x.enum as response_mime_type (or mime_type in response_format) for single-value classification tasks — more efficient than full JSON.
  - Validate semantic correctness of output values in application code even when schema validity is guaranteed.
  - For Gemini 2.5+ models with anyOf/union types, use Python Union type hints in Pydantic — the SDK handles the conversion.
- **Sources:** <https://ai.google.dev/gemini-api/docs/structured-output>, <https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026>, <https://googleapis.github.io/python-genai/>, <https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/>, <https://deepwiki.com/googleapis/python-genai/3.5.1-pydantic-model-integration>, <https://ai.google.dev/gemini-api/docs/changelog>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/gemini_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** GeminiSchematicGenerator._do_generate converts the Pydantic schema into a Gemini FunctionDeclaration via _get_schema_function_declaration (tool-forcing), sends it in GenerateContentConfig with mode=ANY so the model must call that function, then extracts the structured args from response.candidates[0].content.parts[0].function_call.args["log_data"] and validates with self.schema.model_validate().
- **Uses native structured output:** yes (Tool/function-calling forced via FunctionCallingConfigMode.ANY with a single FunctionDeclaration derived from the Pydantic schema — not response_format json_schema. Schema is transmitted as a Gemini Tool definition.)
- **Schema sent to API:** yes
- **Gating:** No "strict" hint gate — tool-forcing is always active. The only relevant hints filtered are "temperature" and "thinking_config" (supported_hints list at line 89). No strict=True default concern.
- **Parsing fallbacks:** None — no normalize_json_output call, no jsonfinder, no manual slicing. The tool-call args dict is passed directly to model_validate().
- **Retries on ValidationError:** no
- **max_tokens handling:** Hardcoded per model class: all concrete models (Gemini_2_0_Flash, Gemini_2_0_Flash_Lite, Gemini_2_5_Flash, Gemini_2_5_Flash_Lite, Gemini_2_5_Pro) return 1024*1024 (lines 270, 284, 310, 337, 351). No max_tokens parameter is sent in the API call; this value is only exposed via the max_tokens property, not forwarded to GenerateContentConfig.
- **Notable issues:**
  - Gemini_2_0_Flash_Lite is pinned to preview model name 'gemini-2.0-flash-lite-preview-02-05' (line 278) — likely stale; GA name is gemini-2.0-flash-lite.
  - model_validate ValidationError is logged and re-raised without retry (lines 224-228) — one transient schema mismatch hard-fails the call.
  - max_tokens property value (1M) is never forwarded to GenerateContentConfig, so the API uses its own default output token limit.
  - FunctionDeclaration is constructed by building a fake function with a manually patched __signature__ (lines 232-254) — fragile hack that relies on google.genai internal from_callable introspection.
  - convert_model_to_gemini_compatible_schema uses setattr(_conversion_cache) to cache on the original class (line 599) — mutates caller's Pydantic model class as a side-effect.
  - UsageInfo guard at line 208-221 duplicates the None-check already done for record_llm_metrics — inconsistent: metrics call uses 'or 0' short-circuit but UsageInfo branch falls back to zeros silently.
  - normalize_json_output in common.py is not used at all in gemini_service.py; it exists only for other adapters.
  - No thinking token accounting — thinking_budget is forced to 0 for Flash/Flash-Lite models but thinking tokens (if any) are not captured in UsageInfo.extra.
  - Gemini_2_5_Flash and Gemini_2_5_Flash_Lite override generate() to inject thinking_config={thinking_budget:0} but other models leave thinking unconstrained, which can inflate latency/cost silently.
- **Key code refs:**
  - `gemini_service.py:89 — supported_hints = ["temperature", "thinking_config"]`
  - `gemini_service.py:114-133 — @policy retry decorator on do_generate; retries NotFound, TooManyRequests, ResourceExhausted; ServerError max 2 retries`
  - `gemini_service.py:143 — gemini_api_arguments filtered from hints`
  - `gemini_service.py:145 — fd = self._get_schema_function_declaration()`
  - `gemini_service.py:147-156 — GenerateContentConfig with tools=[Tool(function_declarations=[fd])] and FunctionCallingConfigMode.ANY`
  - `gemini_service.py:160-164 — await self._client.aio.models.generate_content(...)`
  - `gemini_service.py:177-179 — json_result extracted from function_call.args["log_data"]`
  - `gemini_service.py:185 — self.schema.model_validate(json_result)`
  - `gemini_service.py:224-228 — ValidationError logged and re-raised (no retry)`
  - `gemini_service.py:230-255 — _get_schema_function_declaration: fake callable + patched __signature__ + FunctionDeclaration.from_callable`
  - `gemini_service.py:250-253 — FunctionDeclaration.from_callable(callable=log_data, client=self._client)`
  - `gemini_service.py:278 — preview model name 'gemini-2.0-flash-lite-preview-02-05'`
  - `gemini_service.py:299-306 — Gemini_2_5_Flash.generate() injects thinking_budget:0`
  - `gemini_service.py:596 — setattr(model_cls, '_conversion_cache', converted_model) mutates original class`

### Gaps

#### [HIGH] Tool-forcing used instead of native response_format constrained decoding (SDK >=2.0)

`gemini_service.py:145-179`

The adapter transmits the schema as a FunctionDeclaration and forces a tool call via FunctionCallingConfigMode.ANY. Since SDK 2.0 (May 2026) the canonical structured-output path is response_format with constrained decoding, which filters output tokens at each step to guarantee schema-valid JSON directly. Tool-forcing is an indirect workaround: it adds a function-call round-trip wrapper, requires the bespoke _get_schema_function_declaration hack, and extracts the result from function_call.args['log_data'] rather than the model's response body. The constrained-decoding path is more reliable, lower-latency, and the officially recommended approach for all structured output.

**Recommendation:** Replace the tools/tool_config block and _get_schema_function_declaration with: `config = google.genai.types.GenerateContentConfig(response_format={'text': {'mime_type': 'application/json', 'schema': self.schema.model_json_schema()}}, **gemini_api_arguments)`. Then parse via `response.parsed` (returns a validated Pydantic instance if you pass the model class to response_schema) or `self.schema.model_validate_json(response.text)`. This eliminates _get_schema_function_declaration entirely.

#### [HIGH] Fragile FunctionDeclaration construction via fake callable and patched __signature__

`gemini_service.py:230-255`

_get_schema_function_declaration creates a no-op function, monkey-patches its __signature__ with the converted Pydantic type as a parameter annotation, then relies on FunctionDeclaration.from_callable internal introspection. This is an undocumented internal API usage — from_callable is not a public SDK surface for this pattern. Any SDK upgrade that changes how from_callable inspects signatures will silently break structured output for all Gemini models.

**Recommendation:** Switch to native response_format as described in gap 1. This removes the entire _get_schema_function_declaration method and the convert_model_to_gemini_compatible_schema dependency for structured output. The schema is sent as `self.schema.model_json_schema()` (a plain dict) — no fake callable, no __signature__ patching.

#### [HIGH] Pydantic model class mutated as schema conversion cache

`gemini_service.py:596-599`

convert_model_to_gemini_compatible_schema calls `setattr(model_cls, '_conversion_cache', converted_model)` on the original Pydantic model class passed in by caller code. This silently modifies a shared class object that may be used across threads or requests. The cache is never invalidated, and the attribute name '_conversion_cache' is not namespaced, making collision with user-defined fields possible.

**Recommendation:** Replace with a module-level WeakKeyDictionary: `_schema_conversion_cache: WeakKeyDictionary[type, type] = WeakKeyDictionary()`. Check and store there instead of using setattr on model_cls. With the response_format migration (gap 1) this entire conversion function becomes unnecessary.

#### [HIGH] Stale preview model name for Gemini_2_0_Flash_Lite

`gemini_service.py:276`

Gemini_2_0_Flash_Lite hardcodes 'gemini-2.0-flash-lite-preview-02-05', a preview/experimental model identifier. The GA model name is 'gemini-2.0-flash-lite'. Preview endpoints may be retired without notice, breaking any caller that requests ModelSize.NANO or explicitly instantiates this class.

**Recommendation:** Change line 276 to `model_name='gemini-2.0-flash-lite'`.

#### [MEDIUM] No retry on ValidationError — single transient schema mismatch hard-fails

`gemini_service.py:114-127, gemini_service.py:224-228`

When self.schema.model_validate(json_result) raises ValidationError (line 185), the exception is logged and immediately re-raised. The @policy retry decorator on do_generate only catches NotFound, TooManyRequests, ResourceExhausted, and ServerError — ValidationError is not retried. A single malformed response permanently fails the call even though a retry would succeed in most cases.

**Recommendation:** Add ValidationError to the retry policy on do_generate, or wrap the validation call in a local retry loop (e.g. up to 2 retries). If migrating to response_format + response.parsed, the SDK's own validation raises a google.genai.errors.ClientError on schema mismatch — catch that instead and add it to the @policy retry list.

#### [MEDIUM] max_tokens property value never forwarded to GenerateContentConfig

`gemini_service.py:147-156`

All model subclasses return 1024*1024 from max_tokens but _do_generate never sets max_output_tokens in GenerateContentConfig. The API falls back to its own default output token limit (typically 8192 tokens for most Gemini models), which is far less than 1M. For long structured JSON responses (deeply nested schemas, large arrays) this will cause the model to stop mid-output and produce truncated, unparseable JSON.

**Recommendation:** Add `max_output_tokens=self.max_tokens` to GenerateContentConfig: `config = google.genai.types.GenerateContentConfig(..., max_output_tokens=self.max_tokens, ...)`. Alternatively use a model-appropriate ceiling (e.g. 8192) rather than the 1M placeholder if the intent was just to express 'no artificial limit'.

#### [MEDIUM] Thinking tokens not accounted for in UsageInfo on Gemini 2.5 Pro

`gemini_service.py:208-221`

Gemini_2_5_Flash and Gemini_2_5_Flash_Lite inject thinking_budget=0, suppressing thinking tokens. Gemini_2_5_Pro has no such suppression. When the Pro model uses extended thinking, usage_metadata contains thoughts_token_count in addition to candidates_token_count, but the code only captures candidates_token_count as output_tokens. Thinking tokens are billed and unreported, causing cost accounting to be wrong.

**Recommendation:** Add thoughts_token_count to UsageInfo.extra: `'thinking_tokens': response.usage_metadata.thoughts_token_count or 0 if response.usage_metadata else 0`. Access via `response.usage_metadata.thoughts_token_count` (available in google-genai SDK for 2.5 models).

#### [LOW] response.parsed not used — manual model_validate after tool-call extraction

`gemini_service.py:177-185`

After migrating to response_format (gap 1), the SDK provides response.parsed which returns an already-validated Pydantic instance when a Pydantic model is passed as response_schema. The current code manually calls self.schema.model_validate(json_result) on a raw dict. Using response.parsed removes the manual validation step and leverages SDK-side schema application.

**Recommendation:** When using response_format with a Pydantic class: `config = google.genai.types.GenerateContentConfig(response_format={'text': {'mime_type': 'application/json', 'schema': self.schema.model_json_schema()}})` and then access `response.parsed` (or `self.schema.model_validate_json(response.text)` as the explicit equivalent). The response.parsed path requires passing the Pydantic class to response_schema= instead of the dict to response_format.text.schema — pick the dict path and use model_validate_json(response.text) for full control.

#### [LOW] Gemini 2.0 models missing propertyOrdering in schema

`gemini_service.py:147-156`

The provider docs state that Gemini 2.0 models require an explicit propertyOrdering list in the schema to guarantee output key order; without it the key order is undefined. The current tool-forcing approach passes an auto-derived schema without propertyOrdering. For schemas where key order matters (e.g. step-by-step reasoning fields) this can cause issues on Gemini 2.0 Flash.

**Recommendation:** When building the schema dict for response_format (post-migration), inject propertyOrdering for Gemini 2.0 models: `schema = self.schema.model_json_schema(); schema['propertyOrdering'] = list(schema.get('properties', {}).keys())`. This is a no-op on 2.5+ models where key order is preserved implicitly.


## Google Vertex AI — verdict: **suboptimal**

The Gemini path uses native constrained decoding correctly but passes the schema as a raw dict (model_json_schema()) instead of a Pydantic model, misses response.parsed, and never calls record_llm_metrics. The Claude-on-Vertex path sends no schema to the API at all — it relies entirely on prompt text and jsonfinder scraping — making structured output unreliable. Neither path retries on ValidationError. Additional issues: max_tokens silently dropped for Gemini, a logging bug in __init__, deprecated 1.5-series model classes, and an inaccurate tiktoken-based tokenizer for Claude.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** Controlled generation via response_mime_type + response_schema (OpenAPI 3.0 subset) or response_json_schema (full JSON Schema with conditionals) in GenerateContentConfig; constrained decoding enforces schema at token generation time for Gemini models. For Claude-on-Vertex via AnthropicVertex SDK, structured outputs are listed as supported (GA as of 2026) using output_config.format with type=json_schema or strict tool_use.
- **Exact API fields:** Gemini models (google-genai SDK): client = genai.Client(vertexai=True, project='...', location='us-central1'); config=types.GenerateContentConfig(response_mime_type='application/json', response_schema=MyPydanticModel) OR response_json_schema={...JSON Schema dict...}); response.parsed returns typed object. For enums: response_mime_type='text/x.enum', response_schema=MyEnum. Claude-on-Vertex (AnthropicVertex SDK): client.messages.create(..., output_config={'format': {'type': 'json_schema', 'schema': {...}}}) or strict tool forcing: tools=[{..., 'strict': True}], tool_choice={'type': 'tool', 'name': 'output_json'}.
- **SDK helpers:** Python google-genai: pass Pydantic BaseModel directly to response_schema; response.parsed returns the populated model instance. client.models.generate_content() with types.GenerateContentConfig. AnthropicVertex SDK: client.messages.parse(model=..., output_format=MyPydanticModel, messages=[...]) returns parsed_output with validated Pydantic instance. Environment variable alternative: GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION.
- **Weaker JSON-mode fallback:** Set response_mime_type='application/json' without response_schema to get unconstrained JSON output (no schema enforcement). For Claude-on-Vertex before structured outputs GA: use tool-forcing with tool_choice={'type': 'tool', 'name': 'output_json'} and a tool with the desired input_schema — forces schema-shaped output without native constrained decoding.
- **Deprecated patterns:**
  - Old vertexai Python SDK (google-cloud-aiplatform / vertexai.generative_models) is deprecated in favor of google-genai SDK with vertexai=True. Migrate per https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk
  - Claude structured outputs: older output_format parameter and structured-outputs-2025-11-13 betas header continue working during transition but should migrate to output_config.format
  - Passing schema inline in the prompt text alongside response_schema is discouraged — docs warn it may reduce output quality and is redundant
- **Limitations:**
  - response_schema uses OpenAPI 3.0 subset — z.union / z.record and JSON Schema features like allOf conditionals require response_json_schema field instead
  - response_json_schema field needed for advanced JSON Schema features (if/then/else, allOf, conditional required)
  - Very large or deeply nested schemas may be rejected by the API
  - Claude-on-Vertex structured outputs (output_config.format): no minimum/maximum/minLength/maxLength numeric constraints; no recursive schemas; no external $ref; max 24 optional parameters per request; max 16 union anyOf parameters; max 20 strict tools per request
  - Constrained decoding applies to Gemini models natively; Claude-on-Vertex structured outputs are GA but implemented via Anthropic-side grammar constraints routed through Google infrastructure
  - Consumer Gemini API (ai.google.dev) and Vertex AI API share the same google-genai SDK and response_schema mechanism but auth/client init differs (API key vs vertexai=True + GCP project)
- **Best practices:**
  - Use google-genai SDK with vertexai=True (not the old vertexai SDK) for all new Gemini-on-Vertex work
  - Pass Pydantic BaseModel directly to response_schema — SDK handles schema conversion and response.parsed returns a typed model instance
  - Use response_json_schema (not response_schema) when you need advanced JSON Schema features like if/then conditionals or allOf
  - Do not repeat schema structure in the prompt text when using response_schema — it degrades quality
  - For Claude-on-Vertex, use AnthropicVertex SDK with output_config.format (json_schema) or client.messages.parse() with a Pydantic model for constrained output
  - For Claude-on-Vertex when native structured outputs are insufficient, fall back to strict tool-forcing: define a tool named output_json with input_schema matching desired structure and set tool_choice to force it
  - Set GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION env vars as an alternative to passing vertexai=True in code
- **Sources:** <https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output>, <https://googleapis.github.io/python-genai/>, <https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk>, <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>, <https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai>, <https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/controlled-generation/intro_controlled_generation.ipynb>, <https://ai.google.dev/gemini-api/docs/structured-output>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/vertex_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** Two separate generator classes handle the two providers. VertexAIClaudeSchematicGenerator (line 126) builds a plain messages.create() call via AsyncAnthropicVertex with a single user-role message; the schema is NOT sent to the API — only the prompt text. VertexAIGeminiSchematicGenerator (line 278) builds a generate_content() call via google.genai.Client with response_mime_type=application/json and response_schema=self.schema.model_json_schema() baked directly into the config dict — the schema IS sent. Both classes then parse the raw text response with normalize_json_output + jsonfinder.only_json and validate with pydantic model_validate.
- **Uses native structured output:** yes (Claude path: no structured-output mechanism — prompt-only, output scraped. Gemini path: google.genai response_mime_type=application/json + response_schema (native Gemini JSON-schema constrained decoding).)
- **Schema sent to API:** yes
- **Gating:** No hint gate at all. Gemini always sends response_schema; Claude never does. There is no "strict" hint or any conditional toggle — the provider is chosen solely by model name. Default: schema always sent for Gemini, never for Claude.
- **Parsing fallbacks:** normalize_json_output strips ```json fences (common.py:19-32); jsonfinder.only_json scavenges the first JSON value from the string (vertex_service.py:241 for Claude, :401 for Gemini); Gemini path additionally replaces Unicode curly-quotes with straight quotes (vertex_service.py:395) and fixes double-escaped control chars \\u/\\t/\\n (vertex_service.py:397-399)
- **Retries on ValidationError:** no
- **max_tokens handling:** Claude: hints.get('max_tokens', 8192) — default 8192, overridable via hints (vertex_service.py:208). Gemini: no max_tokens sent at all; 'max_tokens' is not in supported_hints (vertex_service.py:281), so it is silently dropped even if passed. The max_tokens property on both classes reflects context-window size only (200k for Claude, 1M/2M for Gemini) and is not passed to the API.
- **Notable issues:**
  - Claude path sends NO schema to the API — relies entirely on prompt instructions and jsonfinder scraping, making it fragile and inconsistent with the Gemini path
  - Gemini path does not call record_llm_metrics (vertex_service.py:406-436 has no record_llm_metrics call, unlike Claude at line 251) — Gemini token usage is logged via trace but never metered
  - VertexAIService.__init__ logs region as self.project_id twice: 'region {self.project_id}' should be 'region {self.region}' (vertex_service.py:769)
  - gemini-1.5-flash and gemini-1.5-pro model classes are present but those models are effectively deprecated/EOL in Vertex AI as of 2025
  - VertexAIEstimatingTokenizer uses tiktoken gpt-4o-2024-08-06 encoding for Claude with a 1.15× fudge factor (vertex_service.py:92-94) — inaccurate approximation
  - max_tokens is silently ignored for all Gemini models (vertex_service.py:281 supported_hints does not include it)
  - VertexClaudeSonnet35 and VertexClaudeHaiku35 pin claude-3-5-sonnet-v2@20241022 and claude-3-5-haiku@20241022 — older generation models still in use alongside newer claude-4 variants
  - ValidationError on pydantic schema mismatch immediately raises in both paths — no retry (vertex_service.py:271-275 and :434-436)
- **Key code refs:**
  - `vertex_service.py:205 — AsyncAnthropicVertex messages.create() call (Claude)`
  - `vertex_service.py:208 — max_tokens default 8192 from hints`
  - `vertex_service.py:237 — raw_content = response.content[0].text`
  - `vertex_service.py:240 — normalize_json_output(raw_content)`
  - `vertex_service.py:241 — jsonfinder.only_json(json_content)[2]`
  - `vertex_service.py:249 — self.schema.model_validate(json_object)`
  - `vertex_service.py:271 — ValidationError caught and re-raised (no retry)`
  - `vertex_service.py:350-354 — Gemini config dict with response_mime_type and response_schema`
  - `vertex_service.py:358-362 — google.genai aio.models.generate_content() call`
  - `vertex_service.py:393-401 — normalize_json_output + curly-quote fix + double-escape fix + jsonfinder`
  - `vertex_service.py:410 — self.schema.model_validate(json_object) (Gemini)`
  - `vertex_service.py:434 — ValidationError caught and re-raised (no retry, Gemini)`
  - `vertex_service.py:251-257 — record_llm_metrics called for Claude only`
  - `vertex_service.py:406 — usage_metadata logged but record_llm_metrics NOT called for Gemini`
  - `vertex_service.py:769 — bug: logs region as self.project_id instead of self.region`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] Claude path sends no schema to the API — prompt-scraping only

`vertex_service.py:205`

VertexAIClaudeSchematicGenerator.messages.create() passes only a plain user message. The Pydantic schema is never transmitted to the AnthropicVertex API. The model must infer the desired shape from prompt text alone, so any deviation causes a ValidationError with no fallback. The current API (GA as of 2026) supports output_config.format with type='json_schema' for constrained decoding.

**Recommendation:** Add output_config to the messages.create() call: `output_config={'format': {'type': 'json_schema', 'schema': self.schema.model_json_schema()}}`. Alternatively use strict tool-forcing: define a tool `{'name': 'output_json', 'description': '...', 'input_schema': self.schema.model_json_schema()}` and set `tool_choice={'type': 'tool', 'name': 'output_json'}`, then read `response.content[0].input` instead of `.text`.

#### [HIGH] Gemini path passes schema as raw dict instead of Pydantic model — loses response.parsed

`vertex_service.py:350`

At line 352, `response_schema` receives `self.schema.model_json_schema()` (a plain dict). The google-genai SDK accepts a Pydantic BaseModel directly, converts the schema itself, and populates `response.parsed` with a validated model instance — eliminating the normalize_json_output + jsonfinder + curly-quote + double-escape pipeline entirely. Passing a raw dict forces the code to scrape and re-parse text that the API already decoded.

**Recommendation:** Pass the Pydantic class directly: `config = types.GenerateContentConfig(response_mime_type='application/json', response_schema=self.schema, **gemini_api_arguments)`. Then use `response.parsed` (a typed instance of `self.schema`) instead of the jsonfinder pipeline. Remove the normalize_json_output, curly-quote, and double-escape fixups (lines 393–401) once response.parsed is in use.

#### [HIGH] Neither path retries on ValidationError

`vertex_service.py:271`

Both VertexAIClaudeSchematicGenerator (line 271) and VertexAIGeminiSchematicGenerator (line 434) catch ValidationError, log it, and immediately re-raise with no retry. A transient schema mismatch (e.g., the model omitting an optional field or wrapping output in an extra key) causes a permanent hard failure for the caller. The existing @policy decorator only retries network/rate-limit exceptions.

**Recommendation:** Add ValidationError to the retry policy, or implement an inner retry loop (up to 2 attempts) inside _do_generate that re-calls the API and re-parses. For the Claude path this is especially important since the schema is not constrained at generation time. Example: wrap the validate call in `for attempt in range(max_retries): try: return ...; except ValidationError: if attempt == max_retries-1: raise`.

#### [MEDIUM] max_tokens silently ignored for all Gemini models

`vertex_service.py:281`

VertexAIGeminiSchematicGenerator.supported_hints is ['temperature', 'thinking_config'] (line 281). Any max_tokens hint passed by callers is silently dropped — it is never added to the config dict sent to generate_content(). For deeply-nested or large schemas, the model may truncate output mid-JSON causing a jsonfinder or ValidationError failure with no indication that the output was truncated.

**Recommendation:** Add 'max_tokens' to supported_hints and map it to the GenerateContentConfig field: `config = {'response_mime_type': 'application/json', 'response_schema': ..., 'max_output_tokens': hints.get('max_tokens'), ...}`. The google-genai SDK field name is max_output_tokens (not max_tokens).

#### [MEDIUM] Gemini path never calls record_llm_metrics — token usage not metered

`vertex_service.py:406`

VertexAIClaudeSchematicGenerator calls record_llm_metrics at line 251 after a successful generation. VertexAIGeminiSchematicGenerator reads usage_metadata and logs it (line 406) but never calls record_llm_metrics, so Gemini token usage is invisible to the metrics/billing layer.

**Recommendation:** After successful model_validate in _do_generate (around line 410), add: `await record_llm_metrics(self.meter, self.model_name, schema_name=self.schema.__name__, input_tokens=response.usage_metadata.prompt_token_count or 0, output_tokens=response.usage_metadata.candidates_token_count or 0)` — matching the pattern at vertex_service.py:251–257.

#### [MEDIUM] VertexAIService.__init__ logs region as self.project_id twice

`vertex_service.py:768`

Line 768: `f'in project {self.project_id}, region {self.project_id}'` — the second interpolation should be `self.region`. This makes the log misleading (both values show the project ID) and hides which region the client is actually connecting to.

**Recommendation:** Change line 768 to: `f'in project {self.project_id}, region {self.region}'`.

#### [MEDIUM] VertexGemini15Flash and VertexGemini15Pro target EOL models

`vertex_service.py:495`

VertexGemini15Flash (model='gemini-1.5-flash') and VertexGemini15Pro (model='gemini-1.5-pro') are still registered model classes. Gemini 1.5 reached end-of-support on Vertex AI in 2025. Calling these models will return API errors or degraded service once Google finalises the shutdown.

**Recommendation:** Remove VertexGemini15Flash and VertexGemini15Pro classes (lines 495–520) and remove their entries from any model registry/GEMINI_MODELS dict. Callers should migrate to VertexGemini20Flash or VertexGemini25Flash.

#### [LOW] Claude tokenizer uses GPT-4o tiktoken encoding with a 1.15x fudge factor

`vertex_service.py:92`

VertexAIEstimatingTokenizer uses tiktoken's gpt-4o-2024-08-06 BPE encoding (line 92–93) for Claude models and applies a 1.15× multiplier. Claude and GPT-4o use different tokenizers; the approximation can be 10–20% off in either direction depending on content, leading to inaccurate quota checks and potentially missing max_tokens headroom calculations.

**Recommendation:** Use the AnthropicVertex client's native token counting: `await self._anthropic_client.messages.count_tokens(model=self.model_name, messages=[{'role':'user','content':prompt}])`. This requires storing a reference to the AsyncAnthropicVertex client in the tokenizer. Alternatively, accept the approximation but document it and increase the fudge factor to at least 1.25×.


## AWS Bedrock — verdict: **suboptimal**

The Bedrock adapter uses pure prompt-scraping to obtain structured JSON: no schema is sent to the API, no constrained decoding is activated, and post-hoc regex/jsonfinder parsing can silently produce schema-invalid output. The SDK at version 0.83.0 already exposes output_config.format with type="json_schema" on AsyncAnthropicBedrock.messages.create, making native structured output directly available without any new dependencies. Beyond the core structured-output gap there are five additional reliability/correctness issues: a broken environment check, dead max_tokens code, wrong tokenizer, a missing retry on ValidationError, and a stale model ID.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** Two complementary mechanisms: (1) outputConfig.textFormat with json_schema type in Converse/ConverseStream APIs — the primary recommended approach for full-response JSON schema enforcement; (2) toolConfig with strict:true on toolSpec plus toolChoice.tool to force a single tool call — the original workaround that predates native structured output. Both use constrained decoding (compiled grammar artifacts) to guarantee schema-valid output. Generally available since February 4, 2026.
- **Exact API fields:** Converse/ConverseStream API — outputConfig field:

outputConfig={
    'textFormat': {
        'type': 'json_schema',
        'structure': {
            'jsonSchema': {
                'schema': json.dumps(schema_dict),  # JSON Schema Draft 2020-12 subset, serialized as string
                'name': 'my_schema',                # required
                'description': 'optional'           # optional
            }
        }
    }
}

Alternative — toolConfig forced tool use (works for all models supporting tool use):

toolConfig={
    'tools': [{
        'toolSpec': {
            'name': 'extract_data',
            'description': '...',
            'strict': True,           # enforces schema compliance on inputs
            'inputSchema': {
                'json': {
                    'type': 'object',
                    'properties': {...},
                    'required': [...],
                    'additionalProperties': False
                }
            }
        }
    }],
    'toolChoice': {'tool': {'name': 'extract_data'}}  # forces the model to call this tool
}

InvokeModel API (Anthropic Claude): output_config.format with type='json_schema'
InvokeModel API (open-weight models): response_format with json_schema key

All four APIs supported: Converse, ConverseStream, InvokeModel, InvokeModelWithResponseStream
- **SDK helpers:** No first-party Pydantic or high-level helpers in boto3 itself. Schema must be serialized to a JSON string manually (json.dumps(schema_dict)) for the schema field. Parse response with json.loads(response['output']['message']['content'][0]['text']). LangChain-AWS (langchain-aws) has an open issue (#883) to add native structured output support via with_structured_output(). LiteLLM tracks support via issue #21208. Use boto3 bedrock-runtime client: boto3.client('bedrock-runtime', region_name='...')
- **Weaker JSON-mode fallback:** For models that do not support outputConfig.textFormat: use toolConfig with a single tool definition (strict:True + inputSchema) and toolChoice={'tool': {'name': '...'}} to force the model to return structured tool call inputs. Extract the structured data from response['output']['message']['content'][0]['toolUse']['input']. This approach works broadly across all Bedrock models that support tool use (Claude, Nova, many open-weight models).
- **Deprecated patterns:**
  - Prompt engineering only (asking the model to return JSON in system/user prompt) — no schema validation or constrained decoding guarantee; replaced by outputConfig.textFormat
  - Anthropic Messages API on the bedrock-mantle endpoint does not support output_config.format — use bedrock-runtime Converse or InvokeModel instead (returns 400 error otherwise)
  - Prefilling the assistant turn with '{' or '```json' to steer JSON output — legacy technique with no compliance guarantee
- **Limitations:**
  - additionalProperties must be set to false on all objects — this is required, not optional
  - Recursive schemas not supported
  - External $ref references not supported
  - Numerical constraints (minimum, maximum, multipleOf) not supported
  - String length constraints (minLength, maxLength) not supported
  - Array minItems only supports values 0 or 1
  - Schema compilation on first use can take up to a few minutes; compiled grammars cached 24 hours per account
  - outputConfig.textFormat only available on select models: Anthropic Claude 4.5+, and select open-weight models (DeepSeek, Google Gemma, MiniMax, Mistral, Moonshot, NVIDIA Nemotron, OpenAI, Qwen) — not all Bedrock models
  - schema field in jsonSchema must be a JSON-serialized string, not a dict object
- **Best practices:**
  - Use outputConfig.textFormat (Converse API) as the primary approach — it targets the full response and is the recommended native mechanism as of Feb 2026
  - Always set additionalProperties: false on every object in the schema — required for structured outputs to work
  - Use toolConfig with strict:True + toolChoice for models that don't yet support outputConfig.textFormat, or for agentic workflows where you specifically need validated function call parameters
  - Serialize the schema as a JSON string: 'schema': json.dumps(your_schema_dict)
  - Set temperature=0 in inferenceConfig for deterministic, consistent structured outputs
  - Use cross-region inference profile model IDs (e.g. us.anthropic.claude-...) to improve availability
  - Keep schemas simple — avoid recursive refs and numerical/string constraints which are unsupported
  - Pre-warm schemas in non-prod environments if latency matters — first compilation can take minutes; subsequent calls use 24h cached grammar
- **Sources:** <https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html>, <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>, <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html>, <https://aws.amazon.com/about-aws/whats-new/2026/02/structured-outputs-available-amazon-bedrock/>, <https://aws.amazon.com/blogs/machine-learning/structured-outputs-on-amazon-bedrock-schema-compliant-ai-responses/>, <https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/aws_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** AnthropicBedrockAISchematicGenerator._do_generate builds a single-turn request with one user-role message containing the full prompt string. It calls self._client.messages.create (AsyncAnthropicBedrock SDK) with model, max_tokens=4096, and any hints that match supported_hints (only "temperature"). The raw text response is then passed through normalize_json_output (strips ```json fences) then jsonfinder.only_json to extract a JSON object, which is validated with self.schema.model_validate.
- **Uses native structured output:** no (none (prompt-only) — no response_format, json_schema, tool-forcing, or any structured-output API parameter is passed to Bedrock)
- **Schema sent to API:** no
- **Gating:** N/A — no strict hint or any native structured output path exists; supported_hints = ["temperature"] only
- **Parsing fallbacks:** normalize_json_output strips ```json fences (common.py:19-32); jsonfinder.only_json scavenges first JSON object from arbitrary text (aws_service.py:157)
- **Retries on ValidationError:** no
- **max_tokens handling:** Hardcoded to 4096 in the messages.create call (aws_service.py:134). Not sourced from hints. Claude_Sonnet_3_5.max_tokens property returns 200*1024 but this property is never consulted in do_generate.
- **Notable issues:**
  - max_tokens is hardcoded to 4096 (aws_service.py:134) — the Claude_Sonnet_3_5.max_tokens property (aws_service.py:205-207) returning 200*1024 is dead code never used in the API call
  - BedrockService.verify_environment checks ANTHROPIC_API_KEY (aws_service.py:214) instead of AWS_ACCESS_KEY_ID — wrong env var, will always pass even when AWS credentials are missing
  - Only a single 'user' role message is sent; no system role is used even though Anthropic's Messages API supports and recommends a system parameter for instructions
  - ValidationError (pydantic) is caught, logged, and re-raised without retry — a bad model response causes immediate failure with no recovery attempt
  - AnthropicBedrockEstimatingTokenizer uses tiktoken with gpt-4o-2024-08-06 encoding (aws_service.py:58) to estimate tokens for a Claude model — cross-model estimation with a 1.15 fudge factor, not accurate for Anthropic tokenization
  - supported_hints only includes 'temperature'; hints like max_tokens or top_p passed by callers are silently dropped
  - Model pinned to anthropic.claude-3-5-sonnet-20240620-v1:0 (aws_service.py:197) — this is an older snapshot of Claude 3.5 Sonnet; newer versions exist (20241022)
- **Key code refs:**
  - `aws_service.py:131-136 — messages.create call with hardcoded max_tokens=4096`
  - `aws_service.py:127 — hint filtering: only 'temperature' passes through`
  - `aws_service.py:153 — raw_content = response.content[0].text`
  - `aws_service.py:156 — normalize_json_output(raw_content)`
  - `aws_service.py:157 — jsonfinder.only_json(json_content)[2]`
  - `aws_service.py:165 — self.schema.model_validate(json_object)`
  - `aws_service.py:187-191 — ValidationError caught, logged, re-raised (no retry)`
  - `aws_service.py:97-108 — @policy retry decorators (only API/connection errors)`
  - `aws_service.py:194-207 — Claude_Sonnet_3_5 class with dead max_tokens property`
  - `aws_service.py:213-214 — verify_environment checks wrong env var ANTHROPIC_API_KEY`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] Schema never sent to API — constrained decoding unused

`src/parlant/adapters/nlp/aws_service.py:131`

messages.create is called with no output_config parameter, so the model receives no schema contract. The response is a free-form text that is then scraped with normalize_json_output + jsonfinder. This means (a) the model can return any text, (b) field names can be wrong or missing, (c) jsonfinder will silently pick the first JSON-shaped blob even if it does not match the Pydantic schema, and (d) ValidationError is the only backstop — at which point the call has already been billed and the caller receives an exception instead of a result.

**Recommendation:** Pass output_config to messages.create using the schema from self.schema.model_json_schema(). The SDK's OutputConfigParam / JSONOutputFormatParam accept a plain dict:

```python
from anthropic.types.output_config_param import OutputConfigParam
from anthropic.types.json_output_format_param import JSONOutputFormatParam

response = await self._client.messages.create(
    messages=[{"role": "user", "content": prompt}],
    model=self.model_name,
    max_tokens=4096,
    output_config=OutputConfigParam(
        format=JSONOutputFormatParam(
            type="json_schema",
            schema=self.schema.model_json_schema(),
        )
    ),
    **anthropic_api_arguments,
)
```

With this in place, response.content[0].text is guaranteed to be schema-valid JSON; replace the jsonfinder pipeline with json.loads(response.content[0].text) and keep model_validate only as a defensive assertion. Add additionalProperties: false recursively to the schema dict before passing it (required by Bedrock constrained decoding). Gate behind a hints.get('strict', True) flag consistent with other adapters if non-strict mode is still needed.

#### [HIGH] verify_environment checks ANTHROPIC_API_KEY instead of AWS credentials

`src/parlant/adapters/nlp/aws_service.py:213`

BedrockService.verify_environment tests os.environ.get('ANTHROPIC_API_KEY') and returns the warning message when that variable is absent. AWS Bedrock authenticates via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION — not an Anthropic API key. The check will always pass when only AWS credentials are present (the common case), and will incorrectly warn (or silently fail later) in the opposite scenario. Because __init__ reads AWS_ACCESS_KEY_ID directly (line 79), a missing key will throw a KeyError at runtime rather than at startup.

**Recommendation:** Replace the check body:

```python
@staticmethod
def verify_environment() -> str | None:
    missing = [v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION") if not os.environ.get(v)]
    if missing:
        return (
            "You're using the AWS Bedrock NLP service, but the following environment "
            f"variables are missing: {', '.join(missing)}\n"
            "Please set them before running Parlant."
        )
    return None
```

#### [HIGH] max_tokens hardcoded to 4096 — Claude_Sonnet_3_5.max_tokens is dead code

`src/parlant/adapters/nlp/aws_service.py:134`

messages.create always passes max_tokens=4096 regardless of model capacity (line 134). Claude_Sonnet_3_5 declares max_tokens returning 200*1024 (line 205-206) but this property is never consulted in _do_generate, making it entirely dead. A structured JSON response for a large schema or long conversation can easily exceed 4096 tokens, causing the model to truncate mid-JSON, producing malformed output that fails jsonfinder/model_validate with no clear error.

**Recommendation:** Use self.schema.model_json_schema() to derive an output budget, or at minimum use the model's declared max_tokens as a ceiling. A pragmatic fix:

```python
response = await self._client.messages.create(
    messages=[{"role": "user", "content": prompt}],
    model=self.model_name,
    max_tokens=hints.get("max_tokens", 4096),  # allow caller override
    ...
)
```

Also add 'max_tokens' to supported_hints (line 67) so callers can pass it through. The dead max_tokens property on Claude_Sonnet_3_5 should either be wired to the API call or removed.

#### [MEDIUM] No retry on ValidationError — single bad response causes permanent failure

`src/parlant/adapters/nlp/aws_service.py:187`

The @policy decorator at line 97 retries on API-level errors (connection, timeout, rate limit, internal server error) but ValidationError (pydantic) is caught at line 187, logged, and immediately re-raised. Without native structured output this is especially fragile: the model can return slightly malformed JSON on any call. Once output_config is enabled the risk drops significantly but a retry remains valuable for edge cases. Other adapters in the codebase (e.g., openai_service.py:305) also do not retry on ValidationError — but this is particularly painful here because there is no constrained decoding backstop at all.

**Recommendation:** Wrap do_generate's @policy to also retry ValidationError with a small backoff, or implement an inner retry loop in _do_generate:

```python
@policy(
    [
        retry(exceptions=(APIConnectionError, APITimeoutError, RateLimitError, APIResponseValidationError)),
        retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        retry(ValidationError, max_exceptions=2, wait_times=(0.5, 2.0)),
    ]
)
```

This is most valuable before native structured output is enabled but provides defense-in-depth afterwards too.

#### [MEDIUM] Tokenizer uses GPT-4o tiktoken encoding for a Claude model

`src/parlant/adapters/nlp/aws_service.py:58`

AnthropicBedrockEstimatingTokenizer initializes tiktoken with 'gpt-4o-2024-08-06' encoding (line 58) and applies a 1.15 fudge factor. Anthropic Claude uses a different BPE vocabulary; the overlap with GPT-4o is significant but not identical, meaning token count estimates used for context-window budget decisions can be systematically off — particularly for non-English text, code, or structured data with many special characters.

**Recommendation:** Use anthropic's own token counting endpoint for accurate counts, or switch to a Claude-specific estimator. The Anthropic SDK exposes client.beta.messages.count_tokens() for exact counts. For a lightweight approach, consider the anthropic-tokenizer-python package or simply replace tiktoken with the anthropic client's count_tokens call:

```python
count_response = await self._client.beta.messages.count_tokens(
    messages=[{"role": "user", "content": prompt}],
    model=self.model_name,
)
return count_response.input_tokens
```

If a synchronous estimator is required, the fudge factor should be increased (1.3–1.4x) to be conservative rather than the current under-estimate.

#### [LOW] Model pinned to claude-3-5-sonnet-20240620-v1:0 — stale snapshot

`src/parlant/adapters/nlp/aws_service.py:197`

Claude_Sonnet_3_5 hardcodes model_name='anthropic.claude-3-5-sonnet-20240620-v1:0' (line 197). The June 2024 snapshot predates the October 2024 refresh (20241022) and the Claude 3.7 / 4.x family available on Bedrock. More importantly, native structured output via output_config.format requires Claude 4.5+ (per provider docs). The pinned model will not support outputConfig.textFormat once that path is implemented.

**Recommendation:** Update the default model to a version that supports native structured output. For Bedrock, use a cross-region inference profile for better availability:

```python
model_name="us.anthropic.claude-sonnet-4-5-20251101-v1:0"
```

Or at minimum update to the latest 3.5 Sonnet snapshot:

```python
model_name="anthropic.claude-3-5-sonnet-20241022-v2:0"
```

Note: if the model does not support outputConfig.textFormat, fall back to the toolConfig forced-tool-call mechanism (toolConfig with strict:True + toolChoice={'tool': {'name': '...'}}) and extract the result from response.content[0].tool_use.input.

#### [LOW] Prompt caching not enabled — repeated system/instruction tokens billed at full rate

`src/parlant/adapters/nlp/aws_service.py:131`

The messages.create call sends a single user-role message containing the entire prompt string (line 132). Parlant's prompts include large static instruction blocks (guidelines, context, schema descriptions) that are identical across calls within a session. The Anthropic Bedrock SDK supports prompt caching via the cache_control parameter on message content blocks (available in AsyncAnthropicBedrock.messages.create as a top-level parameter per the SDK signature). No caching is configured, so these tokens are re-billed on every call.

**Recommendation:** Split the prompt into a static system instruction block and a dynamic user block, and mark the static portion for caching:

```python
response = await self._client.messages.create(
    system=[
        {
            "type": "text",
            "text": system_instructions,  # static per-schema instructions
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": dynamic_prompt}],
    model=self.model_name,
    max_tokens=4096,
    ...
)
```

This requires refactoring PromptBuilder to expose a system/user split, consistent with how other LLM providers (OpenAI developer role, Anthropic system param) separate static context from dynamic input. Record cached_input_tokens from response.usage.cache_read_input_tokens in the GenerationInfo.usage.extra dict, matching the pattern used in record_llm_metrics.


## Mistral — verdict: **suboptimal**

The Mistral adapter uses json_object mode unconditionally, never transmitting the Pydantic schema to the API. Constrained decoding is entirely unused. Schema validation is a fragile client-side post-hoc operation with no retry, a crash-prone bare assert on usage, and prompt-scraping fallbacks that the API would eliminate. Six concrete gaps exist; two are high severity because they directly affect whether the model produces a schema-conformant response.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** response_format with type=json_schema and nested json_schema object (name, strict, schema); SDK helper client.chat.parse() accepts Pydantic (Python) or Zod (TypeScript) models directly
- **Exact API fields:** response_format={"type": "json_schema", "json_schema": {"schema": {"properties": {...}, "required": [...], "type": "object", "additionalProperties": false}, "name": "<schema_name>", "strict": true}}
- **SDK helpers:** Python: client.chat.parse(model=..., messages=..., response_format=MyPydanticModel) — pass the Pydantic class directly; access parsed result via chat_response.choices[0].message.parsed (typed object) or .content (raw JSON string). TypeScript: client.chat.parse({responseFormat: ZodSchema}). Under the hood, the SDK calls response_format_from_pydantic_model() to extract json_schema with strict=True and additionalProperties=False, then pydantic_model_from_json() to validate the response.
- **Weaker JSON-mode fallback:** response_format={\"type\": \"json_object\"} — guarantees valid JSON but does NOT enforce any specific schema; schema validation must be done client-side. Recommend adding explicit JSON format instructions in the prompt when using this mode.
- **Deprecated patterns:**
  - json_object mode is not formally deprecated but is explicitly described as less reliable than json_schema mode; Mistral docs recommend using custom structured outputs (json_schema) whenever possible instead of json_object
- **Limitations:**
  - codestral-mamba does not support structured outputs (json_schema mode); all other current models are supported
  - Even with json_schema + strict=true, Mistral docs note it is still advisable to iterate on prompts and add clarifying system prompt instructions for optimal reliability
  - The SDK automatically prepends a system prompt injection: 'Your output should be an instance of a JSON object following this schema: {{ json_schema }}' — additional system prompt context is recommended beyond this default
  - json_object mode only guarantees valid JSON, not schema conformance — client-side validation required when using that mode
  - Schema must set additionalProperties: false for strict enforcement; the SDK helper does this automatically
- **Best practices:**
  - Use client.chat.parse() with a Pydantic model as response_format — this is the idiomatic Python SDK pattern
  - Set strict=true and additionalProperties=false in the json_schema object (the SDK does this automatically)
  - Prefer json_schema over json_object whenever you have a known schema
  - Access typed result via .message.parsed (not just .message.content) when using client.chat.parse()
  - Augment the system prompt with additional schema guidance beyond the auto-injected default
  - Use ministral-8b-latest or any current model except codestral-mamba for structured output tasks
- **Sources:** <https://docs.mistral.ai/capabilities/structured_output/custom>, <https://docs.mistral.ai/capabilities/structured_output/json_mode>, <https://docs.mistral.ai/capabilities/structured_output>, <https://platform-docs-public.pages.dev/capabilities/structured-output/custom_structured_output/>, <https://deepwiki.com/mistralai/client-python/4.4-structured-output-with-pydantic>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/mistral_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** MistralSchematicGenerator._do_generate builds a single-message user-role chat request via mistralai SDK's async chat.complete_async. It sets response_format={"type": "json_object"} unconditionally. Hints are filtered to supported_mistral_params=["temperature","max_tokens"] and forwarded as kwargs. The raw string response is first passed through normalize_json_output (strips ```json fences), then json.loads; on JSONDecodeError it falls back to jsonfinder.only_json scavenging. The parsed dict is validated with self.schema.model_validate; ValidationError is logged and re-raised (no retry). Retry is applied only for network/SDK exceptions via the @policy decorator wrapping do_generate.
- **Uses native structured output:** no (response_format json_object — forces JSON mode but does NOT send the schema; the model is prompted for JSON and the output is scraped/validated client-side)
- **Schema sent to API:** no
- **Gating:** None — json_object mode is always on; no "strict" hint gate exists; default is always json_object
- **Parsing fallbacks:** normalize_json_output strips ```json fences (common.py:19-32); jsonfinder.only_json scavenges embedded JSON on JSONDecodeError (mistral_service.py:184)
- **Retries on ValidationError:** no
- **max_tokens handling:** max_tokens is not hardcoded in do_generate; it is passed through hints if the caller includes it (supported_mistral_params includes "max_tokens", mistral_service.py:97). Each model class exposes a max_tokens property (128k) but that property is informational only — it is NOT injected into the API call automatically.
- **Notable issues:**
  - Single user role only — no system message role support; the entire prompt is stuffed into one user message (mistral_service.py:158)
  - json_object mode does not transmit the Pydantic schema to the API — schema enforcement is entirely client-side post-hoc
  - cached_input_tokens is hardcoded to 0 in record_llm_metrics call (mistral_service.py:198) — Mistral prompt cache tokens are never counted
  - assert response.usage (mistral_service.py:190) will raise AssertionError (not retried) if usage is absent — should be a conditional guard
  - MistralEstimatingTokenizer uses GPT-4o tiktoken encoding as a proxy for Mistral tokenization (mistral_service.py:88) — token counts are approximations
  - Mistral_Small_2506 model class is defined but never referenced in get_schematic_generator dispatch (mistral_service.py:243-250 vs 406-413) — dead code
  - ModerationService category parsing uses fragile hasattr duck-typing with nested loops over category_scores (mistral_service.py:349-360) — likely broken against actual Mistral moderation API response shape
- **Key code refs:**
  - `mistral_service.py:157 — self._client.chat.complete_async call site`
  - `mistral_service.py:160 — response_format={"type": "json_object"} unconditional`
  - `mistral_service.py:181 — normalize_json_output applied before json.loads`
  - `mistral_service.py:184 — jsonfinder.only_json fallback on JSONDecodeError`
  - `mistral_service.py:188 — self.schema.model_validate(json_content)`
  - `mistral_service.py:122-133 — @policy retry decorator (ConnectionError, TimeoutError, SDKError, HTTPValidationError only)`
  - `mistral_service.py:214-218 — ValidationError logged and re-raised, no retry`
  - `mistral_service.py:190 — assert response.usage (bare assert, not guarded)`
  - `mistral_service.py:97-98 — supported_hints=["temperature","max_tokens"]`
  - `mistral_service.py:406-413 — get_schematic_generator dispatch (Mistral_Small_2506 absent)`
  - `common.py:19-32 — normalize_json_output strips ```json fences`
  - `common.py:41-83 — record_llm_metrics token counters`

### Gaps

#### [HIGH] json_object mode used instead of json_schema — schema never sent to API

`src/parlant/adapters/nlp/mistral_service.py:160`

Line 160 unconditionally sets response_format={"type": "json_object"}. This guarantees only valid JSON, not schema conformance. The Pydantic model (self.schema) is never forwarded to the API, so the model has no server-side knowledge of required fields, types, or additionalProperties constraints. All schema enforcement is post-hoc client-side model_validate, which fails silently on structurally wrong but JSON-valid responses.

**Recommendation:** Replace the hardcoded json_object dict with the json_schema form. Build it from self.schema at call time:

response_format={
    "type": "json_schema",
    "json_schema": {
        "name": self.schema.__name__,
        "strict": True,
        "schema": {
            **self.schema.model_json_schema(),
            "additionalProperties": False,
        },
    },
}

Alternatively use the SDK helper: replace chat.complete_async with client.chat.parse() passing response_format=self.schema (the Pydantic class directly). Access the typed result via response.choices[0].message.parsed instead of json.loads.

#### [HIGH] No retry on ValidationError — a single bad response is fatal

`src/parlant/adapters/nlp/mistral_service.py:122-133`

At line 214-218, ValidationError is logged and immediately re-raised. The @policy retry decorator at line 122-133 only catches ConnectionError, TimeoutError, SDKError, and HTTPValidationError — not ValidationError. With json_object mode the model can return structurally wrong JSON on any call; there is no second chance. Switching to json_schema reduces (but does not eliminate) this risk, making a retry budget doubly important.

**Recommendation:** Add ValidationError to the retry exception tuple in the @policy decorator, or add a separate inner retry loop in _do_generate around the model_validate call. When using json_schema mode, a retry will re-call the API, giving constrained decoding a second attempt:

@policy([
    retry(
        exceptions=(
            ConnectionError,
            TimeoutError,
            SDKError,
            HTTPValidationError,
            ValidationError,
        ),
    ),
])

Set a bounded max_attempts (e.g. 3) so retries do not loop indefinitely.

#### [MEDIUM] Bare assert on response.usage will raise AssertionError (not retried)

`src/parlant/adapters/nlp/mistral_service.py:190`

Line 190 uses `assert response.usage`. If usage is absent (which the API can return in certain error or streaming-fallback states), this raises AssertionError, which is not in the retry exception list and propagates as an unhandled crash rather than a retried or gracefully degraded failure.

**Recommendation:** Replace the bare assert with a conditional guard:

if not response.usage:
    self.logger.warning("No usage info returned by Mistral API")
    # emit zero-counts or skip metrics
else:
    await record_llm_metrics(..., input_tokens=response.usage.prompt_tokens or 0, ...)

This also removes the implicit dependency between metrics recording and result construction.

#### [MEDIUM] cached_input_tokens hardcoded to 0 — Mistral prompt cache tokens never counted

`src/parlant/adapters/nlp/mistral_service.py:198`

Line 198 passes cached_input_tokens=0 unconditionally. Mistral returns prompt cache hit counts in response.usage (field name: prompt_tokens_details.cached_tokens or similar depending on SDK version). Hardcoding 0 means the cached_input_tokens metric counter is always zero, making it useless for cost tracking.

**Recommendation:** Read the actual cached token count from the usage response. In the mistralai Python SDK, check response.usage for a cached_tokens or prompt_tokens_details attribute and extract it defensively:

cached = getattr(response.usage, "cached_tokens", None) or 0
await record_llm_metrics(..., cached_input_tokens=cached)

Verify the exact field name against the SDK's UsageInfo model before shipping.

#### [MEDIUM] Single user-role message — no system message support; schema prompt guidance absent

`src/parlant/adapters/nlp/mistral_service.py:157-158`

Line 158 stuffs the entire prompt into a single user-role message. Mistral docs recommend augmenting the auto-injected system prompt (which only says 'Your output should be an instance of a JSON object following this schema: ...') with additional schema-specific instructions. Without a dedicated system message, schema guidance competes with the user prompt content and is harder for the model to distinguish.

**Recommendation:** Split the API call into a two-message conversation: inject a system message with schema context, then the user prompt:

messages=[
    {"role": "system", "content": f"You must respond with a valid JSON object conforming to the {self.schema.__name__} schema. Do not include any text outside the JSON object."},
    {"role": "user", "content": prompt},
]

This is consistent with how all other adapters in the codebase structure multi-role prompts and aligns with Mistral's own best-practice guidance for structured output tasks.

#### [LOW] Mistral_Small_2506 defined but never reachable from get_schematic_generator

`src/parlant/adapters/nlp/mistral_service.py:243-250`

Lines 243-250 define the Mistral_Small_2506 class with a valid model name (mistral-small-2506), but lines 406-413 in get_schematic_generator only dispatch to Mistral_Large_2411 and Mistral_Medium_2508. Mistral_Small_2506 is dead code: no caller can instantiate it through the NLPService interface.

**Recommendation:** Either add a dispatch path in get_schematic_generator for workloads that can tolerate a smaller model (e.g. simple classification tasks), or delete the class. If keeping it, document the intended dispatch condition (e.g. a hint key 'model_tier': 'small').


## Together / Fireworks / Cerebras — verdict: **suboptimal**

Three correctness bugs (missing await on Fireworks async call; missing schema_name kwarg causing TypeError on record_llm_metrics in both Fireworks and Cerebras; missing tracer/meter args when constructing specialized Together generators) mean structured output is broken in production for Fireworks and silently wrong for Together. Beyond those, Together AI uses the deprecated json_object mode instead of json_schema, forfeiting schema enforcement. Cerebras silently drops max_tokens hints. jsonfinder scraping is used as a post-processing step even where the API already guarantees valid JSON, adding unnecessary fragility.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** no
- **Mechanism:** All three support OpenAI-compatible response_format with json_schema. Cerebras: constrained decoding at token level when strict:true (guaranteed). Fireworks: constrained decoding via grammar/schema enforcement during generation (guaranteed per docs). Together AI: schema enforcement via response_format but docs do not explicitly claim constrained decoding — output is strongly guided but not documented as token-level guaranteed.
- **Exact API fields:** Together AI: response_format={"type":"json_schema","json_schema":{"name":"<name>","schema":<json_schema_object>}}. Also supports response_format={"type":"regex","pattern":"<regex>"}. Fireworks: response_format={"type":"json_schema","json_schema":{"name":"<name>","schema":<json_schema_object>}} (recommended); response_format={"type":"json_object"} (unschema'd JSON); grammar mode uses separate field (see grammar-mode docs). Cerebras: response_format={"type":"json_schema","json_schema":{"name":"<name>","strict":true,"schema":<json_schema_object>}} — strict:true activates constrained decoding.
- **SDK helpers:** All three: use Pydantic BaseModel.model_json_schema() to generate schema. Cerebras also documents Zod + zodToJsonSchema() for JS. No provider ships a dedicated parse() helper like openai.beta.chat.completions.parse — manual json.loads() of completion.choices[0].message.content is required. Together AI Python SDK is OpenAI-compatible (from together import Together, or openai client pointed at https://api.together.xyz/v1). Fireworks uses openai SDK pointed at https://api.fireworks.ai/inference/v1. Cerebras has first-party SDK: from cerebras.cloud.sdk import Cerebras.
- **Weaker JSON-mode fallback:** response_format={\"type\": \"json_object\"} — supported on all three providers. Produces valid JSON but no schema enforcement. Cerebras docs explicitly recommend against it in favor of json_schema+strict. Fireworks documents it as the weaker \"any valid JSON\" option.
- **Deprecated patterns:**
  - Together AI: older blog (2024) referenced json_object mode with no schema — superseded by json_schema type
  - Fireworks: response_format={"type":"json_object","schema":...} (Pydantic schema inline in json_object type, seen in older blog post) — superseded by dedicated json_schema type
  - Cerebras: json_object mode (JSON Mode) — docs explicitly recommend using json_schema+strict instead whenever possible
- **Limitations:**
  - Together AI: constrained decoding not explicitly documented — schema compliance not token-level guaranteed; truncation (finish_reason=length) yields invalid JSON; models supporting json_schema are a subset of catalog (check serverless/dedicated model pages); no strict: field
  - Fireworks: response_format with json_schema disables reasoning output on reasoning models (DeepSeek R1 etc) — use prompt-embedded schema to preserve both; oneOf composition unsupported; length/size constraints (minLength, maxLength, minItems, maxItems) unsupported; regex patterns unsupported; external $ref URIs unsupported; only in-document $ref works
  - Cerebras: tools and response_format cannot be used simultaneously; root schema must be type:object; every nested object requires additionalProperties:false (Pydantic does not add this automatically — must use model_config or schema customization); max schema length 5000 chars; max nesting depth 10; max 500 properties; recursive schemas unsupported; string pattern/format unsupported; minItems/maxItems unsupported; as of July 21 2026 non-compliant schemas return validation error
  - Together AI supported models (json_schema): Qwen/Qwen3.5-9B, Qwen/Qwen3.5-397B-A17B, Qwen/Qwen3.6-Plus, Qwen/Qwen2.5-7B-Instruct-Turbo, deepseek-ai/DeepSeek-V4-Pro, meta-llama/Llama-3.3-70B-Instruct-Turbo — full list at docs.together.ai/docs/serverless/models
  - Fireworks: supported models not exhaustively listed in structured output docs — kimi-k2p5 used in examples; DeepSeek R1 confirmed via blog; check model catalog for current list
  - Cerebras: only model shown in structured output docs is gpt-oss-120b (Llama-based OSS model)
- **Best practices:**
  - Cerebras (strongest guarantee): use json_schema + strict:true for token-level constrained decoding; explicitly set additionalProperties:false on every object in schema (Pydantic workaround: use model_config = ConfigDict(json_schema_extra={'additionalProperties': False}) or post-process schema); parse with json.loads(completion.choices[0].message.content)
  - Fireworks: use json_schema type (not json_object) for schema enforcement; constrained decoding active during generation; for reasoning models embed schema in prompt instead of response_format to preserve CoT; validate with pydantic model_validate_json() after parsing
  - Together AI: use json_schema type; include explicit JSON instruction in system/user prompt alongside response_format for best compliance; check finish_reason != 'length' before parsing; model must be from structured-output-capable list
  - All providers: generate schema with Pydantic BaseModel.model_json_schema() — cleanest integration; wrap in try/except json.JSONDecodeError since truncation can still produce invalid JSON on Together/Fireworks
  - Fireworks grammar mode: use for non-JSON constrained output (classification enums, custom formats) — separate endpoint field, not response_format
- **Sources:** <https://docs.together.ai/docs/json-mode>, <https://docs.together.ai/docs/serverless/models>, <https://docs.fireworks.ai/structured-responses/structured-response-formatting>, <https://fireworks.ai/blog/constrained-generation-with-reasoning>, <https://inference-docs.cerebras.ai/capabilities/structured-outputs>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/together_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/fireworks_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/cerebras_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** All three adapters call their respective async SDK clients via chat.completions.create with a single user-role message (prompt as one string). Together uses response_format={type: json_object} (no schema). Fireworks and Cerebras use response_format={type: json_schema, json_schema: {schema: self.schema.model_json_schema(), name: ..., strict: True}}. After the response, raw content is passed through normalize_json_output (strips ```json fences) then jsonfinder.only_json to extract the dict, then self.schema.model_validate() to produce the typed result. No hints["strict"] gating — Fireworks and Cerebras always send the schema; Together always uses json_object mode. The Fireworks call is synchronous (self._client.chat.completions.create without await) despite _client being AsyncFireworks.
- **Uses native structured output:** yes (Together: response_format json_object (no schema). Fireworks and Cerebras: response_format json_schema with strict=True hardcoded, sending model_json_schema() in the request body.)
- **Schema sent to API:** yes
- **Gating:** No gating — native structured output (json_schema) is always used on Fireworks and Cerebras, and json_object is always used on Together. No hints["strict"] switch exists. Default: always on for Fireworks/Cerebras, json_object-only for Together.
- **Parsing fallbacks:** normalize_json_output strips ```json / ``` fences (common.py:19-32); jsonfinder.only_json scavenges the first JSON object from the text (all three adapters)
- **Retries on ValidationError:** no
- **max_tokens handling:** max_tokens is a per-class property (hardcoded: Together base=128k, Fireworks models 128k/8192/4096, Cerebras 8B=8192 70B=32k). It is NOT passed to the API automatically — it is only forwarded if the caller includes it in hints and it appears in supported_hints. Together supported_hints includes max_tokens; Cerebras supported_hints only has temperature, so max_tokens hint is silently dropped there. Fireworks supported_hints includes max_tokens.
- **Notable issues:**
  - fireworks_service.py:131 — Fireworks _do_generate calls self._client.chat.completions.create without await despite _client being AsyncFireworks; this likely returns a coroutine, not the response object, causing AttributeError on response.choices
  - together_service.py:433 — get_schematic_generator calls specialized_class(self._logger) missing tracer and meter args (only passes logger), while CustomTogetherAISchematicGenerator correctly passes all three
  - cerebras_service.py:239 — verify_environment error message says 'You\'re using the OpenAI NLP service' (copy-paste bug from openai_service.py)
  - together_service.py:147 — response_format json_object sends NO schema to the API; model is only prompted for JSON, relies entirely on jsonfinder scraping for structure
  - cerebras_service.py:66 — supported_hints only has temperature; max_tokens hint is silently ignored even though all Cerebras models have max_tokens defined
  - record_llm_metrics uses global module-level counter state (_COUNTERS_INITIALIZED flag) — not thread/process-safe and will misattribute if multiple meters are used
  - Fireworks and Cerebras: schema_name is not passed to record_llm_metrics (missing keyword arg) — the function signature requires it but call sites at fireworks_service.py:167 and cerebras_service.py:152 omit it, which would cause a TypeError at runtime
  - Together model classes pin Llama 3.1 variants (8B/70B/405B-Instruct-Turbo) which are superseded; Cerebras Llama3_3_8B class is misnamed (model is llama3.1-8b, not 3.3)
- **Key code refs:**
  - `together_service.py:144-149 — AsyncTogether chat.completions.create with response_format json_object`
  - `together_service.py:159-160 — normalize_json_output + jsonfinder.only_json`
  - `together_service.py:168 — schema.model_validate`
  - `together_service.py:433 — get_schematic_generator passes only logger to specialized class (missing tracer/meter)`
  - `fireworks_service.py:131-143 — missing await on AsyncFireworks.chat.completions.create with json_schema response_format`
  - `fireworks_service.py:156-157 — normalize_json_output + jsonfinder.only_json`
  - `fireworks_service.py:165 — schema.model_validate`
  - `fireworks_service.py:167-172 — record_llm_metrics call missing schema_name kwarg`
  - `cerebras_service.py:112-124 — AsyncCerebras chat.completions.create with json_schema response_format`
  - `cerebras_service.py:141-142 — normalize_json_output + jsonfinder.only_json`
  - `cerebras_service.py:150 — schema.model_validate`
  - `cerebras_service.py:152-157 — record_llm_metrics call missing schema_name kwarg`
  - `cerebras_service.py:239 — wrong service name in error message`
  - `common.py:19-32 — normalize_json_output strips ```json fences`
  - `common.py:41-83 — record_llm_metrics with global counter state`

### Gaps

#### [HIGH] Fireworks: missing await on async client call

`src/parlant/adapters/nlp/fireworks_service.py:131`

fireworks_service.py:131 calls self._client.chat.completions.create without await. _client is AsyncFireworks, so this returns a coroutine object. Subsequent access to response.choices[0] raises AttributeError at runtime, meaning every Fireworks structured-output call fails.

**Recommendation:** Change line 131 to: response = await self._client.chat.completions.create(...). No other change needed — the rest of the method correctly uses response.choices and response.usage.

#### [HIGH] Fireworks and Cerebras: missing schema_name kwarg on record_llm_metrics call causes TypeError

`src/parlant/adapters/nlp/fireworks_service.py:167`

record_llm_metrics signature (common.py:41) requires schema_name as a positional keyword argument. Both call sites at fireworks_service.py:167-172 and cerebras_service.py:152-157 omit it. This raises TypeError at the metrics-recording step — after a successful API response — causing the entire _do_generate call to fail with an exception.

**Recommendation:** Add schema_name=self.schema.__name__ to both call sites: await record_llm_metrics(self.meter, self.model_name, schema_name=self.schema.__name__, input_tokens=..., output_tokens=...). Compare with the correct Together call at together_service.py:170-176.

#### [HIGH] Together: get_schematic_generator passes only logger to specialized class, dropping tracer and meter

`src/parlant/adapters/nlp/together_service.py:433`

together_service.py:433 calls specialized_class(self._logger) with one positional arg. The constructor of all TogetherAISchematicGenerator subclasses (e.g. Llama3_3_70B) requires (logger, tracer, meter). This raises TypeError for any known model. Only the CustomTogetherAISchematicGenerator path (line 436-441) passes all three args correctly.

**Recommendation:** Change line 433 to: return specialized_class(self._logger, self._tracer, self._meter)

#### [HIGH] Together AI: uses deprecated json_object mode instead of json_schema — no schema enforcement

`src/parlant/adapters/nlp/together_service.py:144`

together_service.py:147 sends response_format={"type": "json_object"}, which instructs the model to produce any valid JSON with no schema enforcement. Structured compliance relies entirely on the prompt text and post-hoc jsonfinder scraping. The current docs define json_schema type for Together AI (e.g. for Qwen3.5, Llama-3.3-70B-Instruct-Turbo — the default model). json_object is the deprecated, weaker form.

**Recommendation:** Replace the response_format dict with: response_format={"type": "json_schema", "json_schema": {"name": self.schema.__name__, "schema": self.schema.model_json_schema()}}. Add a JSON instruction in the system/user prompt for best compliance on Together AI (docs recommend belt-and-suspenders prompting since Together does not guarantee token-level constrained decoding). Check finish_reason != 'length' before parsing.

#### [HIGH] Cerebras: model_json_schema() output lacks additionalProperties:false on nested objects — schema rejected at runtime

`src/parlant/adapters/nlp/cerebras_service.py:118`

cerebras_service.py:118 passes self.schema.model_json_schema() directly. Cerebras docs require additionalProperties:false on every nested object in the schema for constrained decoding to activate; as of July 21 2026 non-compliant schemas return a validation error from the API. Pydantic's model_json_schema() does NOT add this field on nested models by default.

**Recommendation:** Post-process the schema before sending it. A recursive helper: def _add_additional_properties(schema): if isinstance(schema, dict): if schema.get('type') == 'object' or 'properties' in schema: schema['additionalProperties'] = False for v in schema.get('properties', {}).values(): _add_additional_properties(v) for v in schema.get('definitions', {}).values(): _add_additional_properties(v) for v in schema.get('$defs', {}).values(): _add_additional_properties(v). Alternatively use model_config = ConfigDict(json_schema_extra={'additionalProperties': False}) on every Pydantic model in the schema hierarchy.

#### [MEDIUM] Cerebras: max_tokens not in supported_hints — hint silently dropped, truncation risk

`src/parlant/adapters/nlp/cerebras_service.py:66`

cerebras_service.py:66 lists supported_hints = ["temperature"] only. If a caller passes hints={"max_tokens": N}, the value is filtered out and never sent to the API. A low default output budget combined with complex schemas increases the risk of truncated JSON, which the constrained decoding cannot recover from.

**Recommendation:** Add 'max_tokens' to supported_hints: supported_hints = ["temperature", "max_tokens"]. The AsyncCerebras client accepts max_tokens as a standard parameter. Ensure each model subclass' max_tokens property reflects the actual model context window to bound the default.

#### [MEDIUM] jsonfinder scraping applied even when API guarantees valid JSON (Fireworks, Cerebras)

`src/parlant/adapters/nlp/fireworks_service.py:156`

Both Fireworks (line 156-157) and Cerebras (lines 141-142) run normalize_json_output + jsonfinder.only_json on the response content even though both providers use json_schema with strict=True and claim constrained decoding (Cerebras: token-level guarantee; Fireworks: grammar enforcement). jsonfinder can silently extract a partial or wrong JSON object if the content contains multiple JSON-like structures, and it inverts the reliability guarantee from the API. It also masks the real failure mode (truncation) by returning a partial dict that then fails model_validate with a confusing error.

**Recommendation:** For Fireworks and Cerebras: replace the jsonfinder block with direct json.loads(raw_content) inside a try/except json.JSONDecodeError. If finish_reason == 'length' is detectable, log a specific truncation error before raising. Keep normalize_json_output only for Together AI where json_object mode may wrap output in fences.

#### [MEDIUM] common.py: global counter state is not thread/process-safe and mis-attributes metrics across Meter instances

`src/parlant/adapters/nlp/common.py:36`

common.py:36-68 uses module-level globals (_COUNTERS_INITIALIZED, _INPUT_TOKENS_COUNTER etc.) initialized from whichever Meter arrives first. If multiple NLPService instances are created with different Meter objects (e.g. in tests, or multi-tenant setups), all subsequent calls reuse counters from the first Meter. _COUNTERS_INITIALIZED is not protected by a lock, so concurrent initialization in async context can create duplicate counters.

**Recommendation:** Move counter creation into a class or use a dict keyed by meter identity: _counters: dict[int, tuple[Counter, Counter, Counter]] = {}. In record_llm_metrics, look up by id(meter) and create if absent. This is safe for asyncio single-threaded use without a lock and correctly isolates counters per Meter instance.

#### [LOW] Cerebras: verify_environment error message says 'OpenAI NLP service' — copy-paste bug

`src/parlant/adapters/nlp/cerebras_service.py:239`

cerebras_service.py:239 reads: "You're using the OpenAI NLP service, but CEREBRAS_API_KEY is not set." This is a copy-paste from openai_service.py and will confuse operators.

**Recommendation:** Change the message to: "You're using the Cerebras NLP service, but CEREBRAS_API_KEY is not set.\nPlease set CEREBRAS_API_KEY in your environment before running Parlant."

#### [LOW] Together and Fireworks: pinned to Llama 3.1 model names that are superseded

`src/parlant/adapters/nlp/fireworks_service.py:197`

Fireworks specialized classes (FireworksLlama3_1_8B, 70B, 405B) pin accounts/fireworks/models/llama-v3p1-*-instruct. Together classes Llama3_1_8B/70B/405B pin meta-llama/Meta-Llama-3.1-*-Instruct-Turbo. Neither family appears in the current Together AI json_schema-capable model list (which now includes Qwen3.5, DeepSeek-V4-Pro, Llama-3.3-70B). The Fireworks default in FireworksService.__init__ also defaults to llama-v3p1-8b-instruct. Requests to retired endpoints return 404 or fall back to degraded modes.

**Recommendation:** Update Together defaults and specialized classes to models confirmed on the json_schema capable list: meta-llama/Llama-3.3-70B-Instruct-Turbo, Qwen/Qwen2.5-7B-Instruct-Turbo. Update Fireworks default to a currently-active model (kimi-k2p5 or current llama-v3 series — verify at fireworks.ai/models). Remove or deprecate the 3.1-specific subclasses or redirect their model_name strings.


## DeepSeek / Qwen / ModelScope — verdict: **suboptimal**

All three adapters use json_object mode correctly as the only available mechanism, but share several reliability and correctness defects: a broken override in ModelScope that bypasses the base-class dispatch and telemetry pipeline; a confirmed getattr dotted-string bug that silently zeros out cached-token metrics on DeepSeek and Qwen; a hardcoded max_tokens=8192 that conflicts with Qwen's documented truncation hazard and creates a potential duplicate-kwarg TypeError at runtime; zero retry on schema ValidationError despite provider docs recommending 2-3 attempts; a dead 'strict' hint that advertises behavior that is never implemented; and no 'json' keyword enforcement in the prompt despite it being a hard API requirement for both DeepSeek and Qwen. None of the adapters transmit the Pydantic schema to the API (which is expected — no provider in this group supports json_schema constrained decoding at the hosted API layer), so that is not a gap.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** no
- **Mechanism:** json_object mode only for DeepSeek and Qwen/DashScope official APIs — both expose response_format={type:json_object} which guarantees syntactically valid JSON but does NOT enforce a user-supplied schema. ModelScope API-Inference is an OpenAI-compatible proxy (base_url https://api-inference.modelscope.cn/v1) that passes requests to the underlying model; no additional structured-output layer is documented. No provider in this group exposes a native json_schema type with constrained decoding in their official hosted API as of mid-2026.
- **Exact API fields:** DeepSeek: POST https://api.deepseek.com/chat/completions — response_format={"type":"json_object"} (also supported: "type":"text"). Prompt must contain the word 'json'. No json_schema type, no strict field. Qwen/DashScope OpenAI-compat: POST https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions — response_format={"type":"json_object"}. Must contain 'json' keyword in messages. Do NOT set max_tokens. DashScope native SDK: dashscope.Generation.call(..., response_format={'type':'json_object'}). ModelScope: POST https://api-inference.modelscope.cn/v1/chat/completions — OpenAI-compatible, response_format={"type":"json_object"} pass-through to the model.
- **SDK helpers:** DeepSeek: Use openai Python SDK with base_url='https://api.deepseek.com'. For Pydantic-validated output use the `instructor` library (pip install instructor): client = instructor.from_provider('deepseek/deepseek-chat', base_url='https://api.deepseek.com') with Mode.TOOLS (default) or Mode.MD_JSON for deepseek-reasoner. Qwen/DashScope: openai Python SDK with base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1' and api_key=DASHSCOPE_API_KEY, or official dashscope SDK (pip install dashscope). Java DashScope SDK 2.18.4+: ResponseFormat.builder().type('json_object').build(). ModelScope: openai Python SDK with base_url='https://api-inference.modelscope.cn/v1' and api_key=MODELSCOPE_API_KEY.
- **Weaker JSON-mode fallback:** All three providers fall back to (and in mid-2026 are limited to) response_format={\"type\":\"json_object\"} — syntactically valid JSON is guaranteed but schema conformance is not. Apply post-generation validation with jsonschema (Python), Ajv (JS), or Everit (Java). For strict schema enforcement use the instructor library with Mode.TOOLS which re-routes schema enforcement through function/tool calling arguments where strict JSON Schema is supported in DeepSeek beta mode.
- **Deprecated patterns:**
  - DeepSeek beta endpoint (https://api.deepseek.com/beta) for FIM/fill-in-middle — unrelated to structured output but worth noting. DeepSeek's strict tool-call schema (tools[].function.strict:true) is the closest to json_schema enforcement and is the preferred path via instructor Mode.TOOLS rather than raw json_object prompting.
  - Qwen thinking-mode models (any model variant with thinking enabled) cannot use response_format at all — this is a hard API error, not a deprecation, but represents a major compatibility break vs. non-thinking calls.
  - DashScope classic SDK call pattern dashscope.Generation.call() is functional but the OpenAI-compatible SDK path is the recommended migration target per Alibaba Cloud 2026 guidance.
- **Limitations:**
  - Neither DeepSeek nor Qwen/DashScope official hosted API supports response_format={type:json_schema} — there is no way to supply a JSON Schema and have the API enforce it via constrained decoding at the hosted API layer.
  - DeepSeek json_object mode may occasionally return empty content — Alibaba Cloud Qwen may similarly return malformed JSON; both require retry logic.
  - Qwen: setting max_tokens together with response_format json_object causes JSON truncation and parse failures — do not set max_tokens when using structured output.
  - All Qwen thinking-mode models (enabled via enable_thinking:true or thinking model variants) are incompatible with response_format — structured output requires switching to a non-thinking model.
  - ModelScope API-Inference has no published structured-output-specific documentation; capabilities depend entirely on the underlying model served and are not guaranteed by the platform.
  - Schema enforcement must be done post-generation by the caller (jsonschema/Pydantic validation + retry) — no provider in this group guarantees schema-valid output at the API level.
  - DeepSeek prompt MUST include the word 'json' (any case) in system or user message when using json_object mode — omitting it returns an API error.
  - Qwen messages MUST contain the word 'JSON' (case-insensitive) — same constraint, same error on omission.
- **Best practices:**
  - Use the instructor library (Mode.TOOLS) for all three providers to get Pydantic-model-enforced structured output via tool/function calling rather than raw json_object mode — this is the closest to guaranteed schema compliance available.
  - Always include the word 'json' in your system prompt when using json_object mode on both DeepSeek and Qwen.
  - Do not set max_tokens when using Qwen structured output — truncation breaks JSON parsing.
  - Validate all json_object responses with jsonschema or Pydantic before passing to downstream systems; implement a retry loop (2-3 attempts) on parse failure.
  - For DeepSeek: check finish_reason — if 'length', the JSON was truncated; increase context or reduce schema complexity.
  - For Qwen with thinking models: call the thinking model first for reasoning, then pass its output to a non-thinking model with response_format to produce clean structured JSON.
  - For local/self-hosted inference of Qwen or DeepSeek models (e.g., via vLLM), use guided_json / structured_outputs parameter for true constrained decoding — this is not available on the hosted APIs but is the production-grade approach when running your own inference.
  - ModelScope API-Inference is best treated as a thin proxy — structure your code around the underlying model's known capabilities rather than relying on ModelScope-level guarantees.
- **Sources:** <https://api-docs.deepseek.com/guides/json_mode>, <https://api-docs.deepseek.com/api/create-chat-completion>, <https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output>, <https://www.alibabacloud.com/help/en/model-studio/json-mode>, <https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope>, <https://python.useinstructor.com/integrations/deepseek/>, <https://modelscope.cn/docs/model-service/API-Inference/intro>, <https://modelscope.cn/docs/model-service/API-Inference/api-provider>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/deepseek_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/qwen_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/modelscope_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** All three adapters use the openai AsyncClient pointed at provider-specific base URLs (DeepSeek: https://api.deepseek.com, Qwen: dashscope intl/domestic, ModelScope: https://api-inference.modelscope.cn/v1). Each _do_generate builds a single-turn request: messages=[{"role": "user", "content": prompt}] (no system role), response_format={"type": "json_object"}, a hardcoded max_tokens, and any provider-specific params filtered from hints. The response content is parsed via normalize_json_output (strips ```json fences) then json.loads; on JSONDecodeError it falls back to jsonfinder.only_json scavenging. The parsed dict is validated with self.schema.model_validate(). On ValidationError the error is logged and re-raised immediately — no retry. ModelScope uniquely uses stream=True and consumes the async generator to reassemble raw_content, then applies the same parse/validate path. Usage tokens for ModelScope are estimated via tiktoken rather than read from the API response.
- **Uses native structured output:** no (response_format json_object (prompt-level JSON mode only; no json_schema / tool-forcing; schema is NOT transmitted to the API))
- **Schema sent to API:** no
- **Gating:** "strict" is listed in supported_hints for DeepSeek and ModelScope (deepseek_service.py:76, modelscope_service.py:77) but is NEVER read or acted upon anywhere in _do_generate — it is a declared-but-dead hint. Default is irrelevant since the code path never branches on it.
- **Parsing fallbacks:** normalize_json_output strips ```json...``` fences (common.py:19-32); jsonfinder.only_json scavenges JSON from arbitrary text on JSONDecodeError (deepseek_service.py:158, qwen_service.py:261, modelscope_service.py:161)
- **Retries on ValidationError:** no
- **max_tokens handling:** Hardcoded at call site: DeepSeek=8192 (deepseek_service.py:144), Qwen=8*1024=8192 (qwen_service.py:246), ModelScope=8192 (modelscope_service.py:147). The hints key \"max_tokens\" IS in supported_*_params for all three, so a caller can override via hints — but only if the caller explicitly passes it; there is no default from hints, and the hardcoded value is always set as a positional kwarg that would be shadowed by **deepseek_api_arguments if \"max_tokens\" is also in hints (potential duplicate-kwarg error if hints[\"max_tokens\"] is supplied alongside the hardcoded argument).
- **Notable issues:**
  - dead 'strict' hint: declared in supported_hints (deepseek_service.py:76, modelscope_service.py:77) but never read — native structured output is never activated regardless of hint value
  - potential duplicate-kwarg TypeError: max_tokens is passed as a hardcoded keyword AND max_tokens is in supported_*_params, so hints["max_tokens"] would produce a duplicate keyword argument error at runtime (deepseek_service.py:135-146, qwen_service.py:240-249, modelscope_service.py:136-149)
  - ModelScope overrides do_generate as 'generate' (modelscope_service.py:120) — the @override decorator is on the wrong method name; BaseSchematicGenerator expects do_generate, so the @policy retry decorator and the base class dispatch won't wrap the actual implementation
  - ModelScope usage tokens are estimated via tiktoken rather than read from the streaming response (modelscope_service.py:167-168) — actual billed tokens are never recorded; cached_input_tokens is hardcoded to 0
  - Qwen _do_generate does not assert response.usage before accessing it (qwen_service.py:270-271) — will raise AttributeError if usage is None, unlike DeepSeek which has an assert (deepseek_service.py:164)
  - cached_input_tokens uses getattr(response, 'usage.prompt_cache_hit_tokens', 0) with a dotted string — this always returns 0, never the actual nested attribute (deepseek_service.py:176, qwen_service.py:273); correct access would be response.usage.prompt_cache_hit_tokens
  - All three tokenizers use tiktoken with 'gpt-4o-2024-08-06' encoding as a proxy for non-OpenAI models — token counts will be inaccurate
  - Single-turn only (no system message role) — all prompt context is packed into a single user message
  - QwenService.get_schematic_generator asserts the model name is in a hardcoded mapping (qwen_service.py:406) — will raise AssertionError (not a descriptive exception) for any QWEN_MODEL value not in {qwen-max, qwen-plus, qwen2.5-72b-instruct}
- **Key code refs:**
  - `deepseek_service.py:140-146 — chat.completions.create call with response_format json_object and hardcoded max_tokens=8192`
  - `deepseek_service.py:75-76 — supported_hints declares 'strict' but it is never consumed`
  - `deepseek_service.py:154-158 — normalize_json_output then jsonfinder fallback`
  - `deepseek_service.py:162-202 — schema.model_validate, ValidationError re-raised immediately`
  - `deepseek_service.py:173-176 — getattr dotted-string bug for cached tokens`
  - `qwen_service.py:243-249 — chat.completions.create, hardcoded max_tokens=8*1024`
  - `qwen_service.py:265-278 — schema.model_validate without usage None-guard`
  - `modelscope_service.py:120 — method named 'generate' instead of 'do_generate' (wrong override)`
  - `modelscope_service.py:141-149 — stream=True with extra_body enable_thinking=False; response_format json_object still set`
  - `modelscope_service.py:152-155 — async streaming reassembly loop`
  - `modelscope_service.py:167-177 — tiktoken-estimated token counts instead of API usage; cached=0 hardcoded`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] ModelScope overrides 'generate' instead of 'do_generate' — base-class telemetry and @override contract are silently broken

`src/parlant/adapters/nlp/modelscope_service.py:120`

BaseSchematicGenerator.generate() calls self.do_generate() and wraps it with request-duration histogram instrumentation and tracer events. ModelScopeSchematicGenerator overrides 'generate' (line 120) rather than 'do_generate', so the @policy/@retry decorators are placed on the same method that BaseSchematicGenerator already provides. The net effect is: (1) the base-class generate() calls do_generate() which is never implemented and will raise NotImplementedError or route to an empty abstract stub; (2) if the class somehow resolves, the retry policy is on the wrong method and will not correctly wrap the inner API call. This is a dispatch-level defect that makes ModelScope structured generation unreliable in production.

**Recommendation:** Rename the method at modelscope_service.py:120 from 'generate' to 'do_generate'. Remove the @override decorator from 'generate' and apply it to 'do_generate'. The @policy decorator should remain on do_generate, matching the DeepSeek pattern at deepseek_service.py:104-119. The base class's generate() will then correctly call do_generate() with histogram wrapping.

#### [HIGH] getattr with dotted string always returns 0 — cached_input_tokens metric is silently wrong on DeepSeek and Qwen

`src/parlant/adapters/nlp/deepseek_service.py:172`

Both deepseek_service.py:172-176 and qwen_service.py:273-277 call getattr(response, 'usage.prompt_cache_hit_tokens', 0). Python's getattr does not resolve dotted attribute paths; it looks for a single attribute literally named 'usage.prompt_cache_hit_tokens' on the response object, which does not exist, so it always returns 0. The correct cached token count (response.usage.prompt_cache_hit_tokens) is never recorded. This silently zeroes all cached-token metrics and makes prompt-cache ROI invisible in billing dashboards.

**Recommendation:** Replace both occurrences with: cached_input_tokens = response.usage.prompt_cache_hit_tokens if response.usage else 0. The DeepSeek OpenAI-compatible response populates usage.prompt_tokens_details.cached_tokens (openai SDK) or usage.prompt_cache_hit_tokens (raw API field). Verify the exact field name from the openai SDK's CompletionUsage model — it may be response.usage.prompt_tokens_details.cached_tokens rather than prompt_cache_hit_tokens.

#### [HIGH] Hardcoded max_tokens=8192 on Qwen causes JSON truncation and can raise TypeError if hints also supply max_tokens

`src/parlant/adapters/nlp/qwen_service.py:246`

Qwen's official docs explicitly state: do NOT set max_tokens when using response_format json_object — truncation breaks JSON parsing. qwen_service.py:246 hardcodes max_tokens=8*1024 on every call. Additionally, 'max_tokens' is in supported_qwen_params (qwen_service.py:181), so if a caller passes hints={'max_tokens': N}, line 248 (**qwen_api_arguments) produces a duplicate keyword argument TypeError at runtime because max_tokens is already passed as a positional keyword argument on line 246. The same duplicate-kwarg risk exists in deepseek_service.py:143 and modelscope_service.py:146.

**Recommendation:** For Qwen: remove the max_tokens=8*1024 hardcoded argument entirely (qwen_service.py:246) and also remove 'max_tokens' from supported_qwen_params to prevent callers from setting it. For DeepSeek and ModelScope: guard the hardcoded max_tokens by only setting it when not already present in hints — e.g., if 'max_tokens' not in deepseek_api_arguments: deepseek_api_arguments['max_tokens'] = 8192 — then pass it via the dict rather than as a separate keyword argument.

#### [HIGH] No retry on schema ValidationError — a single malformed response is a hard failure

`src/parlant/adapters/nlp/deepseek_service.py:198`

Provider docs recommend 2-3 retry attempts on parse/validation failure because json_object mode guarantees syntactically valid JSON but not schema-valid content. All three adapters catch ValidationError and immediately re-raise (deepseek_service.py:198-202, qwen_service.py:299-303, modelscope_service.py:192-197). The existing @policy retry decorator only retries on network/API exceptions (APIConnectionError, RateLimitError, etc.), not on ValidationError. A single off-schema response permanently fails the call.

**Recommendation:** Add ValidationError to the retry exception set in the @policy decorator on do_generate for all three adapters, with max_exceptions=2 and short wait_times=(0.5, 1.0). Example for deepseek_service.py:104-116: add retry(ValidationError, max_exceptions=2, wait_times=(0.5, 1.0)) to the policy list. Alternatively, implement an inner retry loop in _do_generate that catches ValidationError and re-calls the API up to 2 more times before re-raising.

#### [HIGH] Prompt does not guarantee the word 'json' — DeepSeek and Qwen will return an API error when it is absent

`src/parlant/adapters/nlp/deepseek_service.py:141`

Both DeepSeek and Qwen/DashScope require the word 'json' (case-insensitive) to appear somewhere in the system or user messages when response_format={type:json_object} is set. Omitting it returns an API error, not a fallback. The adapters pass raw caller-supplied prompts with no enforcement of this requirement. Since all messages are packed into a single user role, there is no system message safety net. If a caller's prompt happens to not contain the string 'json', every call will fail with a provider API error.

**Recommendation:** In _do_generate for all three adapters, after building the prompt string, add a guard: if 'json' not in prompt.lower(): prompt += '\n\nRespond in JSON format.' Alternatively, add a system message: messages=[{"role": "system", "content": "Always respond with valid JSON."}, {"role": "user", "content": prompt}]. The system-message approach is also more aligned with OpenAI-compatible API best practices.

#### [MEDIUM] Dead 'strict' hint — declared in supported_hints but never read, creating false API surface

`src/parlant/adapters/nlp/deepseek_service.py:76`

Both DeepSeek (deepseek_service.py:76) and ModelScope (modelscope_service.py:77) declare 'strict' in supported_hints but never branch on it anywhere in _do_generate. Callers who pass hints={'strict': True} expecting stronger schema enforcement receive exactly the same behavior as without it. The hint is misleading documentation that implies a capability (e.g., tool-based schema enforcement via instructor Mode.TOOLS) that is not implemented.

**Recommendation:** Either implement the hint — when hints.get('strict') is True, use tool/function-calling to enforce the schema by passing it as a tools=[{"type":"function","function":{"name":"output","parameters":self.schema.model_json_schema()}}] argument, which is the closest DeepSeek supports to constrained schema output — or remove 'strict' from supported_hints entirely to avoid misleading callers.

#### [MEDIUM] Qwen _do_generate accesses response.usage without a None guard — AttributeError on usage-less responses

`src/parlant/adapters/nlp/qwen_service.py:271`

qwen_service.py:271 calls response.usage.prompt_tokens directly after a non-null usage check at line 252-253 (which only logs, not guards). If usage is None, the code reaches line 271 and raises AttributeError: 'NoneType' object has no attribute 'prompt_tokens'. DeepSeek correctly has assert response.usage at deepseek_service.py:164 before accessing usage fields, providing a clear AssertionError rather than an AttributeError deep in the call stack.

**Recommendation:** Add assert response.usage at qwen_service.py before line 271, matching the DeepSeek pattern: assert response.usage (qwen_service.py between lines 265 and 267). This makes the failure mode explicit and consistent across adapters.

#### [MEDIUM] ModelScope token counts are tiktoken estimates, not API actuals — billing metrics are inaccurate

`src/parlant/adapters/nlp/modelscope_service.py:167`

modelscope_service.py:167-168 estimates input and output token counts via tiktoken (gpt-4o-2024-08-06 encoding) instead of reading from the streaming response. ModelScope's OpenAI-compatible API returns usage in the final stream chunk (stream_options={"include_usage": true} must be set, or usage appears in the last chunk's usage field). Using GPT-4o tokenizer for Qwen/DeepSeek-based models served via ModelScope produces systematically wrong counts, and cached_input_tokens is hardcoded to 0.

**Recommendation:** Pass stream_options={"include_usage": True} in the chat.completions.create call (modelscope_service.py:141). Capture the usage object from the last streaming chunk: usage_data = chunk.usage in the async for loop. After the loop, use usage_data.prompt_tokens and usage_data.completion_tokens if usage_data is not None, falling back to tiktoken estimates only when the API returns no usage. Remove the tiktoken estimate path once confirmed working.

#### [MEDIUM] QwenService.get_schematic_generator raises AssertionError for any model not in the hardcoded mapping

`src/parlant/adapters/nlp/qwen_service.py:406`

qwen_service.py:406 uses assert qwen_generator is not None with an f-string message. AssertionError is the wrong exception type for a configuration error (assertions can be stripped with -O) and the mapping only covers three models: qwen-max, qwen-plus, qwen2.5-72b-instruct. Any other value of QWEN_MODEL silently fails at service initialization rather than at startup validation.

**Recommendation:** Replace the assert with: if qwen_generator is None: raise ValueError(f'Unsupported Qwen model: {self.model_name}. Supported models: {list(model_mapping.keys())}') — or add an else-branch in _get_specialized_generator_class that falls back to constructing a generic QwenSchematicGenerator[T] with the configured model name, which would support arbitrary Qwen-compatible models without code changes.

#### [LOW] All three tokenizers use GPT-4o encoding as a proxy for non-OpenAI models

`src/parlant/adapters/nlp/deepseek_service.py:66`

DeepSeekEstimatingTokenizer, QwenEstimatingTokenizer, and ModelScopeEstimatingTokenizer all call tiktoken.encoding_for_model('gpt-4o-2024-08-06'). DeepSeek models (based on DeepSeek-V3/R1 architecture) and Qwen models use different tokenizers with different vocabulary sizes; token counts will be systematically off. This affects context-window utilization checks (max_tokens comparisons) and the fallback token estimates in ModelScope.

**Recommendation:** For DeepSeek: use the deepseek-ai/DeepSeek-V3 tokenizer from HuggingFace (AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')) or accept the approximation and document it. For Qwen: use tiktoken with the cl100k_base encoding (closer to Qwen's vocab) or the qwen-tokenizer package. At minimum, add a comment acknowledging the approximation so future maintainers are not surprised by token-count discrepancies.


## Zhipu GLM — verdict: **suboptimal**

Both adapters use the deprecated json_object mode and transmit no schema to the API. The GLM adapter (glm_service.py) has a dead "strict" hint that implies json_schema support but never implements it, a hardcoded max_tokens=4096 that will raise TypeError at runtime if a caller also passes max_tokens in hints, and a broken getattr call for cache-hit token accounting that always returns 0. The Zhipu adapter (zhipu_service.py) calls the synchronous ZhipuAI SDK client from an async function, blocking the event loop on every generation. Neither adapter retries on ValidationError or sends a system message. Together these gaps mean structured output reliability, throughput, and observability are all significantly below what the provider's current API supports.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** no
- **Mechanism:** response_format with type=json_schema (recommended) or type=json_object (legacy). json_schema passes a named schema object with optional strict flag. The official API reference at docs.bigmodel.cn only documents json_object, but GLM-4.5+/GLM-4.6/GLM-4.7/GLM-5/GLM-5.1 expose json_schema through the OpenAI-compatible endpoint at https://open.bigmodel.cn/api/paas/v4.
- **Exact API fields:** response_format={"type": "json_schema", "json_schema": {"name": "<string, max 64 chars, a-z/A-Z/0-9/_/->", "schema": {<JSON Schema object>}, "strict": true, "description": "<optional string>"}}. For legacy mode: response_format={"type": "json_object"} (requires explicit JSON instruction in the prompt).
- **SDK helpers:** No first-party Pydantic or parse() helper. Use the zhipuai SDK (latest: 2.1.5.20250825) or the OpenAI Python SDK pointed at the compatible base URL (https://open.bigmodel.cn/api/paas/v4 or https://api.z.ai/api/paas/v4). Manually construct response_format dict and call json.loads() on response.choices[0].message.content. No client.beta.chat.completions.parse() equivalent exists in the zhipuai SDK.
- **Weaker JSON-mode fallback:** response_format={"type": "json_object"} — requires the prompt to explicitly instruct JSON output; no schema enforcement; model may produce invalid JSON in edge cases.
- **Deprecated patterns:**
  - response_format={"type": "json_object"} is described as 'an older method' in current docs; json_schema is the recommended replacement for GLM-4.5+/4.6/4.7/5/5.1.
- **Limitations:**
  - constrained decoding (grammar-based hard guarantee) is NOT documented — strict:true encourages strict adherence but the docs do not claim 100% guarantee; one source cites >95% accuracy with json_schema+strict
  - Only a subset of JSON Schema is supported when strict:true (exact subset not enumerated in public docs)
  - json_schema support is surfaced via the OpenAI-compatible endpoint; the Chinese-language BigModel API reference (docs.bigmodel.cn) only documents json_object — the feature gap may differ by endpoint
  - No first-party SDK parse helper or Pydantic auto-schema generation — manual json.loads() required
  - response_format only applies to text chat completion models; vision, audio, CharGLM-4, Emohaa do not support it
  - schema name must match [a-zA-Z0-9_-], max 64 chars
- **Best practices:**
  - Use response_format={"type":"json_schema",...} with strict:true on GLM-4.5+/4.6/4.7/5/5.1 via the OpenAI-compatible endpoint
  - Always include explicit JSON instructions in the system prompt even with json_schema (belt-and-suspenders)
  - Validate the parsed response with jsonschema library post-hoc; don't assume the output is always schema-valid
  - Use the OpenAI Python SDK with base_url='https://open.bigmodel.cn/api/paas/v4' and api_key=<zhipu key> for full json_schema support if the native zhipuai SDK lags behind
  - For Pydantic integration, manually call MyModel.model_json_schema() to generate the schema dict and pass it to json_schema.schema; parse with MyModel.model_validate_json(content) after receiving the response
- **Sources:** <https://docs.z.ai/guides/capabilities/struct-output>, <https://docs.z.ai/api-reference/llm/chat-completion.md>, <https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8>, <https://docs.aimlapi.com/api-references/text-models-llm/zhipu/glm-4.7>, <https://help.apiyi.com/en/glm-4-7-text-structuring-guide-en.html>, <https://pypi.org/project/zhipuai/>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/zhipu_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/glm_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** Both adapters use the same pattern: build a single-turn `[{"role": "user", "content": prompt}]` message list, pass `response_format={"type": "json_object"}` to request JSON mode, call the synchronous (zhipu) or async (glm) chat.completions.create, then parse the text content with json.loads(normalize_json_output(...)) and validate with pydantic model_validate. No schema is sent in the API payload — only JSON-mode is enabled. Supported hints (temperature, max_tokens, top_p for zhipu; temperature, max_tokens for glm) are forwarded as kwargs; unknown hints are silently dropped. GLM also lists "strict" in supported_hints but never reads it.
- **Uses native structured output:** no (response_format json_object (JSON mode only — no schema transmitted))
- **Schema sent to API:** no
- **Gating:** GLM lists "strict" in supported_hints (glm_service.py:159) but the implementation never reads hints["strict"] and always uses the same json_object path regardless. Default: N/A — the gate is declared but dead.
- **Parsing fallbacks:** normalize_json_output strips ```json...``` fences (common.py:19-32); jsonfinder.only_json scavenges JSON from arbitrary text on JSONDecodeError (zhipu_service.py:246, glm_service.py:239)
- **Retries on ValidationError:** no
- **max_tokens handling:** zhipu_service.py: max_tokens forwarded from hints if present, no hardcoded default (zhipu_service.py:212-223). glm_service.py: hardcoded max_tokens=4096 always sent (glm_service.py:224), but hints["max_tokens"] is also forwarded as a kwarg in glm_api_arguments (glm_service.py:218) so if caller passes max_tokens hint it would conflict/override — the hardcoded value is shadowed by any hint value since **glm_api_arguments is spread after the positional keyword.
- **Notable issues:**
  - glm_service.py:159 — 'strict' is declared in supported_hints but never read; the native structured output path promised by the hint name is completely unimplemented (dead declaration)
  - glm_service.py:224 — max_tokens=4096 is hardcoded as a positional keyword; hints['max_tokens'] in glm_api_arguments (spread via **) would silently override it since Python raises TypeError on duplicate kwargs — actually this will raise TypeError at runtime if caller passes max_tokens hint, making the hint unusable
  - glm_service.py:253-257 and 270-274 — cached_input_tokens uses getattr(response, 'usage.prompt_cache_hit_tokens', 0) with a dotted string attribute name, which always returns 0 (getattr does not traverse dotted paths); should be getattr(response.usage, 'prompt_cache_hit_tokens', 0)
  - zhipu_service.py:218-223 — synchronous (blocking) _client.chat.completions.create called from async context; ZhipuAI SDK is sync-only so this blocks the event loop
  - Both files — all prompts are sent as single 'user' role only; no system message is used
  - Both files — ValidationError immediately re-raises without retry; any schema mismatch is terminal
  - glm_service.py:335 — GLMService.get_schematic_generator ignores the schema type t and always returns GLM_4_5 for every schema, unlike ZhipuService which routes specific schemas to GLM_4_Plus
  - zhipu_service.py:231-234 — usage logging uses hasattr guard, but line 253 uses bare assert response.usage; inconsistent null handling can raise AssertionError at runtime if usage is None
- **Key code refs:**
  - `zhipu_service.py:218-223 — ZhipuAI sync chat.completions.create call`
  - `zhipu_service.py:221 — response_format={"type": "json_object"}`
  - `zhipu_service.py:243-247 — normalize_json_output + jsonfinder fallback`
  - `zhipu_service.py:251 — self.schema.model_validate(json_content)`
  - `zhipu_service.py:277-281 — ValidationError caught and re-raised (no retry)`
  - `zhipu_service.py:162-173 — @policy retry decorator (APIConnectionError, APITimeoutError, APIReachLimitError, APIServerFlowExceedError)`
  - `zhipu_service.py:172 — retry(APIInternalError, max_exceptions=2, wait_times=(1.0, 5.0))`
  - `zhipu_service.py:284-350 — model classes GLM_4_Plus (128K), GLM_4_Flash (128K), GLM_4_Air (128K)`
  - `zhipu_service.py:633-638 — ZhipuService.get_schematic_generator routing table`
  - `glm_service.py:157-159 — GLMSchematicGenerator.supported_hints includes 'strict' (dead)`
  - `glm_service.py:218 — glm_api_arguments filter (temperature, max_tokens only)`
  - `glm_service.py:221-227 — openai AsyncClient chat.completions.create with hardcoded max_tokens=4096`
  - `glm_service.py:225 — response_format={"type": "json_object"}`
  - `glm_service.py:236-240 — normalize_json_output + jsonfinder fallback`
  - `glm_service.py:243 — self.schema.model_validate(json_content)`
  - `glm_service.py:279-283 — ValidationError caught and re-raised (no retry)`
  - `glm_service.py:253-257 — broken getattr dotted-path for cached tokens (always 0)`
  - `glm_service.py:286-293 — GLM_4_5 model class (96K context)`
  - `glm_service.py:335 — GLMService always returns GLM_4_5 regardless of schema type`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] json_object used instead of json_schema — no schema transmitted to API

`glm_service.py:225, zhipu_service.py:221`

Both adapters pass response_format={"type": "json_object"} (the deprecated path). GLM-4.5 / GLM-4.6+ support response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": true}} via the OpenAI-compatible endpoint the GLM adapter already targets (https://open.bigmodel.cn/api/paas/v4). Without transmitting the schema, the model has no structural contract to honour — any field name or type deviation falls through to the client-side pydantic parse, which raises ValidationError with no recourse.

**Recommendation:** Replace the response_format dict in both _do_generate methods with: response_format={"type": "json_schema", "json_schema": {"name": self.schema.__name__[:64], "schema": self.schema.model_json_schema(), "strict": True}}. For the zhipu adapter this requires switching to the OpenAI AsyncClient pointed at base_url="https://open.bigmodel.cn/api/paas/v4" (same as the GLM adapter already uses). Keep the prompt-level JSON instruction as belt-and-suspenders.

#### [HIGH] ZhipuAI sync client called from async context — event loop blocked

`zhipu_service.py:136, zhipu_service.py:218`

zhipu_service.py:218 calls self._client.chat.completions.create(...) synchronously. ZhipuAI SDK (zhipuai) only exposes a synchronous client; calling it directly from an async function blocks the entire asyncio event loop for the duration of the network round-trip, serialising all concurrent Parlant coroutines.

**Recommendation:** Replace the zhipuai SDK client with the OpenAI AsyncClient pointed at the compatible base URL: self._client = AsyncClient(base_url="https://open.bigmodel.cn/api/paas/v4", api_key=os.environ["ZHIPUAI_API_KEY"]). This is the same client the GLM adapter already uses. The call becomes: response = await self._client.chat.completions.create(...). Also migrate the ZhipuEmbedder to AsyncClient.embeddings.create.

#### [HIGH] Hardcoded max_tokens=4096 in GLM adapter causes TypeError when hint also passed

`glm_service.py:224, glm_service.py:218, glm_service.py:226`

glm_service.py:224 passes max_tokens=4096 as a keyword argument, then spreads **glm_api_arguments at line 226. glm_api_arguments already includes max_tokens if the caller provided that hint (line 218 filter includes "max_tokens"). Python raises TypeError: got multiple values for keyword argument 'max_tokens' at runtime, making the max_tokens hint completely unusable and crashing generation.

**Recommendation:** Remove the hardcoded max_tokens=4096 positional keyword. If a default is needed, set it only when the hint is absent: the create call should use **{"max_tokens": hints.get("max_tokens", 4096), **{k: v for k, v in hints.items() if k in supported_glm_params and k != "max_tokens"}} or simply let the model use its own default. The GLM-4.5 context window is 96K; 4096 output tokens is an unnecessary constraint that can truncate large schemas.

#### [HIGH] Dead "strict" hint in GLM adapter — native structured output gate never implemented

`glm_service.py:157-159, glm_service.py:218-227`

glm_service.py:159 declares "strict" in supported_hints, implying callers can enable stricter schema enforcement. The value is never read anywhere in _do_generate. The existing json_object path is taken unconditionally regardless of whether strict=True is passed. This is a broken contract: callers believe they are enabling a stricter mode but nothing changes.

**Recommendation:** Either remove "strict" from supported_hints entirely (if json_schema mode is implemented uniformly as recommended above), or read hints.get("strict", True) to gate the json_schema path: use json_schema with strict=True when strict is truthy (the default), and fall back to json_object only when strict=False. Do not leave the hint declared but unread.

#### [MEDIUM] Broken getattr for cache-hit token accounting — always records 0

`glm_service.py:253-257, glm_service.py:270-274`

glm_service.py:253-257 and 270-274 call getattr(response, "usage.prompt_cache_hit_tokens", 0). Python's getattr does not traverse dotted attribute paths; it looks for a single attribute literally named "usage.prompt_cache_hit_tokens" on the response object, which does not exist, so it silently returns 0. Cached token counts are never reported, making the cached_input_tokens metric meaningless.

**Recommendation:** Replace with getattr(response.usage, "prompt_cache_hit_tokens", 0) (two separate attribute accesses). Guard for None usage first: cached = getattr(response.usage, "prompt_cache_hit_tokens", 0) if response.usage else 0.

#### [MEDIUM] No retry on ValidationError — any schema mismatch is terminal

`zhipu_service.py:277-281, glm_service.py:279-283`

Both adapters catch ValidationError, log it, and immediately re-raise (zhipu_service.py:277-281, glm_service.py:279-283). With json_object mode the model can produce structurally valid JSON that fails Pydantic validation (wrong field names, wrong types). Even with json_schema+strict the provider docs note >95% — not 100% — accuracy. A single retry with the same prompt frequently succeeds on transient schema deviation.

**Recommendation:** Wrap the model_validate call in a retry loop (max 2 attempts). On ValidationError, log the failure and re-issue the API call before raising. This is consistent with the existing retry(APIInternalError, max_exceptions=2) pattern already used for network errors. Alternatively integrate ValidationError into the @policy retry decorator at the do_generate level.

#### [MEDIUM] No system message — schema description and JSON instruction conveyed only via user turn

`zhipu_service.py:219, glm_service.py:222`

Both adapters send messages=[{"role": "user", "content": prompt}] with no system message. Provider best practice recommends an explicit JSON instruction even when using json_schema mode. A system message is also the correct place to convey output format requirements, keeping the user-turn content as the task payload. Mixing format instructions into the user prompt reduces prompt clarity and may degrade instruction-following.

**Recommendation:** Add a system message: messages=[{"role": "system", "content": "You are a precise JSON-outputting assistant. Always respond with valid JSON matching the provided schema. Output only the JSON object, no prose."}, {"role": "user", "content": prompt}]. This is especially important as belt-and-suspenders alongside json_schema mode per the provider's best practices.

#### [MEDIUM] GLMService.get_schematic_generator ignores schema type — always returns GLM_4_5

`glm_service.py:335`

glm_service.py:335 returns GLM_4_5[t] unconditionally regardless of the schema type t. ZhipuService (zhipu_service.py:633-638) routes certain heavy schemas (CannedResponseDraftSchema, JourneyBacktrackNodeSelectionSchema) to the higher-capability GLM_4_Plus. The GLM adapter skips this routing entirely, using the same model for all schemas including those that benefit from a stronger model.

**Recommendation:** Mirror the routing table from ZhipuService.get_schematic_generator. At minimum, map the same heavy schemas to a higher-capability model if one is available in the GLM lineup (e.g., a glm-4.6 or glm-5 class), and use GLM_4_5 as the default for lighter schemas.

#### [MEDIUM] Inconsistent usage null-handling in zhipu_service — AssertionError risk

`zhipu_service.py:231, zhipu_service.py:253`

zhipu_service.py:231 guards usage access with hasattr(...) and response.usage truthiness check, but zhipu_service.py:253 then does assert response.usage unconditionally inside the try block. If the API returns a response with usage=None (possible on some error paths that still return a completion), the assert raises AssertionError rather than a handled exception, bypassing all retry and error-reporting logic.

**Recommendation:** Replace assert response.usage with an explicit guard: if response.usage is None: raise RuntimeError(f"Missing usage info from {self.model_name}"). Alternatively, default to zero counts: input_tokens = response.usage.prompt_tokens if response.usage else 0. The GLM adapter already uses the safer if response.usage: pattern at line 230.

#### [LOW] jsonfinder fallback does work the API-level schema mode would eliminate

`zhipu_service.py:244-247, glm_service.py:237-240, common.py:19-32`

Both adapters fall back to jsonfinder.only_json() on JSONDecodeError (zhipu_service.py:246, glm_service.py:239). This scavenges JSON from arbitrary text output — a symptom of the model ignoring the json_object instruction and wrapping output in prose. With json_schema+strict mode enabled, the model is strongly constrained to emit only the JSON object; the fence-stripping in normalize_json_output and the jsonfinder fallback become dead code paths in the common case.

**Recommendation:** After switching to json_schema mode, retain the fallback defensively but log a high-severity warning (not just warning) when it fires, since it should be a near-impossible code path. This makes regressions in the API's schema compliance immediately visible in production logs.


## Ollama — verdict: **good-but-improvable**

The implementation correctly uses Ollama's native constrained-decoding path (`format=schema.model_json_schema()`, `stream=False`) — the most important best practice is already in place. However, several reliability and correctness gaps remain: metrics are silently dropped for all named-model generators (a concrete bug), post-constrained-decoding parsing fallbacks are redundant dead weight, there is no retry on `ValidationError`, the temperature default of 0.3 diverges from the provider's recommended 0 for structured output, and the tokenizer is wired to a GPT-4o encoding for all Llama/Gemma models.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** format parameter accepting a full JSON schema object (or the string "json") passed to /api/chat or /api/generate; uses llama.cpp grammar-based constrained decoding under the hood. The OpenAI-compatible /v1/chat/completions endpoint also accepts response_format={type:json_schema,json_schema:{schema:{...}}} which is internally translated to the same native format field.
- **Exact API fields:** Native API: POST /api/chat or /api/generate with `format: <json-schema-object>` (e.g. format={"type":"object","properties":{...},"required":[...]}). OpenAI-compat API: POST /v1/chat/completions with `response_format={"type":"json_schema","json_schema":{"schema":{...}}}`. Setting `stream: false` is recommended for structured outputs.
- **SDK helpers:** ollama-python: pass `format=MyModel.model_json_schema()` to `ollama.chat()` or `ollama.generate()`; parse response with `MyModel.model_validate_json(response.message.content)`. OpenAI SDK path: `client.beta.chat.completions.parse()` with `response_format=MyModel` works via the /v1/chat/completions endpoint using the json_schema translation layer.
- **Weaker JSON-mode fallback:** Set `format: \"json\"` (native API) or `response_format={\"type\":\"json_object\"}` (OpenAI-compat) for unstructured JSON mode — output is valid JSON but not schema-constrained. Requires prompting the model to produce JSON explicitly.
- **Deprecated patterns:**
  - format: "json" (plain JSON mode) is the older pattern — superseded by format: <json-schema-object> for schema-constrained outputs. The /api/embeddings endpoint is deprecated in favor of /api/embed.
- **Limitations:**
  - Ollama Cloud does not support structured outputs — local deployment only.
  - Constrained decoding guarantees structural schema compliance but does NOT guarantee value accuracy (hallucinations still possible within valid structure).
  - Optional fields may be left unpopulated even when source data contains the relevant information.
  - OpenAI-compat response_format json_schema support (via /v1/chat/completions) was a known gap (issue #10001, opened Mar 2025) but the translation layer (openai/openai.go lines 618-620) was merged — it maps json_schema.schema directly to the native format field.
  - response_format: {type: text} and some advanced OpenAI structured-output features (strict mode, additionalProperties: false enforcement) may not be fully supported.
  - Model must support the schema type — not all quantized/community models behave identically under grammar constraints.
- **Best practices:**
  - Use Pydantic BaseModel + model_json_schema() to generate the schema; validate with model_validate_json() for type-safe results.
  - Set temperature=0 for deterministic, extraction-focused completions.
  - Add Field(description=...) to Pydantic fields — descriptions are included in the JSON schema and measurably improve accuracy on ambiguous fields.
  - Pass stream=False when using structured outputs.
  - Prefer the native ollama-python format= parameter over the OpenAI-compat endpoint when possible — it is the canonical, best-tested path.
  - For codebases already using the openai Python SDK, use response_format={"type":"json_schema","json_schema":{"schema":MyModel.model_json_schema()}} against http://localhost:11434/v1.
- **Sources:** <https://docs.ollama.com/capabilities/structured-outputs>, <https://ollama.com/blog/structured-outputs>, <https://docs.ollama.com/api/openai-compatibility>, <https://github.com/ollama/ollama/blob/main/openai/openai.go>, <https://deepwiki.com/ollama/ollama-python/4.4-structured-outputs>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/ollama_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** OllamaSchematicGenerator._do_generate builds a single-turn request using ollama.AsyncClient.generate() with the prompt as a plain string (no message roles). The Pydantic schema's JSON schema is passed directly as the `format` parameter (line 264: `format=self.schema.model_json_schema()`), which Ollama uses as a grammar/constrained-decoding spec. Generation options (temperature, num_predict, top_p, top_k, repeat_penalty, num_ctx) are passed via the `options` dict. Response is read from response["response"], stripped of ```json fences by normalize_json_output, then scavenged by jsonfinder.only_json, then validated with self.schema.model_validate().
- **Uses native structured output:** yes (ollama.AsyncClient.generate(format=schema.model_json_schema()) — Ollama's native JSON-schema-constrained generation (grammar-based decoding on the server side))
- **Schema sent to API:** yes
- **Gating:** No gating — schema is always sent unconditionally. There is no `strict` hint or any conditional path. Default: always enabled.
- **Parsing fallbacks:** normalize_json_output strips ```json fences (common.py:19-32); jsonfinder.only_json scavenges JSON from the normalized string (ollama_service.py:303)
- **Retries on ValidationError:** no
- **max_tokens handling:** Context window size (num_ctx) is hardcoded per model-size heuristic in the max_tokens property (lines 186-197: 1b→12288, 4b/8b/12b/70b→16384, 27b/405b→32768, default→16384). The hints[\"max_tokens\"] maps to num_predict (output token limit) via _create_options (line 206). num_ctx is always set from self.max_tokens as a default (line 217), never overridable from hints.
- **Notable issues:**
  - Specialized generator classes (OllamaGemma3_1B, etc.) are instantiated without tracer or meter arguments (line 724: `specialized_class(logger=self.logger, base_url=self.base_url)`) — meter and tracer are silently dropped, so metrics are never recorded for specialized generators.
  - Parsing fallbacks (normalize_json_output + jsonfinder) are redundant when Ollama's grammar-constrained format param is used — the model cannot produce fenced or malformed JSON in that mode.
  - ValidationError is caught, logged, and immediately re-raised (lines 338-350) — no retry. The @policy retry decorator (lines 225-233) only catches OllamaConnectionError, OllamaTimeoutError, ollama.ResponseError.
  - max_tokens property uses substring heuristics on model name (e.g., '70b' in name) rather than a proper config — fragile for custom model names like 'my-llama3.1-70b-q4'.
  - tiktoken encoding is hardcoded to gpt-4o-2024-08-06 (line 142) for all Ollama models, which is incorrect for Llama/Gemma tokenizers — token estimates will be inaccurate.
  - OllamaEmbedder reads OLLAMA_BASE_URL from env at __init__ time (line 483) instead of from the base_url passed by OllamaService, so base_url override in OllamaService has no effect on embedders.
  - supported_hints list (line 154) is declared but never enforced — unknown hints are silently ignored by _create_options rather than warned about.
  - MODEL_RECOMMENDATIONS dict at module level (lines 756-764) is dead code — never read at runtime.
- **Key code refs:**
  - `ollama_service.py:260-268 — asyncio.wait_for wrapping ollama.AsyncClient.generate() with format=schema.model_json_schema()`
  - `ollama_service.py:302-303 — normalize_json_output then jsonfinder.only_json parsing`
  - `ollama_service.py:315 — self.schema.model_validate(json_object)`
  - `ollama_service.py:338-350 — ValidationError caught, re-raised (no retry)`
  - `ollama_service.py:225-233 — @policy retry on OllamaConnectionError/OllamaTimeoutError/ResponseError only`
  - `ollama_service.py:186-197 — max_tokens heuristic by model name substring`
  - `ollama_service.py:199-223 — _create_options: hints to Ollama options mapping`
  - `ollama_service.py:724 — specialized_class instantiated without tracer/meter (bug)`
  - `ollama_service.py:142 — tiktoken hardcoded to gpt-4o-2024-08-06`
  - `ollama_service.py:483 — OllamaEmbedder reads OLLAMA_BASE_URL from env, ignores constructor base_url`
  - `common.py:19-32 — normalize_json_output strips ```json fences`
  - `common.py:41-83 — record_llm_metrics with lazy global counter initialization`

### Gaps

#### [HIGH] Specialized generators instantiated without tracer/meter — metrics silently dropped

`ollama_service.py:724`

In `OllamaService.get_schematic_generator` (line 724), the specialized class is called as `specialized_class(logger=self.logger, base_url=self.base_url)`. The `tracer` and `meter` arguments are omitted. Every named-model generator (`OllamaGemma3_1B`, `OllamaGemma3_4B`, etc.) requires those arguments in its `__init__` signature (lines 355-362). Depending on whether `BaseSchematicGenerator.__init__` has defaults, this either silently passes `None` or relies on an uninitialized fallback, meaning `record_llm_metrics` is called with a null/no-op meter for every request that goes through a named model.

**Recommendation:** Pass all required arguments at line 724: `specialized_class(logger=self.logger, tracer=self._tracer, meter=self._meter, base_url=self.base_url)`. This mirrors the `CustomOllamaSchematicGenerator` instantiation on lines 727-733 which does pass them correctly.

#### [HIGH] No retry on ValidationError after constrained-decoding failure

`ollama_service.py:338-350`

Grammar-based constrained decoding guarantees the output is structurally valid JSON matching the schema, but Pydantic `model_validate` can still fail if the model emits a value that is valid JSON/schema-structurally but violates a Pydantic field-level constraint (e.g., wrong enum value, failed validator). The `@policy` retry decorator (lines 225-233) only catches `OllamaConnectionError`, `OllamaTimeoutError`, and `ollama.ResponseError`. `ValidationError` is caught, logged, and immediately re-raised (lines 338-350) — no retry attempt is made. A single retry with a slightly higher temperature often recovers from this class of failure.

**Recommendation:** Add `ValidationError` to the retry policy, or add an inner retry loop in `_do_generate` around the `model_validate` call. Using the existing `@policy` decorator: `retry(exceptions=(OllamaConnectionError, OllamaTimeoutError, ollama.ResponseError, ValidationError), max_exceptions=3, wait_times=(1.0, 2.0, 4.0))`. Alternatively, wrap `model_validate` in a loop that retries the full `_client.generate` call up to N times before re-raising.

#### [MEDIUM] Post-constrained-decoding parsing pipeline is redundant and fragile

`ollama_service.py:302-309`

Lines 302-303 run `normalize_json_output` (strips ```json fences) and then `jsonfinder.only_json` to scavenge JSON from the normalized string. When Ollama's grammar-constrained `format=<json-schema>` is active, the server-side llama.cpp grammar enforcer guarantees the raw response is already clean JSON — no fences, no prose. This parsing pipeline is therefore dead code in the happy path and adds `jsonfinder` as a dependency. If the pipeline does trigger (implying the grammar-constrained output is malformed), `jsonfinder` silently extracts partial JSON, potentially masking a real failure and feeding `model_validate` with corrupted data.

**Recommendation:** Replace lines 302-309 with a direct `json.loads(raw_content)` call. Catch `json.JSONDecodeError` and log/raise it without fallback scavenging: `import json; json_object = json.loads(raw_content)`. The `normalize_json_output` helper and `jsonfinder` import can then be removed from this module.

#### [MEDIUM] Default temperature 0.3 instead of 0 for structured output

`ollama_service.py:214`

The provider's best-practice documentation explicitly states: 'Set temperature=0 for deterministic, extraction-focused completions.' The `_create_options` method sets `options.setdefault('temperature', 0.3)` (line 214). This increases non-determinism in structured extraction tasks, raising the likelihood that constrained decoding produces an unexpected-but-schema-valid value.

**Recommendation:** Change line 214 from `options.setdefault('temperature', 0.3)` to `options.setdefault('temperature', 0)`. The existing model-specific override for `1b` (line 219) that sets `temperature=0.1` should similarly be changed to `0`.

#### [MEDIUM] tiktoken hardcoded to GPT-4o encoding for all Ollama models

`ollama_service.py:142`

`OllamaEstimatingTokenizer.__init__` (line 142) calls `tiktoken.encoding_for_model('gpt-4o-2024-08-06')` for every Ollama model including Llama 3.1 and Gemma 3. These models use SentencePiece/BPE tokenizers with different vocabularies. The 1.15× fudge factor (line 148) partially compensates but is not reliable. Underestimates cause `num_ctx` to be set too low, which truncates JSON mid-output and produces parse failures.

**Recommendation:** Use `tiktoken.get_encoding('cl100k_base')` as a closer approximation for Llama models (same base as GPT-4), or increase the fudge factor to at least 1.3 for Llama/Gemma families. Longer term, replace with the `transformers` tokenizer for accuracy: `from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(hf_model_id)`. For immediate safety, raise the multiplier at line 148 from `1.15` to `1.35`.

#### [MEDIUM] OllamaEmbedder ignores constructor base_url, reads env var directly

`ollama_service.py:483`

`OllamaEmbedder.__init__` (line 483) reads `OLLAMA_BASE_URL` from the environment instead of accepting `base_url` from the caller. `OllamaService.get_embedder` (lines 739-747) constructs embedder instances without passing `base_url`, so `OllamaService.base_url` (set from env at line 653) is never forwarded. This is redundant now, but becomes a bug if a caller constructs `OllamaService` with a programmatic `base_url` override — the embedder will silently use a different endpoint than the generator.

**Recommendation:** Add `base_url: str = 'http://localhost:11434'` to `OllamaEmbedder.__init__`, remove the `os.environ.get` call on line 483, and set `self.base_url = base_url.rstrip('/')`. Update `OllamaService.get_embedder` to pass `base_url=self.base_url` to each embedder constructor.

#### [LOW] max_tokens context window sized by fragile substring heuristics

`ollama_service.py:186-197`

The `max_tokens` property (lines 186-197) maps model names to context window sizes using `'70b' in self.model_name.lower()`. A model named `my-llama3.1-70b-quantized-q4_K_M` will match correctly, but a model named `llama3.3-70b-instruct` and one named `model70b` are treated identically even though their actual context windows differ. The same num_ctx is also used for both input context and output limit (line 217 vs. line 206), meaning `num_predict` and `num_ctx` are conflated.

**Recommendation:** Separate `max_context_tokens` (for `num_ctx`) from `max_output_tokens` (for `num_predict`). Allow callers to pass `num_ctx` as an explicit hint rather than hardcoding it. For the heuristic itself, prefer checking model name prefixes or an explicit config dict over scattered `in` substring checks.

#### [LOW] MODEL_RECOMMENDATIONS dict is dead code

`ollama_service.py:756-764`

The `MODEL_RECOMMENDATIONS` dict (lines 756-764) is defined at module level but never referenced anywhere in the module or (per the audit) elsewhere in the codebase. It cannot influence runtime behavior.

**Recommendation:** Either wire it into `_log_model_warnings` so the descriptions are emitted as structured log entries, or delete it to reduce maintenance surface.


## OpenRouter / LiteLLM — verdict: **suboptimal**

Both adapters use json_object mode (JSON mode only) with no schema transmitted to the API. This means the provider applies zero schema enforcement — any structurally invalid response that passes json.loads() reaches Pydantic validation, and failures raise immediately with no retry. Several additional correctness bugs compound the problem: cached token metrics always read 0 due to a dotted-string getattr bug, the litellm adapter hardcodes max_tokens=5000 ignoring hints, the "strict" hint is dead code, and the openrouter adapter omits record_llm_metrics entirely. Native json_schema mode with strict:true is directly available on the models already targeted (gpt-4o, claude-3.5-sonnet) and would eliminate the entire normalize/jsonfinder/fallback pipeline for those providers.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** no
- **Mechanism:** OpenRouter: response_format with type=json_schema passes schema to underlying provider; for capable providers (OpenAI GPT-4o+, Gemini, Anthropic Claude Sonnet 4.5+/Opus 4.1+, Fireworks models) this triggers native constrained decoding on the backend provider — not on OpenRouter itself. OpenRouter is a routing layer; schema enforcement fidelity depends on the downstream provider. LiteLLM: passes response_format json_schema through to the underlying provider when supports_response_schema() returns true; for OpenRouter via LiteLLM, the openrouter/ prefix historically stripped the schema — workaround is extra_body passthrough.
- **Exact API fields:** OpenRouter direct API:
{
  "model": "<model-id>",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "schema_name",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": { ... },
        "required": [...],
        "additionalProperties": false
      }
    }
  },
  "provider": {
    "require_parameters": true
  },
  "plugins": [{ "id": "response-healing" }]  // optional, non-streaming only
}

LiteLLM native (for supported providers):
litellm.completion(
  model="openai/gpt-4o",
  response_format=MyPydanticModel,  # or raw json_schema dict
  messages=[...]
)

LiteLLM + OpenRouter workaround (extra_body):
litellm.completion(
  model="openrouter/google/gemini-2.0-flash",
  messages=[...],
  extra_body={
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "response",
        "strict": True,
        "schema": your_schema
      }
    }
  }
)
- **SDK helpers:** OpenRouter: use instructor library — instructor.from_provider('openrouter/<model>', base_url='https://openrouter.ai/api/v1'); pass response_model=MyPydanticModel + extra_body={'provider': {'require_parameters': True}}. LiteLLM: pass Pydantic BaseModel directly as response_format; validate with MyModel.model_validate_json(resp.choices[0].message.content). For LiteLLM proxy with OpenAI SDK: client.beta.chat.completions.parse(model=..., response_format=MyPydanticModel). Check support: litellm.supports_response_schema(model='...', custom_llm_provider='...') and litellm.get_supported_openai_params(model='...'). Enable client-side validation: litellm.enable_json_schema_validation = True.
- **Weaker JSON-mode fallback:** Set response_format={"type": "json_object"} — guarantees valid JSON but not schema adherence. Must also instruct the model to produce JSON via system/user message. In LiteLLM, used automatically when supports_response_schema() returns False for the target model.
- **Deprecated patterns:**
  - OpenRouter: plain json_object mode is the older/weaker pattern superseded by json_schema with strict:true
  - LiteLLM: passing response_format as a raw dict with json_schema to openrouter/* models without extra_body — the OpenrouterConfig.map_openai_params() adapter historically strips/rewrites the schema; this behavior may persist in older LiteLLM versions
  - instructor MODES.JSON (json_object fallback) is superseded by MODES.JSON_SCHEMA where supported
- **Limitations:**
  - OpenRouter is a routing proxy — constrained decoding (grammar-based schema enforcement) is only as strong as the downstream provider implements it; OpenRouter itself does not apply constrained decoding
  - Response Healing plugin only works for non-streaming requests; it fixes JSON syntax errors (missing brackets, commas, etc.) but does NOT guarantee schema adherence — wrong field names or missing required properties are not caught
  - LiteLLM OpenRouter adapter (openrouter/* prefix) has a known bug where supports_response_schema() returns False and the adapter strips response_format schemas — workaround: use extra_body to bypass
  - Not all models on OpenRouter support json_schema; must filter by supported_parameters=structured_outputs on OpenRouter models page or set provider.require_parameters=true to avoid routing to incapable providers
  - Streaming + json_schema: OpenRouter states it delivers 'valid partial JSON' progressively but full schema validation only applies to the complete response
  - Some schemas unsupported: OpenAI GPT-4o does not support regex pattern constraints in schemas; Gemini models typically do
  - Gemini 1.5 uses OpenAPI-style responseSchema (different format); Gemini 2.0+ uses native responseJsonSchema — LiteLLM handles translation but older Vertex models may not support schema passthrough
  - litellm.enable_json_schema_validation=True adds client-side jsonschema validation AFTER the fact — does not prevent schema-invalid responses from being generated, only raises an error when one is received
- **Best practices:**
  - Always set strict: true in json_schema.json_schema for OpenRouter requests
  - Set provider.require_parameters: true in the OpenRouter request body to ensure only structured-output-capable providers receive the request
  - For OpenRouter via LiteLLM: use extra_body to pass response_format directly, bypassing the adapter's schema-stripping behavior until officially fixed
  - Use instructor library with from_provider() for cleanest Pydantic integration with OpenRouter
  - Filter model selection on OpenRouter's models page using supported_parameters=structured_outputs before hardcoding a model
  - For non-streaming requests, enable the response-healing plugin (plugins: [{id: 'response-healing'}]) as a cheap safety net for syntax errors
  - Check support programmatically: litellm.supports_response_schema(model, custom_llm_provider) and litellm.get_supported_openai_params(model) before sending json_schema requests
  - Always set additionalProperties: false in the schema to maximize compliance across providers
  - For LiteLLM proxy + OpenAI SDK, use client.beta.chat.completions.parse() with Pydantic models for the cleanest integration
  - Do not rely on Response Healing to fix schema-level errors — validate the parsed response in application code
- **Sources:** <https://openrouter.ai/docs/guides/features/structured-outputs>, <https://openrouter.ai/docs/api/reference/parameters>, <https://openrouter.ai/docs/guides/routing/provider-selection>, <https://openrouter.ai/docs/guides/features/plugins/response-healing>, <https://docs.litellm.ai/docs/completion/json_mode>, <https://docs.litellm.ai/docs/providers/openrouter>, <https://python.useinstructor.com/integrations/openrouter/>, <https://github.com/BerriAI/litellm/discussions/11652>, <https://openrouter.ai/announcements/response-healing-reduce-json-defects-by-80percent>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/openrouter_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/litellm_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** Both adapters call chat completions with `response_format={"type": "json_object"}` (JSON mode), parse the raw string through normalize_json_output + json.loads, then validate with pydantic schema.model_validate(). Neither sends the full JSON schema to the API. The openrouter adapter adds a BadRequestError fallback that retries without json_object mode using a system-prompt instruction. The litellm adapter has no such fallback and hardcodes max_tokens=5000 at the call site regardless of what the LiteLLM_Default.max_tokens property returns.
- **Uses native structured output:** no (response_format json_object (JSON mode only — no schema sent, no tool-forcing, no beta.parse))
- **Schema sent to API:** no
- **Gating:** The "strict" key is listed in LiteLLMSchematicGenerator.supported_hints (litellm_service.py:86) but is never read inside do_generate — it is silently ignored. No native structured output path exists, so no gating actually occurs. Default is irrelevant because the feature is dead code.
- **Parsing fallbacks:** normalize_json_output strips ```json fences (common.py:19-32); jsonfinder.only_json scavenges JSON from arbitrary text (openrouter_service.py:265, 254; litellm_service.py:155); openrouter: empty/blank response falls back to json_content={} before schema validation (openrouter_service.py:243)
- **Retries on ValidationError:** no
- **max_tokens handling:** openrouter: max_tokens property is defined per subclass (8192 or 128*1024) and also configurable via OPENROUTER_MAX_TOKENS env var for dynamic models, but is NOT passed to the API call — only hints-provided max_tokens flows through (openrouter_service.py:160-162, 171-176). litellm: hardcoded max_tokens=5000 at call site (litellm_service.py:137) overrides any hints-provided value; LiteLLM_Default.max_tokens property (litellm_service.py:215) returns 5000 but is also never forwarded to the API.
- **Notable issues:**
  - litellm_service.py:86 — 'strict' is in supported_hints but never consumed in do_generate; dead code
  - litellm_service.py:137 — max_tokens=5000 hardcoded at the acompletion call site, silently overrides any hint-supplied value and ignores the max_tokens property
  - litellm_service.py:169-173 — cached_input_tokens uses getattr(response, 'usage.prompt_cache_hit_tokens', 0) with a dotted string key, which will always return 0 (wrong attribute access pattern); same bug repeated at line 186-189
  - openrouter_service.py:280 — assert response.usage (bare assert, not a proper guard); will raise AssertionError not a descriptive exception if usage is None
  - openrouter_service.py:160-162 — the openrouter adapter never forwards its own max_tokens property to the API; only hints can set it, so the property is effectively cosmetic
  - openrouter_service.py:334 — OpenRouterClaude35Sonnet pins 'anthropic/claude-3.5-sonnet' (no version suffix), which will silently follow OpenRouter's latest alias; not pinned to a specific model version
  - Neither adapter calls record_llm_metrics (openrouter_service.py does NOT import it); only litellm_service uses it, making token metric coverage inconsistent across adapters
  - litellm_service.py:99 — self._client = litellm (module-level object used as client), meaning litellm global state (e.g. litellm.set_verbose) affects all instances
  - Both adapters use a single user-role message for the entire prompt with no system role, which may degrade instruction-following on some models
  - openrouter_service.py:189 — JSON fallback system message is injected as role=system only in the BadRequestError retry path, not the primary path
- **Key code refs:**
  - `openrouter_service.py:171 — primary acompletion call with response_format json_object`
  - `openrouter_service.py:187-197 — BadRequestError fallback: retry without json_object, add system prompt`
  - `openrouter_service.py:246 — normalize_json_output + json.loads`
  - `openrouter_service.py:254-256 — jsonfinder.only_json on empty parsed JSON`
  - `openrouter_service.py:265 — jsonfinder.only_json on JSONDecodeError path`
  - `openrouter_service.py:278 — self.schema.model_validate(json_content)`
  - `openrouter_service.py:301-311 — ValidationError caught, logged, re-raised (no retry)`
  - `litellm_service.py:132-140 — litellm.acompletion call with hardcoded max_tokens=5000 and response_format json_object`
  - `litellm_service.py:150-156 — normalize_json_output + json.loads + jsonfinder fallback`
  - `litellm_service.py:159 — self.schema.model_validate(json_content)`
  - `litellm_service.py:194-198 — ValidationError caught, logged, re-raised (no retry)`
  - `litellm_service.py:86 — supported_hints includes 'strict' but it is never read`
  - `litellm_service.py:169 — getattr(response, 'usage.prompt_cache_hit_tokens', 0) dotted-string bug`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] Schema never transmitted to API — json_object mode used instead of json_schema

`src/parlant/adapters/nlp/openrouter_service.py:174`

Both adapters call the API with response_format={"type": "json_object"} (lines openrouter_service.py:174, litellm_service.py:138). This tells the model to produce valid JSON but does NOT constrain it to the Pydantic schema. The schema is only applied client-side via model_validate() after the fact. For gpt-4o and claude-sonnet-4.5+, OpenRouter supports response_format with type=json_schema and strict=true, which routes constrained schema decoding to the downstream provider. LiteLLM supports passing a Pydantic model directly as response_format when litellm.supports_response_schema() returns true.

**Recommendation:** For the openrouter adapter, replace the response_format argument in the _client.chat.completions.create() call (openrouter_service.py:174) with:

  response_format={
    "type": "json_schema",
    "json_schema": {
      "name": self.schema.__name__,
      "strict": True,
      "schema": self.schema.model_json_schema()
    }
  }

Also add to the request body:
  extra_body={"provider": {"require_parameters": True}}

For the litellm adapter (litellm_service.py:138), first check support and then pass the model:
  if litellm.supports_response_schema(model=self.model_name):
      response_format = self.schema  # Pydantic BaseModel class
  else:
      response_format = {"type": "json_object"}

For openrouter/* models via LiteLLM, use extra_body to bypass the adapter's schema-stripping:
  extra_body={"response_format": {"type": "json_schema", "json_schema": {"name": self.schema.__name__, "strict": True, "schema": self.schema.model_json_schema()}}}

Add additionalProperties: false to the generated schema before sending.

#### [HIGH] Hardcoded max_tokens=5000 in litellm adapter truncates large JSON responses

`src/parlant/adapters/nlp/litellm_service.py:137`

litellm_service.py:137 passes max_tokens=5000 unconditionally to acompletion(). This is a keyword argument that appears before **litellm_api_arguments (line 139), so a hints-supplied max_tokens would override it — but only by accident of dict unpacking order. The LiteLLM_Default.max_tokens property (line 215) also returns 5000 and is never read. Parlant schemas for agent reasoning can easily exceed 5000 output tokens. A truncated JSON payload will fail json.loads() and fall through to jsonfinder, which may silently return a partial object that passes model_validate() with wrong values.

**Recommendation:** Remove the hardcoded max_tokens=5000 from the acompletion() call (litellm_service.py:137) and instead forward self.max_tokens as the default, letting hints override:

  effective_max_tokens = hints.get("max_tokens", self.max_tokens)
  response = await self._client.acompletion(
      ...,
      max_tokens=effective_max_tokens,
      ...
  )

Also remove the redundant max_tokens=5000 from supported_litellm_params filtering logic so it cannot be silently shadowed.

#### [HIGH] Dotted-string getattr bug causes cached_input_tokens to always return 0

`src/parlant/adapters/nlp/litellm_service.py:169`

litellm_service.py:169 and 186 both call getattr(response, "usage.prompt_cache_hit_tokens", 0). Python's getattr does not traverse dotted paths — it looks for an attribute literally named "usage.prompt_cache_hit_tokens" on the response object, which does not exist, so it always returns the default 0. The same pattern appears in the GenerationInfo.extra dict at line 186. This means cached token metrics are silently zeroed out for all LiteLLM completions.

**Recommendation:** Replace both occurrences with chained attribute access:

  cached = getattr(getattr(response, "usage", None), "prompt_cache_hit_tokens", 0) or 0

Apply at litellm_service.py:169 (record_llm_metrics call) and litellm_service.py:186 (GenerationInfo.extra). The same pattern exists in openrouter_service.py:293-296 but uses response.usage.prompt_cache_hit_tokens directly after the assert response.usage guard, which is correct — only the LiteLLM adapter has this bug.

#### [MEDIUM] 'strict' hint in supported_hints is dead code — never consumed

`src/parlant/adapters/nlp/litellm_service.py:86`

LiteLLMSchematicGenerator.supported_hints (litellm_service.py:86) lists 'strict', but do_generate() filters hints using supported_litellm_params (line 122-124), which does not include 'strict'. The key is therefore read from hints by neither the filter nor any explicit branch. There is no native structured output path, so the hint has no effect regardless. Callers that pass hints={"strict": True} expecting schema-level enforcement receive silent json_object mode instead.

**Recommendation:** Once gap #1 is fixed (json_schema mode), implement the strict hint as the gate:

  use_native_schema = hints.get("strict", False) or litellm.supports_response_schema(self.model_name)

If not implementing native schema now, remove 'strict' from supported_hints entirely to avoid misleading callers. At minimum add a warning log when strict=True is received but cannot be honored.

#### [MEDIUM] No retry on ValidationError — a single schema-invalid response is a fatal failure

`src/parlant/adapters/nlp/openrouter_service.py:301`

Both adapters catch ValidationError (openrouter_service.py:301, litellm_service.py:194) and immediately re-raise after logging. With json_object mode providing no schema guarantee, schema validation failures are more likely, especially on less capable models. The existing @policy retry decorator only covers network/API errors (APIConnectionError, RateLimitError, InternalServerError), not validation failures. For json_schema mode, a retry with temperature lowered or the schema re-injected as a system message would recover most cases.

**Recommendation:** Add a retry loop around the completion+validation cycle. Define a custom exception (e.g. SchemaValidationError) and add it to the @policy retry list, or implement a manual retry in do_generate:

  for attempt in range(max_retries):
      raw = await self._call_api(...)
      try:
          return self.schema.model_validate(json.loads(raw))
      except (ValidationError, json.JSONDecodeError):
          if attempt == max_retries - 1:
              raise
          # optionally lower temperature on retry

Two retries (max_retries=3 total) cover the majority of transient schema failures without significant cost increase.

#### [MEDIUM] openrouter adapter does not call record_llm_metrics — token metrics inconsistently tracked

`src/parlant/adapters/nlp/openrouter_service.py:282`

litellm_service.py imports and calls record_llm_metrics (lines 28, 162-173), but openrouter_service.py never imports or calls it (openrouter_service.py has no reference to record_llm_metrics). Token usage is logged via self._logger.trace() (line 230) but not emitted as meter counters. This means OpenRouter completions produce no input_tokens, output_tokens, or cached_input_tokens counter increments, making cross-adapter usage dashboards incomplete.

**Recommendation:** Import record_llm_metrics in openrouter_service.py:
  from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics

Then call it after model_validate() succeeds (after line 300), matching the pattern in litellm_service.py:162-173:
  await record_llm_metrics(
      self.meter,
      self.model_name,
      schema_name=self.schema.__name__,
      input_tokens=response.usage.prompt_tokens,
      output_tokens=response.usage.completion_tokens,
      cached_input_tokens=getattr(response.usage, "prompt_cache_hit_tokens", 0),
  )

#### [MEDIUM] max_tokens property defined on openrouter generators but never forwarded to the API

`src/parlant/adapters/nlp/openrouter_service.py:160`

OpenRouterSchematicGenerator.max_tokens returns 8192 (line 134); subclasses override it to 128*1024 for GPT-4o models (line 321). But do_generate() only forwards what comes in through hints (line 160-162): k in self.supported_openrouter_params. If no caller passes max_tokens in hints, the property value is ignored and OpenRouter uses its own default. For models with 128k output limits, this means the effective ceiling is whatever OpenRouter defaults to, not what the subclass advertises.

**Recommendation:** In do_generate(), use self.max_tokens as the default for the max_tokens API argument, overridable by hints:

  openrouter_api_arguments = {
      k: v for k, v in hints.items() if k in self.supported_openrouter_params
  }
  openrouter_api_arguments.setdefault("max_tokens", self.max_tokens)

This makes the property meaningful rather than cosmetic.

#### [MEDIUM] OpenRouterClaude35Sonnet uses unversioned alias — silently follows OpenRouter's latest model

`src/parlant/adapters/nlp/openrouter_service.py:337`

OpenRouterClaude35Sonnet hardcodes model_name="anthropic/claude-3.5-sonnet" (openrouter_service.py:337). OpenRouter's unversioned aliases follow the latest available model. Claude 3.5 Sonnet has multiple point releases (20240620, 20241022) with different capability and pricing. An OpenRouter alias update can silently change behavior mid-deployment, complicating debugging and cost attribution.

**Recommendation:** Pin to a specific versioned model ID:
  model_name="anthropic/claude-3.5-sonnet:20241022"

Or use an environment variable override so pinning can be changed without code deployment. OpenRouter's model IDs with version suffixes (e.g. :20241022) are stable and do not follow aliases.

#### [LOW] bare assert on response.usage will raise AssertionError instead of a descriptive exception

`src/parlant/adapters/nlp/openrouter_service.py:280`

openrouter_service.py:280 uses `assert response.usage` to guard usage attribute access. Python asserts are disabled when running with -O (optimized mode) and raise a non-descriptive AssertionError rather than a domain exception. The same pattern appears in litellm_service.py:160.

**Recommendation:** Replace the bare assert with an explicit guard:
  if response.usage is None:
      raise ValueError(f"OpenRouter response for model '{self.model_name}' returned no usage data")

This survives -O mode and produces an actionable error message.

#### [LOW] litellm module used as client instance — global litellm state shared across all generator instances

`src/parlant/adapters/nlp/litellm_service.py:99`

LiteLLMSchematicGenerator.__init__ sets self._client = litellm (litellm_service.py:99), storing the module object itself as the client. This means any global mutation (e.g. litellm.set_verbose = True, litellm.success_callback = [...]) affects all instances simultaneously. In a multi-tenant or multi-model deployment this creates hidden coupling between generator instances.

**Recommendation:** Create an isolated client using LiteLLM's Router or by using the litellm.completion function directly without aliasing the module. If module-level usage is intentional, document the shared-state implication explicitly. Alternatively, avoid mutating litellm global state in tests by using litellm's mock_completion utilities instead.


## Snowflake Cortex — verdict: **suboptimal**

The adapter has partial structured-output support: it can send a JSON schema to the API, but only when callers opt in via hints["strict"]=True (default False), schema-export failures are silently swallowed so the call degrades to prompt-only JSON with no signal, and the REST endpoint used (/api/v2/cortex/inference:complete) differs from the OpenAI-compatible endpoint documented for json_schema-style structured output. The parsing layer does unnecessary work that constrained decoding would eliminate, and several metrics/observability gaps exist.

### Provider best practice (June 2026 docs)

- **Constrained decoding (guaranteed schema-valid):** yes
- **Mechanism:** Token-level validation against a user-supplied JSON schema via the response_format field. Snowflake verifies each generated token against the schema definition to guarantee conformance — this is constrained decoding, not post-processing.
- **Exact API fields:** SQL/Python (AI_COMPLETE / snowflake-ml-python): response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {...}, 'required': [...]}}. REST API / OpenAI-compatible endpoint (POST https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions): response_format={'type': 'json_schema', 'json_schema': {'name': '<name>', 'schema': {'type': 'object', 'properties': {...}, 'required': [...]}}}. Messages API (Anthropic-compatible): uses output_config parameter with identical JSON schema body.
- **SDK helpers:** snowflake-ml-python >= 1.8.0: pass response_format inside CompleteOptions to snowflake.cortex.complete(). Pydantic integration: call MyModel.model_json_schema() and pass the result as the schema value. OpenAI Python SDK also works against the REST endpoint using base_url pointed at the Snowflake account and a Snowflake PAT as api_key.
- **Weaker JSON-mode fallback:** Set response_format={'type': 'json_object'} on OpenAI-compatible models (non-Claude) to request JSON output without a strict schema. Output is encouraged but NOT guaranteed to match any particular structure.
- **Deprecated patterns:**
  - SNOWFLAKE.CORTEX.COMPLETE (legacy SQL function) is deprecated and will be removed by end of 2026. Migrate to AI_COMPLETE (the new AI SQL interface).
  - Type-literal syntax (TYPE OBJECT(...)) in response_format is the older SQL-only form; JSON schema form works across SQL, Python SDK, and REST API and is preferred for new code.
- **Limitations:**
  - All models support structured output, but output quality varies — more capable models (e.g. claude-sonnet-4-6, llama3.3-70b) produce better results.
  - Top-level schema type must be 'object'.
  - Property names: letters, digits, hyphens, underscores only; max 64 characters.
  - Unsupported JSON Schema keywords: minLength, maxLength, format, multipleOf, minimum, maximum, exclusiveMinimum, exclusiveMaximum, uniqueItems, contains, minContains, maxContains, minItems, maxItems, patternProperties, minProperties, maxProperties, propertyNames.
  - SQL type restrictions: VARIANT, MAP, and datetime types are unsupported.
  - External $ref URLs not supported; only internal #/$defs/ references (must use $defs key at schema root).
  - For OpenAI-family models via REST: additionalProperties must be false and required must list all properties.
  - Claude models via REST API support only json_schema type; json_object type is not supported for Claude.
  - Max output tokens: 8,192.
  - No additional cost for schema verification overhead, but complex schemas increase token consumption.
- **Best practices:**
  - Use AI_COMPLETE (not SNOWFLAKE.CORTEX.COMPLETE) for all new SQL code.
  - Prefer JSON schema form over TYPE OBJECT type-literal form — it works across SQL, Python SDK, and REST API.
  - Use Pydantic: define your model, call .model_json_schema(), pass result as schema value; this handles $defs references automatically.
  - Use the OpenAI Python SDK with base_url='https://<account>.snowflakecomputing.com/api/v2/cortex/v1' and a Snowflake PAT as api_key for REST access.
  - Always include required array listing all mandatory fields.
  - For OpenAI-family models set additionalProperties: false in the schema.
  - Choose a capable model (claude-sonnet-4-6, llama3.3-70b, mistral-large2) for complex schemas.
  - Do not rely on unsupported constraint keywords (minLength, maxItems, etc.) — they are silently ignored or cause errors.
- **Sources:** <https://docs.snowflake.com/en/user-guide/snowflake-cortex/complete-structured-outputs>, <https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api>, <https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex>, <https://docs.snowflake.com/en/release-notes/2025/other/2025-02-11-cortex-complete-structured-outputs>

### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/snowflake_cortex_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** CortexSchematicGenerator._do_generate builds a plain chat payload with a single user-role message, then conditionally adds a response_format block only when hints["strict"] is truthy. It POSTs to POST {SNOWFLAKE_CORTEX_BASE_URL}/api/v2/cortex/inference:complete via raw httpx.AsyncClient (no Snowflake Python SDK). Response text is extracted from choices[0].message.content (with a content_list fallback), stripped of ```json fences via normalize_json_output, then json.loads'd, then pydantic model_validate'd. ValidationError is not retried — it raises immediately.
- **Uses native structured output:** yes (response_format {"type": "json", "schema": &lt;pydantic model_json_schema()&gt;} — only when hints["strict"] is truthy; otherwise prompt-only)
- **Schema sent to API:** yes
- **Gating:** hints["strict"] — default is False (native structured output is OFF by default; callers must explicitly pass strict=True)
- **Parsing fallbacks:** normalize_json_output strips ```json...``` fences (common.py:19-32); double-try block: first attempt on raw string, second attempt on str(raw) if first fails (snowflake_cortex_service.py:189-203); content_list[0]['text'] fallback if choices[0].message.content is None (snowflake_cortex_service.py:182-184); raw dict passed directly to json (skips normalize) if isinstance(raw, dict) (snowflake_cortex_service.py:193-194)
- **Retries on ValidationError:** no
- **max_tokens handling:** Driven by env var SNOWFLAKE_CORTEX_MAX_TOKENS (default 8192, read at __init__ time). Exposed via max_tokens property. NOT injected into API payload automatically — only added to payload if caller passes max_tokens in hints (line 152-154). So the env-var default is used only for token-budget estimation, not enforced server-side unless explicitly hinted.
- **Notable issues:**
  - Single user-role message only (line 145) — no system role support; entire prompt including instructions lands in user turn
  - strict=False default means schema is never sent to the API unless callers opt in — easy to miss, silently falls back to prompt-only
  - Schema export failure in strict path is silently swallowed (lines 163-165): response_format key is never set, so the call proceeds without any structured output enforcement with no warning to the caller
  - max_tokens env-var default (8192) does NOT auto-populate the API payload; it only sizes the tokenizer budget (lines 88, 102-103, 152-154)
  - record_llm_metrics passes cached_input_tokens=0 unconditionally — Cortex usage block is not inspected for cache fields (line 216-222)
  - UsageInfo.extra={} always empty — no cache or reasoning token pass-through (line 232)
  - No streaming support despite Cortex supporting SSE; supports_streaming returns False (line 385-386)
  - CortexEmbedder.max_tokens is hardcoded to 8192 (line 338) rather than read from env/config
  - _COUNTERS_INITIALIZED global in common.py is module-level state shared across all adapter instances — not thread/process safe for multi-service setups
- **Key code refs:**
  - `snowflake_cortex_service.py:145 — single user-role message construction`
  - `snowflake_cortex_service.py:152-154 — provider params from hints injected into payload`
  - `snowflake_cortex_service.py:157-165 — strict gate: response_format with schema only when hints['strict'] is truthy`
  - `snowflake_cortex_service.py:167-175 — httpx POST to /api/v2/cortex/inference:complete + raise_for_status`
  - `snowflake_cortex_service.py:179-186 — response parsing: choices[0].message.content with content_list fallback`
  - `snowflake_cortex_service.py:189-203 — double-try JSON parse with normalize_json_output fallback`
  - `snowflake_cortex_service.py:206-212 — pydantic model_validate; ValidationError raises immediately (no retry)`
  - `snowflake_cortex_service.py:214-222 — usage extraction from response; cached_input_tokens hardcoded 0`
  - `snowflake_cortex_service.py:88 — SNOWFLAKE_CORTEX_MAX_TOKENS env var read`
  - `common.py:19-32 — normalize_json_output strips ```json fences`
  - `common.py:41-83 — record_llm_metrics with module-level counter globals`

### Gaps

#### [HIGH] Structured output is OFF by default — callers must opt in via hints["strict"]=True

`src/parlant/adapters/nlp/snowflake_cortex_service.py:157`

Line 157 gates the entire response_format block behind hints.get("strict", False). Any caller that omits the hint receives no schema enforcement — the API returns free-form text and the adapter hopes prompt-based instructions produce valid JSON. The provider supports constrained decoding on every call at no extra cost.

**Recommendation:** Flip the default: always send response_format unless the caller explicitly opts out. Change line 157 to `if not hints.get('disable_strict', False):` and remove `hints.get('strict', False)`. The response_format block should be the unconditional path, matching best practice of always including the required array and schema.

#### [HIGH] Silent schema-export failure degrades to prompt-only JSON with no caller signal

`src/parlant/adapters/nlp/snowflake_cortex_service.py:163-165`

Lines 163-165: when schema.model_json_schema() raises, the exception is caught, logged at DEBUG, and execution continues without setting response_format. The API call proceeds as if structured output were never requested. The caller receives a SchematicGenerationResult that may or may not conform to the schema, with no indication that constrained decoding was bypassed.

**Recommendation:** Remove the bare except. Let the exception propagate so the caller knows structured output could not be configured. If a fallback is genuinely needed, at minimum log at ERROR and raise a descriptive exception: `raise RuntimeError(f'Cannot export JSON schema for {schema.__name__}: {e}') from e`. Never silently continue without response_format when structured output was the intent.

#### [HIGH] Wrong response_format type: uses "json" instead of "json_schema" for the REST/OpenAI-compatible endpoint

`src/parlant/adapters/nlp/snowflake_cortex_service.py:159-162`

Line 159 sets `"type": "json"` in response_format. The OpenAI-compatible REST endpoint (POST .../api/v2/cortex/v1/chat/completions) requires `{"type": "json_schema", "json_schema": {"name": "<name>", "schema": {...}}}`. The `"type": "json"` form is the SQL/Python-SDK (AI_COMPLETE / snowflake-ml-python) form, not the REST form. Using the wrong shape against the REST endpoint may cause the schema to be ignored or the request to be rejected.

**Recommendation:** Switch to the OpenAI-compatible REST format: `payload["response_format"] = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}`. Also add `"additionalProperties": False` and ensure `"required"` lists all properties (required for OpenAI-family models per the docs). The endpoint should also be changed from `/api/v2/cortex/inference:complete` to `/api/v2/cortex/v1/chat/completions` to use the OpenAI-compatible surface that fully supports json_schema.

#### [MEDIUM] No retry on ValidationError — a single malformed response is fatal

`src/parlant/adapters/nlp/snowflake_cortex_service.py:206-212`

Lines 206-212: when model_validate raises ValidationError, it is logged and immediately re-raised. There is no retry. When strict mode IS active, constrained decoding makes this unlikely — but when response_format is missing (see gap 1) or when the model produces a subtly wrong value, a retry with the same or corrected prompt would succeed. The existing @policy decorator only retries HTTP-level errors, not validation failures.

**Recommendation:** Wrap the model_validate call in a retry loop (max 2 retries). On ValidationError, append the validation error message to the prompt or re-issue the request. Alternatively, use the `retry` policy decorator from `parlant.core.nlp.policies` to add `retry(ValidationError, max_exceptions=2, wait_times=(0.0, 0.0))` to the @policy list, then let the outer do_generate re-enter _do_generate.

#### [MEDIUM] max_tokens env-var default never reaches the API payload — truncated JSON output possible

`src/parlant/adapters/nlp/snowflake_cortex_service.py:88-103`

Line 88 reads SNOWFLAKE_CORTEX_MAX_TOKENS into self._max_tokens_hint (default 8192). Lines 152-154 only add max_tokens to payload if the caller passes it in hints. So the env-var default is used only for token-budget estimation (max_tokens property), never enforced server-side. The API has a hard limit of 8,192 output tokens; if the server's default is lower, complex schema outputs can be truncated mid-JSON, causing parse failures.

**Recommendation:** Always inject max_tokens into the payload from self._max_tokens_hint as a floor: after the hints loop (line 154), add `if 'max_tokens' not in payload: payload['max_tokens'] = self._max_tokens_hint`. This ensures the full 8,192-token budget is requested by default without requiring every caller to pass the hint.

#### [MEDIUM] Single user-role message — no system prompt support

`src/parlant/adapters/nlp/snowflake_cortex_service.py:145`

Line 145 constructs messages as a single user-role turn. All instructions (persona, constraints, schema hints) are concatenated into the user message. Cortex REST API supports the standard messages array with role=system. Without a system role, the model has less structural separation between instructions and user input, which can reduce output quality — especially relevant for complex schemas where precise instruction following matters.

**Recommendation:** If PromptBuilder produces a system section, split it into a system-role message: `messages = [{'role': 'system', 'content': system_part}, {'role': 'user', 'content': user_part}]`. Inspect PromptBuilder's API for a `system_prompt` property or similar to extract the system section before building.

#### [MEDIUM] Prompt-scraping fallbacks (normalize_json_output, double-try parse) do work the API guarantees

`src/parlant/adapters/nlp/snowflake_cortex_service.py:189-203`

Lines 189-203 implement a two-stage JSON extraction pipeline: strip ```json fences, try json.loads, fall back to str(raw) parse. This machinery exists because without response_format the model may emit markdown-wrapped JSON or non-JSON text. Once response_format with a schema is always sent (gap 1 fixed), the API guarantees a bare JSON object — these fallbacks become dead code that adds latency and masks bugs.

**Recommendation:** After fixing gap 1 (always send response_format), simplify the parse path to a direct `parsed = resp.json()['choices'][0]['message']['content']` followed by a single `json.loads` — no fence stripping needed. Keep normalize_json_output only as a last-resort fallback behind a logged warning so regressions are visible.

#### [LOW] cached_input_tokens hardcoded to 0 — cache metrics always zero

`src/parlant/adapters/nlp/snowflake_cortex_service.py:216-222`

Line 216-222: record_llm_metrics is called without cached_input_tokens, so the default of 0 is always used. The Cortex usage block may include cache fields (e.g. prompt_tokens_details.cached_tokens or similar). UsageInfo.extra={} at line 233 also discards any cache or reasoning token data from the response.

**Recommendation:** Inspect usage_block for cache token fields. At minimum: `cached = usage_block.get('prompt_tokens_details', {}).get('cached_tokens', 0)` and pass it to `record_llm_metrics(..., cached_input_tokens=cached)`. Also populate `UsageInfo.extra` with any additional fields from the usage block for observability.

#### [LOW] Module-level counter globals in common.py are not safe for multi-service setups

`src/parlant/adapters/nlp/common.py:35-68`

common.py lines 35-38 declare _INPUT_TOKENS_COUNTER, _OUTPUT_TOKENS_COUNTER, _CACHED_TOKENS_COUNTER, and _COUNTERS_INITIALIZED as module-level globals. When multiple NLPService instances (e.g. Cortex + another adapter) coexist in the same process, the first service to call record_llm_metrics initializes the counters with its meter. Subsequent services use the same counter objects regardless of their own meter instance, silently misrouting metrics.

**Recommendation:** Move counter initialization into the Meter instance or pass meter-keyed counters. Simplest fix: use a `dict[int, tuple[Counter, Counter, Counter]]` keyed by `id(meter)` instead of flat globals, initializing per-meter on first call. Alternatively, make record_llm_metrics a class method on a per-adapter metrics helper instantiated in __init__.


## Emcie (internal) — verdict: **suboptimal**

The Emcie adapter sends only the schema class name (a string) to the API — the actual JSON schema definition is never transmitted. This means constrained decoding and server-side schema enforcement are impossible unless the backend has an out-of-band registry of every Pydantic model name. On top of that, ValidationError is never retried, cached_input_tokens is always 0, and the cost field from responses is silently discarded. The remaining issues (jsonfinder fallback, tiktoken model name, no-role flattening) are robustness concerns that the backend should be able to eliminate once the schema is transmitted.


### Current implementation

- **Files:** `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/emcie_service.py`, `/Users/ibanez/projects/parlant/parlant/src/parlant/adapters/nlp/common.py`
- **Approach:** EmcieSchematicGenerator._do_generate builds a raw httpx POST to {BASE_URL}/v1/completions with a JSON body containing model_tier, model_role, the full prompt as a plain string, schema_name (string, not the schema itself), and filtered hints. The response body's "completion" field is expected to already be JSON text; it is parsed with json.loads(normalize_json_output(raw_content)), then validated with self.schema.model_validate(json_content). No OpenAI/Anthropic SDK is used — raw httpx only.
- **Uses native structured output:** no (none (prompt-only) — only the schema class name string is sent as "schema_name"; the actual JSON schema is never transmitted. Structured output enforcement is delegated entirely to the Emcie backend, which may or may not enforce it server-side.)
- **Schema sent to API:** no
- **Gating:** No strict hint or any schema-gating exists. The only hints forwarded are those in supported_emcie_params = ["temperature"], so hints["strict"] would be silently dropped. No default applies.
- **Parsing fallbacks:** normalize_json_output strips ```json...``` fences (common.py:19-32); jsonfinder.only_json(raw_content)[2] scavenges JSON from surrounding text when json.loads fails (emcie_service.py:235)
- **Retries on ValidationError:** no
- **max_tokens handling:** max_tokens is a read-only property on each model class (Jackal/Bison both return 128*1024, emcie_service.py:290/312). It is never sent to the API — not included in the POST body. Hints cannot override it; it exists only for the tokenizer budget layer above.
- **Notable issues:**
  - schema_sent_to_api is false — only self.schema.__name__ (a string) goes in the request body at line 184; the actual Pydantic JSON schema is never serialized or transmitted
  - No 'strict' hint support; hints dict is filtered to only 'temperature' (line 187), so any structured-output hint a caller passes is silently discarded
  - max_tokens property (128k) is defined but never sent to the API endpoint (not present in the POST json body at lines 180-189)
  - cached_input_tokens is hardcoded to 0 in record_llm_metrics call (line 248) — cache hits are never accounted for
  - No message roles / chat turns — the entire conversation is flattened into a single 'prompt' string (line 183); no system/user/assistant role separation
  - ValidationError on pydantic model_validate (line 239) logs and re-raises immediately with no retry — the @policy retry decorators only cover RateLimitError and EmcieAPIError (lines 137-141)
  - EmcieEstimatingTokenizer uses tiktoken with 'gpt-4.1' encoding (line 87) — this is a non-standard model name for tiktoken and may fall back silently or raise
  - InsufficientCreditsError is not included in the @policy retry decorators, so it re-raises immediately (correct behavior, but asymmetric with RateLimitError logging)
  - cost field is read from response (line 226) but never stored or surfaced in GenerationInfo/UsageInfo — it is only logged at trace level
- **Key code refs:**
  - `emcie_service.py:174 — httpx POST to {BASE_URL}/v1/completions`
  - `emcie_service.py:180-189 — request body construction (model_tier, model_role, prompt, schema_name, hints, payload)`
  - `emcie_service.py:184 — schema_name sent as string only, not schema definition`
  - `emcie_service.py:187 — hints filtered to supported_emcie_params=["temperature"]`
  - `emcie_service.py:229 — raw_content = response_data["completion"]`
  - `emcie_service.py:232 — json.loads(normalize_json_output(raw_content))`
  - `emcie_service.py:235 — jsonfinder.only_json fallback on JSONDecodeError`
  - `emcie_service.py:239 — self.schema.model_validate(json_content)`
  - `emcie_service.py:264-268 — ValidationError caught, logged, re-raised (no retry)`
  - `emcie_service.py:137-141 — @policy retry: RateLimitError (unlimited), EmcieAPIError (max 2, waits 1s/5s)`
  - `emcie_service.py:248 — cached_input_tokens hardcoded to 0`
  - `emcie_service.py:290 — Jackal.max_tokens returns 128*1024 (never sent to API)`
  - `emcie_service.py:312 — Bison.max_tokens returns 128*1024 (never sent to API)`
  - `emcie_service.py:87 — tiktoken.encoding_for_model("gpt-4.1") non-standard model name`
  - `common.py:19-32 — normalize_json_output strips ```json fences`

### Gaps

#### [HIGH] Actual JSON schema never transmitted — only a name string is sent

`src/parlant/adapters/nlp/emcie_service.py:184`

Line 184 sends `"schema_name": self.schema.__name__` — a plain Python class name string. The actual Pydantic JSON schema (`self.schema.model_json_schema()`) is never serialized or included in the POST body. If the Emcie backend performs constrained decoding or any server-side validation it cannot do so without the schema definition. Any schema that is not pre-registered server-side will receive unconstrained generation, making structured output unreliable.

**Recommendation:** Add `"schema": self.schema.model_json_schema()` alongside `"schema_name"` in the POST body (emcie_service.py:180-189). Example: `json={..., "schema_name": self.schema.__name__, "schema": self.schema.model_json_schema(), ...}`. This is the standard practice used by OpenAI (`response_format=self.schema` in `beta.chat.completions.parse`) and Gemini (`response_schema=...`). The backend can then apply constrained decoding against the transmitted schema.

#### [HIGH] No retry on ValidationError — a single bad generation is a hard failure

`src/parlant/adapters/nlp/emcie_service.py:137-141`

The `@policy` decorators at lines 137-141 cover `RateLimitError` and `EmcieAPIError` only. `ValidationError` raised at line 264 propagates immediately with no retry. Every other adapter in the codebase (OpenAI, Anthropic) also raises on ValidationError without retry, but the Emcie adapter is the one that cannot enforce schema server-side, making client-side validation failures far more likely. Without retry, a single malformed response aborts the entire generation.

**Recommendation:** Either (a) wrap `_do_generate` in a retry policy that catches `ValidationError` (e.g., add `retry(ValidationError, max_exceptions=2, wait_times=(0.0, 0.0))` to the `@policy` list at line 137) so the call is re-issued when the model returns invalid JSON, or (b) transmit the schema (see gap above) so the backend guarantees conformance and ValidationError becomes a true unexpected error. Option (b) is preferred; option (a) is the minimum safety net.

#### [MEDIUM] cached_input_tokens is hardcoded to 0 — cache hits never accounted for

`src/parlant/adapters/nlp/emcie_service.py:247`

Line 247 passes `cached_input_tokens=0` to `record_llm_metrics` unconditionally. The Emcie API response already contains a `usage` dict (line 224). If the backend returns a `cached_input_tokens` (or similarly named) field, it is silently ignored. This means the `cached_input_tokens` counter metric (defined in common.py:63-66) is always zero for the Emcie provider, making cost attribution and cache-hit monitoring useless for this adapter.

**Recommendation:** Read the cached token count from the response: `cached_input_tokens=int(usage.get("cached_input_tokens", 0))` at line 247, and propagate it into `UsageInfo.extra` at line 259 (`extra={"cached_input_tokens": int(usage.get("cached_input_tokens", 0))}`). Mirror the pattern from openai_service.py:227 and openai_service.py:240.

#### [MEDIUM] cost field read from response but never surfaced — billing data discarded

`src/parlant/adapters/nlp/emcie_service.py:225-227`

Line 225-227 reads `cost = response_data["cost"]` and logs it at trace level only. It is not stored in `GenerationInfo`, `UsageInfo.extra`, or any metric. Operators have no programmatic way to track per-call billing from the Emcie adapter; they must parse trace logs manually.

**Recommendation:** Store cost fields in `UsageInfo.extra` at line 259: `extra={"cost": cost, "cached_input_tokens": ...}`. This makes the data available to any caller inspecting `SchematicGenerationResult.info.usage.extra`, consistent with how the OpenAI adapter surfaces `cached_input_tokens` in the same dict.

#### [MEDIUM] jsonfinder fallback does work the API should guarantee

`src/parlant/adapters/nlp/emcie_service.py:232-236`

Lines 233-236 use `jsonfinder.only_json(raw_content)[2]` to scavenge JSON from surrounding prose when `json.loads` fails. This fallback is only necessary because the backend has no schema to enforce against. If the schema is transmitted and constrained decoding is applied server-side (see gap 1), the `completion` field will always be valid JSON and these fallbacks become dead code. Until then, the fallback is the only guard against malformed responses.

**Recommendation:** Transmit the schema (gap 1) to enable server-side enforcement. Once that is confirmed working, the `jsonfinder` fallback at line 235 and `normalize_json_output` fence-stripping at line 232 can be removed, simplifying the parsing path. In the interim, keep the fallbacks but add a metric increment when they fire so you can track how often constrained decoding is failing.

#### [MEDIUM] hints dict filtered to temperature only — any schema-enforcement hint is silently dropped

`src/parlant/adapters/nlp/emcie_service.py:112`

`supported_emcie_params = ["temperature"]` at line 112 means any hint a caller passes — including a hypothetical `"strict": True` or `"response_format": ...` — is silently discarded at line 185-187. There is no warning, no logging, and no way for a caller to know their hint was ignored. This is particularly problematic for structured-output hints.

**Recommendation:** Log a warning for any hint key present in the input `hints` dict that is NOT in `supported_emcie_params`. Example: add `for k in hints: if k not in self.supported_emcie_params: self.logger.warning(f"Hint '{k}' is not supported by the Emcie adapter and will be ignored")` before the filtered dict construction at line 185. Additionally, once the schema is transmitted, consider adding a `"strict"` param to `supported_emcie_params` and forwarding it to the API to opt into constrained decoding.

#### [LOW] Single flat prompt string — no system/user/assistant role separation

`src/parlant/adapters/nlp/emcie_service.py:183`

Line 183 sends the entire conversation as `"prompt": prompt` — a single string with no message roles. Modern LLM APIs (including all other adapters in this repo) use a `messages` array with `role` fields (`system`, `user`, `assistant`). Without role separation the backend cannot apply role-specific constraints (e.g., a system prompt that reinforces the output schema), and instruction-following quality may degrade for long multi-turn contexts.

**Recommendation:** If the Emcie API supports a `messages` field (similar to `POST /v1/chat/completions`), migrate from `"prompt": prompt` to `"messages": [{"role": "user", "content": prompt}]` or split the `PromptBuilder` output into system/user parts. Check the `PromptBuilder.props` dict (available at line 158) for any existing role separation that could be forwarded directly.

#### [LOW] tiktoken gpt-4.1 encoding works but is implicit — should use a stable encoding name

`src/parlant/adapters/nlp/emcie_service.py:87`

Line 87 calls `tiktoken.encoding_for_model("gpt-4.1")`. This currently resolves to `o200k_base` (confirmed in the environment), which is correct for the Emcie models. However, the mapping is internal to tiktoken's model registry and could change or fail if tiktoken drops or renames the `gpt-4.1` entry in a future release. The Emcie models are not OpenAI models so the indirect lookup is fragile.

**Recommendation:** Replace `tiktoken.encoding_for_model("gpt-4.1")` with `tiktoken.get_encoding("o200k_base")` to pin the encoding directly. This is explicit, stable, and does not depend on tiktoken's model-name registry. Note: the OpenAI adapter (openai_service.py:109) uses `encoding_for_model` with the actual OpenAI model name — that is correct there but inappropriate for a non-OpenAI service.


---

# PR plan

Pinned SDK versions on `develop` (from `uv.lock`) — all recent enough for native structured
output: `openai 2.21.0` (non-beta `chat.completions.parse`), `anthropic 0.83.0`
(GA `output_config.format` / `messages.parse`), `google-genai 1.64.0` (pre-2.0 surface:
`response_mime_type` + `response_schema` + `response.parsed`), `mistralai 1.12.4`
(`chat.parse`), `ollama 0.6.1`, `litellm 1.81.14`, `zhipuai 2.1.5`, `cerebras-cloud-sdk 1.67.0`,
`together 2.2.0`. The `fireworks` extra is disabled in `pyproject.toml` (protobuf CVE), so
Fireworks tests must mock the SDK at the module level.

Merge order matters: later PRs in the list rebase over earlier ones; the structured-output PRs
(5-7) intentionally subsume per-adapter `max_tokens`/metrics fixes for the adapters they rewrite.

| # | Branch | Scope | Risk |
|---|---|---|---|
| 1 | `fix/structured-output-runtime-bugs` | Fireworks missing `await` + `schema_name` TypeError; Cerebras `schema_name` TypeError; dotted-`getattr` cache-token bug (deepseek, qwen, glm, litellm); ModelScope `generate`→`do_generate`; AWS `verify_environment` env check; Zhipu async client | None — fixes broken code |
| 2 | `fix/llm-metrics-observability` | `common.py` per-meter counters; missing `record_llm_metrics` call sites (OpenRouter generate path, Azure non-strict path, Vertex-Gemini path); cache-token accounting (Mistral, Snowflake hardcoded 0) | Low — metrics only |
| 3 | `fix/max-tokens-handling` | Remove hardcoded `max_tokens` in litellm (5000), qwen (8192), glm (4096), aws (4096); derive from hints with model-property fallback; fix hint/kwarg collision | Low-medium |
| 4 | `feat/schematic-retry-on-validation-error` | Retry-with-feedback (2 retries, error fed back into prompt) on `ValidationError`/`JSONDecodeError` in `BaseSchematicGenerator.generate()` — fixes the universal no-retry gap for all 20 adapters in one place | Medium |
| 5 | `feat/openai-azure-native-structured-output` | Default to non-beta `chat.completions.parse(response_format=PydanticModel)` for OpenAI + Azure; strict-incompatible schemas fall back to `json_object`; handle `.refusal`; remove `assert` crashes on usage fields | High value / medium risk |
| 6 | `feat/anthropic-native-structured-output` | Use GA `output_config.format` json_schema (SDK 0.83.0) for Anthropic; fix hardcoded `max_tokens=4096`; cache-token accounting | High value / medium risk |
| 7 | `feat/gemini-vertex-native-structured-output` | Gemini: replace tool-forcing workaround with `response_mime_type='application/json'` + `response_schema=<PydanticModel>` + `response.parsed`; Vertex-Gemini: pass Pydantic class instead of raw dict, add metrics; Vertex-Claude: same as PR 6 via AnthropicVertex | High value / medium risk |

OpenRouter `json_schema` mode (with `provider.require_parameters` routing and a
per-instance `json_object` fallback) ships in PR 2 alongside the metrics fixes.

Deferred (documented here for a follow-up agent): Mistral `chat.parse`, Bedrock Converse
`outputConfig`, Together/GLM `json_schema` mode, prompt-side `model_json_schema()`
embedding for json_object-only providers (DeepSeek/Qwen), schema simplification
(`str | list[str]` unions, `dict[str, str | None]`) to satisfy strict modes, and
`DefaultBaseModel` `extra` policy review.

## Completeness critique (what this audit may have missed)

- hugging_face.py — embedder only (HuggingFaceEmbedder, JinaAIEmbedder), no SchematicGenerator; correctly excluded from structured-output audit.
- lakera.py — moderation service only (LakeraGuard), no generation path at all; correctly excluded.
- common.py itself: the module-level _COUNTERS_INITIALIZED / _INPUT_TOKENS_COUNTER / _OUTPUT_TOKENS_COUNTER / _CACHED_TOKENS_COUNTER globals are a single shared process-level state. If two different Meter instances (e.g., separate services in the same process) call record_llm_metrics, the first one wins and all subsequent calls silently reuse its counters — meter ownership is broken. This was flagged in passing but not given a dedicated per-provider analysis section.
- openrouter_service.py: the SchematicGeneratorOpenRouter class imports normalize_json_output from common but never imports or calls record_llm_metrics. Token usage is therefore never metered on the openrouter path — confirmed by grep (no call site). The per-provider audit mentions this as medium severity but did not identify that the import is absent, not just uncalled.
- ValidationError re-raise path in BaseSchematicGenerator.generate() (core/nlp/generation.py): the core audit notes no retry exists at engine level, but does not explicitly audit whether FallbackSchematicGenerator wraps ValidationError specially. Confirmed: it catches bare Exception, so ValidationError is silently swallowed in a fallback chain and an empty/wrong result may be returned instead of raising.
- No audit of whether DefaultBaseModel sets extra='forbid' — it does not (confirmed: only validate_default=True and model_title_generator). This means extra LLM output fields are silently dropped by Pydantic across all 20+ engine schemas, masking model hallucinations rather than surfacing them.

## Top priorities (ranked)

- 1. Add ValidationError/JSONDecodeError retry at BaseSchematicGenerator.generate() in core/nlp/generation.py. A single catch-and-retry loop (2-3 attempts with the error included in a corrective prompt) would instantly improve structured-output reliability across all 20 providers without changing any adapter code. This is the highest-leverage single change in the entire codebase because it fixes the universal no-retry gap in one place.
- 2. Make the OpenAI and Azure adapters default to native structured output (json_schema / beta.chat.completions.parse with strict=True) unconditionally, or have the engine pass hints={'strict':True}. These are the two dominant providers; their default json_object path provides zero schema enforcement and relies on fragile post-processing. Switching to native constrained decoding eliminates the normalize_json_output + jsonfinder pipeline for all OpenAI/Azure traffic and makes schema compliance a server-side guarantee.
- 3. Fix the dotted-string getattr bug in deepseek_service.py, qwen_service.py, litellm_service.py, glm_service.py, and openrouter_service.py: change getattr(response, 'usage.prompt_cache_hit_tokens', 0) to getattr(response.usage, 'prompt_cache_hit_tokens', 0). This is a one-line fix per file that restores accurate cache-hit token accounting for five adapters — a correctness bug with zero risk.
- 4. Fix common.py record_llm_metrics counter singleton: replace module-global _COUNTERS_INITIALIZED / _INPUT_TOKENS_COUNTER state with per-meter-instance counter creation (e.g., a dict keyed by id(meter), or convert the function to a class). In a multi-provider deployment every adapter after the first silently writes metrics to the wrong Meter, making all token observability data unreliable.
- 5. Embed model_json_schema() in engine prompts (or pass it as a structured tool spec). Currently all engine batches describe the expected output only in prose. Adding the Pydantic JSON Schema to the prompt — even for json_object-mode adapters — gives the model a machine-readable contract, reduces hallucinated field names and types, and is a prerequisite for activating json_schema mode safely on any provider. Start with the three highest-traffic schemas: GenericActionableGuidelineMatchesSchema (shallow, clean), MessageSchema (complex, high value), and SingleToolBatchSchema (deeply nested, most failure-prone).
