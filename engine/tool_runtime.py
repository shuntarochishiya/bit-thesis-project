"""Runtime validation for tool calls dispatched by the DAG engine.

A compilation contract says *what* can run; this module checks the data at
call time. It cannot make a non-idempotent side effect safe to retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import time


class ToolRuntimeError(RuntimeError):
    """Bad tool input, output, or a failed tool invocation."""


@dataclass(frozen=True)
class RuntimeToolSpec:
    name: str
    read_only: bool
    required_input: tuple[str, ...] = ()
    required_output: tuple[str, ...] = ()
    # Validation is deliberately narrow: no coercion of values returned by agents.
    output_types: tuple[tuple[str, type], ...] = ()
    transient_retries: int = 0


PRECONDITION_SPEC = RuntimeToolSpec(
    name="precondition",
    read_only=True,
    required_input=("intent", "target", "game_state", "player_input"),
    required_output=("success", "message", "state_updates"),
    output_types=(("success", bool), ("message", str), ("state_updates", dict)),
    transient_retries=0,
)


class ToolRuntime:
    """Small reusable tool boundary, without altering agent return contracts."""

    @staticmethod
    def validate_inputs(spec: RuntimeToolSpec, inputs: Mapping[str, Any]) -> None:
        if not isinstance(inputs, Mapping):
            raise ToolRuntimeError(f"{spec.name}: input must be a mapping")
        for field in spec.required_input:
            if field not in inputs:
                raise ToolRuntimeError(f"{spec.name}: missing input {field!r}")
        if "game_state" in inputs and not isinstance(inputs["game_state"], dict):
            raise ToolRuntimeError(f"{spec.name}: game_state must be a dict")

    @staticmethod
    def validate_output(spec: RuntimeToolSpec, output: Any) -> None:
        if not isinstance(output, dict):
            raise ToolRuntimeError(f"{spec.name}: output must be a dict")
        for field in spec.required_output:
            if field not in output:
                raise ToolRuntimeError(f"{spec.name}: missing output {field!r}")
        for field, type_ in spec.output_types:
            if field in output and not isinstance(output[field], type_):
                raise ToolRuntimeError(f"{spec.name}: output {field!r} must be {type_.__name__}")

    def call(
        self,
        spec: RuntimeToolSpec,
        function: Callable[..., Any],
        inputs: Mapping[str, Any],
    ) -> Any:
        self.validate_inputs(spec, inputs)
        # Never replay side-effecting calls. Retries only for declared read-only
        # operations and transient transport failures, not invalid results.
        retries = spec.transient_retries if spec.read_only else 0
        for attempt in range(retries + 1):
            try:
                output = function(**inputs)
                self.validate_output(spec, output)
                return output
            except (TimeoutError, ConnectionError, OSError):
                if attempt == retries:
                    raise
                time.sleep(0.1)
        raise AssertionError("unreachable")

    @staticmethod
    def validate_state_updates(updates: Any) -> None:
        if not isinstance(updates, dict):
            raise ToolRuntimeError("state_updates must be a dict")
        if any(not isinstance(key, str) or not key for key in updates):
            raise ToolRuntimeError("state_updates keys must be non-empty strings")

    @staticmethod
    def validate_fallback(result: Any) -> None:
        spec = RuntimeToolSpec(
            name="fallback", read_only=False,
            required_output=("success", "message", "state_updates"),
            output_types=(("success", bool), ("message", str), ("state_updates", dict)),
        )
        ToolRuntime.validate_output(spec, result)
