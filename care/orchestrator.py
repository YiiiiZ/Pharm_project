"""Pure-Python orchestration for routing and hallucination repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .generation_loop import (
    Evaluator,
    RetryLoopResult,
    generate_with_retry_loop,
)
from .router import RoutingStrategy, assess_complexity


class GeneratorFactory(Protocol):
    def __call__(
        self,
        *,
        prompt_version: str,
        model: str,
    ) -> Any:
        """Return a generator compatible with generate_with_retry_loop()."""


@dataclass(frozen=True)
class OrchestrationResult:
    strategy: RoutingStrategy
    loop_result: RetryLoopResult
    decision_log: tuple[str, ...] = field(default_factory=tuple)

    @property
    def prompt_version(self) -> str:
        return self.strategy.prompt_version

    @property
    def model(self) -> str:
        return self.strategy.model

    @property
    def status(self) -> Any:
        return self.loop_result.status

    @property
    def output(self) -> str | None:
        return self.loop_result.final_output

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.as_dict(),
            "loop_result": {
                "status": self.loop_result.status.value,
                "attempts_used": self.loop_result.attempts_used,
                "retries_used": self.loop_result.retries_used,
                "original_output": self.loop_result.original_output,
                "final_output": self.loop_result.final_output,
                "total_cost": self.loop_result.total_cost,
                "decision_log": list(self.loop_result.decision_log),
            },
            "decision_log": list(self.decision_log),
        }


class CarePlanOrchestrator:
    """Coordinate complexity routing and hallucination repair."""

    def __init__(
        self,
        *,
        router: Callable[..., RoutingStrategy] = assess_complexity,
        retry_loop: Callable[..., RetryLoopResult] = generate_with_retry_loop,
    ) -> None:
        self.router = router
        self.retry_loop = retry_loop

    def process(
        self,
        *,
        patient_record: Mapping[str, Any],
        generator_factory: GeneratorFactory,
        evaluator: Evaluator,
        initial_prompt: str,
        max_retries: int = 2,
        router_kwargs: Mapping[str, Any] | None = None,
        retry_loop_kwargs: Mapping[str, Any] | None = None,
    ) -> OrchestrationResult:
        router_kwargs = dict(router_kwargs or {})
        retry_loop_kwargs = dict(retry_loop_kwargs or {})

        strategy = self.router(patient_record, **router_kwargs)
        decision_log = list(strategy.decision_log)
        decision_log.append(
            "router selected "
            f"complexity={strategy.complexity} "
            f"prompt_version={strategy.prompt_version} "
            f"model={strategy.model}"
        )

        generator = generator_factory(
            prompt_version=strategy.prompt_version,
            model=strategy.model,
        )
        decision_log.append(
            "generator factory bound "
            f"prompt_version={strategy.prompt_version} "
            f"model={strategy.model}"
        )

        loop_result = self.retry_loop(
            generator,
            evaluator,
            initial_prompt=initial_prompt,
            max_retries=max_retries,
            **retry_loop_kwargs,
        )
        decision_log.append(
            "loop completed "
            f"status={loop_result.status.value} "
            f"attempts={loop_result.attempts_used} "
            f"retries={loop_result.retries_used} "
            f"total_cost={loop_result.total_cost:.4f}"
        )
        decision_log.extend(loop_result.decision_log)

        return OrchestrationResult(
            strategy=strategy,
            loop_result=loop_result,
            decision_log=tuple(decision_log),
        )
