from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping

import yaml

logger = logging.getLogger(__name__)


class PromptError(Exception):
    """Base exception for prompt configuration and rendering failures."""


class PromptConfigurationError(PromptError):
    """Raised when config.yaml is missing or invalid."""


class PromptNotFoundError(PromptError):
    """Raised when a configured prompt file does not exist."""


class PromptRenderError(PromptError):
    """Raised when required template variables are missing."""


@dataclass(frozen=True)
class PromptTemplate:
    workflow: str
    requested_version: str
    resolved_version: str
    content: str
    path: Path
    checksum: str

    def render(self, variables: Mapping[str, Any]) -> "RenderedPrompt":
        string_variables = {
            key: "" if value is None else str(value)
            for key, value in variables.items()
        }
        try:
            content = Template(self.content).substitute(string_variables)
        except KeyError as exc:
            missing = exc.args[0]
            raise PromptRenderError(
                f"Missing variable '{missing}' for "
                f"{self.workflow}/{self.resolved_version}"
            ) from exc

        logger.info(
            "Rendered prompt workflow=%s version=%s checksum=%s",
            self.workflow,
            self.resolved_version,
            self.checksum,
        )
        return RenderedPrompt(
            workflow=self.workflow,
            requested_version=self.requested_version,
            version=self.resolved_version,
            content=content,
            path=self.path,
            checksum=self.checksum,
        )


@dataclass(frozen=True)
class RenderedPrompt:
    workflow: str
    requested_version: str
    version: str
    content: str
    path: Path
    checksum: str

    def metadata(self) -> dict[str, str]:
        """Return fields suitable for logs, database records, or eval results."""
        return {
            "prompt_workflow": self.workflow,
            "prompt_version": self.version,
            "prompt_requested_version": self.requested_version,
            "prompt_checksum": self.checksum,
        }


class PromptManager:
    """Load and render versioned prompts configured in prompts/config.yaml."""

    def __init__(
        self,
        prompts_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.prompts_dir = (
            Path(prompts_dir)
            if prompts_dir
            else Path(__file__).resolve().parent
        )
        self.config_path = (
            Path(config_path)
            if config_path
            else self.prompts_dir / "config.yaml"
        )
        self.config = self._load_config()

    def load(
        self,
        workflow: str,
        version: str | None = None,
        scenario: str | None = None,
    ) -> PromptTemplate:
        requested_version = version or self._configured_version(workflow, scenario)
        resolved_version = self._resolve_alias(workflow, requested_version)
        filename = self._version_filename(
            workflow, requested_version, resolved_version
        )
        path = self.prompts_dir / workflow / filename

        if not path.is_file():
            raise PromptNotFoundError(
                f"Prompt file not found for {workflow}/{requested_version}: {path}"
            )

        content = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return PromptTemplate(
            workflow=workflow,
            requested_version=requested_version,
            resolved_version=resolved_version,
            content=content,
            path=path,
            checksum=checksum,
        )

    def render(
        self,
        workflow: str,
        variables: Mapping[str, Any],
        version: str | None = None,
        scenario: str | None = None,
    ) -> RenderedPrompt:
        return self.load(
            workflow=workflow,
            version=version,
            scenario=scenario,
        ).render(variables)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            raise PromptConfigurationError(
                f"Prompt config not found: {self.config_path}"
            )

        with self.config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        if not isinstance(config, dict):
            raise PromptConfigurationError(
                f"Prompt config must contain a mapping: {self.config_path}"
            )
        return config

    def _configured_version(self, workflow: str, scenario: str | None) -> str:
        if scenario:
            scenario_config = self.config.get("scenarios", {}).get(scenario)
            if scenario_config is None:
                raise PromptConfigurationError(
                    f"Unknown prompt scenario: {scenario}"
                )
            if workflow in scenario_config:
                return str(scenario_config[workflow])

        defaults = self.config.get("defaults", {})
        if workflow not in defaults:
            raise PromptConfigurationError(
                f"No default version configured for workflow: {workflow}"
            )
        return str(defaults[workflow])

    def _resolve_alias(self, workflow: str, version: str) -> str:
        aliases = self.config.get("aliases", {}).get(workflow, {})
        return str(aliases.get(version, version))

    def _version_filename(
        self,
        workflow: str,
        requested_version: str,
        resolved_version: str,
    ) -> str:
        versions = self.config.get("versions", {}).get(workflow, {})
        if resolved_version not in versions:
            raise PromptConfigurationError(
                f"Unknown version '{resolved_version}' for workflow '{workflow}'"
            )

        # Keep current.txt useful for humans while recording the concrete version.
        if requested_version == "current":
            return "current.txt"
        return str(versions[resolved_version])
