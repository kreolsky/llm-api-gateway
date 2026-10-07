"""Unit tests for src/core/config_manager.py — ConfigManager class."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.core.config_manager import ConfigManager, ReloadOutcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROVIDERS_YAML = "providers:\n  openai:\n    type: openai\n    base_url: https://api.openai.com\n"
MODELS_YAML = "models:\n  gpt-4:\n    provider: openai\n"
USER_KEYS_YAML = "user_keys:\n  test-key:\n    project_name: test\n"

ALL_YAMLS = {
    "providers.yaml": PROVIDERS_YAML,
    "models.yaml": MODELS_YAML,
    "user_keys.yaml": USER_KEYS_YAML,
}


def _multi_open(file_map):
    """Return a side_effect for builtins.open that dispatches by file path suffix."""
    from io import StringIO

    def _side_effect(path, *args, **kwargs):
        for key, content in file_map.items():
            if path.endswith(key):
                sio = StringIO(content)
                # Make it usable as context manager
                sio.__enter__ = lambda s: s
                sio.__exit__ = lambda s, *a: None
                return sio
        raise FileNotFoundError(f"No such file: {path}")

    return _side_effect


def _build_config_manager(file_map=None, env=None):
    """Build ConfigManager with mocked file I/O and env."""
    if file_map is None:
        file_map = ALL_YAMLS
    env_vars = env or {}

    with patch("builtins.open", side_effect=_multi_open(file_map)), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=1000.0), \
         patch.dict("os.environ", env_vars, clear=False), \
         patch("src.core.config_manager.logger"):
        cm = ConfigManager(config_dir="/fake/config")
    return cm


# ===================================================================
# _load_config
# ===================================================================

class TestLoadConfig:

    def test_loads_all_three_yaml_files(self):
        """Loads providers, models, and user_keys from YAML files correctly."""
        cm = _build_config_manager()
        config = cm.get_config()
        assert "openai" in config.providers
        assert "gpt-4" in config.models
        assert "test-key" in config.user_keys
        assert config.providers["openai"].type == "openai"
        assert config.models["gpt-4"].provider == "openai"

    def test_missing_file_fail_on_error_true_raises(self):
        """Missing file with fail_on_error=True raises RuntimeError."""
        file_map = {
            "providers.yaml": PROVIDERS_YAML,
            # models.yaml missing
            "user_keys.yaml": USER_KEYS_YAML,
        }
        with pytest.raises(RuntimeError, match="Critical config file missing"):
            with patch("builtins.open", side_effect=_multi_open(file_map)), \
                 patch("os.path.exists", return_value=True), \
                 patch("os.path.getmtime", return_value=1000.0), \
                 patch("src.core.config_manager.logger"):
                ConfigManager(config_dir="/fake/config")

    def test_missing_file_fail_on_error_false_partial(self):
        """Missing file with fail_on_error=False returns partial config."""
        cm = _build_config_manager()
        # Call _load_config with fail_on_error=False, simulating a missing file
        file_map_missing = {
            "providers.yaml": PROVIDERS_YAML,
            "user_keys.yaml": USER_KEYS_YAML,
        }
        with patch("builtins.open", side_effect=_multi_open(file_map_missing)), \
             patch("src.core.config_manager.logger"):
            partial_config = cm._load_config(fail_on_error=False)
        assert "providers" in partial_config
        assert "user_keys" in partial_config
        assert "models" not in partial_config

    def test_invalid_yaml_fail_on_error_true_raises(self):
        """Invalid YAML with fail_on_error=True raises RuntimeError."""
        # The ": invalid: yaml: [[[" is actually parseable by some YAML parsers
        # so let's force a YAML error via mock
        cm = _build_config_manager()

        def _open_with_bad_yaml(path, *args, **kwargs):
            from io import StringIO
            for key, content in {
                "providers.yaml": PROVIDERS_YAML,
                "user_keys.yaml": USER_KEYS_YAML,
            }.items():
                if path.endswith(key):
                    sio = StringIO(content)
                    sio.__enter__ = lambda s: s
                    sio.__exit__ = lambda s, *a: None
                    return sio
            if path.endswith("models.yaml"):
                sio = StringIO("bad yaml")
                sio.__enter__ = lambda s: s
                sio.__exit__ = lambda s, *a: None
                return sio
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=_open_with_bad_yaml), \
             patch("yaml.safe_load", side_effect=[
                 yaml.safe_load(PROVIDERS_YAML),
                 yaml.YAMLError("bad yaml"),
             ]), \
             patch("src.core.config_manager.logger"), pytest.raises(RuntimeError, match="Failed to parse config"):
            cm._load_config(fail_on_error=True)


# ===================================================================
# Top-level key validation
# ===================================================================

class TestTopLevelKeyValidation:

    def test_flat_yaml_without_wrapper_raises(self):
        """A flat providers.yaml (no top-level 'providers:' key) raises RuntimeError."""
        # Missing the 'providers:' wrapper — just a mapping of provider entries
        flat_providers = (
            "openai:\n  type: openai\n  base_url: https://api.openai.com\n"
        )
        file_map = {
            "providers.yaml": flat_providers,
            "models.yaml": MODELS_YAML,
            "user_keys.yaml": USER_KEYS_YAML,
        }
        with patch("builtins.open", side_effect=_multi_open(file_map)), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getmtime", return_value=1000.0), \
             patch("src.core.config_manager.logger"):
            with pytest.raises(RuntimeError, match="missing top-level 'providers:'"):
                ConfigManager(config_dir="/fake/config")

    def test_flat_yaml_at_reload_warns_skips(self):
        """At reload (fail_on_error=False) a malformed file is skipped, others load."""
        flat_models = "gpt-4:\n  provider: openai\n"
        file_map = {
            "providers.yaml": PROVIDERS_YAML,
            "models.yaml": flat_models,
            "user_keys.yaml": USER_KEYS_YAML,
        }
        cm = _build_config_manager()
        with patch("builtins.open", side_effect=_multi_open(file_map)), \
             patch("src.core.config_manager.logger"):
            config = cm._load_config(fail_on_error=False)
        assert "providers" in config
        assert "user_keys" in config
        # models.yaml had no top-level 'models:' → skipped
        assert "models" not in config


# ===================================================================
# reload_config
# ===================================================================

class TestReloadConfig:

    @pytest.mark.asyncio
    async def test_invokes_callbacks_on_change(self):
        """reload_config invokes registered callbacks when config changes."""
        cm = _build_config_manager()
        callback = AsyncMock()
        cm.add_reload_callback(callback, name="test_cb")

        # Reload with full config
        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            await cm.reload_config()

        cm_full_config = cm.get_config()
        callback.assert_awaited_once_with(cm_full_config)

    @pytest.mark.asyncio
    async def test_rejects_partial_config_keeps_previous(self):
        """Partial config is rejected; previous config is kept."""
        cm = _build_config_manager()
        original_config = cm.get_config()
        callback = AsyncMock()
        cm.add_reload_callback(callback, name="test_cb")

        # Reload with only providers (missing models and user_keys)
        partial_map = {"providers.yaml": PROVIDERS_YAML}
        with patch("builtins.open", side_effect=_multi_open(partial_map)), \
             patch("src.core.config_manager.logger"):
            await cm.reload_config()

        # Callback should NOT be invoked
        callback.assert_not_called()
        # Config should remain unchanged
        assert cm.get_config() is original_config

    @pytest.mark.asyncio
    async def test_callback_failure_keeps_previous_config(self):
        """When a callback raises, self.config is NOT swapped (old config retained)."""
        cm = _build_config_manager()
        original_config = cm.get_config()
        failing_cb = AsyncMock(side_effect=RuntimeError("boom"))
        cm.add_reload_callback(failing_cb, name="failing")

        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            await cm.reload_config()

        failing_cb.assert_awaited_once()
        assert cm.get_config() is original_config  # old config retained


class TestHardValidationVeto:
    """A hard config error (parse_config's ConfigError) refuses to start and
    vetoes a reload with the previous config still serving."""

    BAD_DIALECT_YAML = PROVIDERS_YAML + "    reasoning_dialect: typo\n"

    def test_unknown_provider_type_refuses_to_start(self):
        from src.core.config_schema import ConfigError
        file_map = {**ALL_YAMLS,
                    "providers.yaml": "providers:\n  p:\n    type: bedrock\n    base_url: https://x\n"}
        with pytest.raises(ConfigError) as exc_info:
            _build_config_manager(file_map)
        assert "providers.p" in str(exc_info.value)
        assert "bedrock" in str(exc_info.value)

    def test_anthropic_provider_type_parses(self):
        """`type: anthropic` is a known second protocol, not a startup refusal."""
        file_map = {**ALL_YAMLS,
                    "providers.yaml": ("providers:\n  claude:\n    type: anthropic\n"
                                       "    base_url: https://api.anthropic.com\n"
                                       "    api_key_env: ANTHROPIC_API_KEY\n")}
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-x"}, clear=False):
            cm = _build_config_manager(file_map)
        entry = cm.get_config().providers["claude"]
        assert entry.type == "anthropic"
        assert entry.base_url == "https://api.anthropic.com"

    @pytest.mark.asyncio
    async def test_bad_dialect_vetoes_reload_and_keeps_old_config(self):
        cm = _build_config_manager()
        original_config = cm.get_config()
        pre = AsyncMock()
        post = AsyncMock()
        cm.add_reload_callback(pre, name="pre")
        cm.add_post_swap_callback(post, name="post")

        file_map = {**ALL_YAMLS, "providers.yaml": self.BAD_DIALECT_YAML}
        with patch("builtins.open", side_effect=_multi_open(file_map)), \
             patch("src.core.config_manager.logger") as mock_logger:
            assert await cm.reload_config() is ReloadOutcome.REJECTED

        assert cm.get_config() is original_config
        pre.assert_not_called()
        post.assert_not_called()
        rejection = str(mock_logger.error.call_args)
        assert "reload rejected" in rejection
        assert "reasoning_dialect" in rejection

    def test_every_bad_entry_is_listed(self):
        """One ConfigError names every broken entry, not only the first."""
        from src.core.config_schema import ConfigError, parse_config
        with pytest.raises(ConfigError) as exc_info:
            parse_config({
                "providers": {"a": {"type": "openai", "identity": "x"},
                              "b": {"type": "openai", "headers": {"Host": "h"}}},
                "models": {"m": "not-a-mapping"},
            })
        message = str(exc_info.value)
        assert "providers.a" in message
        assert "providers.b" in message
        assert "models.m" in message


class TestMaxConcurrentValidation:
    """max_concurrent is a hard check in parse_provider: only a positive int
    (or absent/null) parses — a quoted "3" would silently disable the pool's
    gate and `true` (bool is an int subclass) would build a Semaphore(1)."""

    @pytest.mark.parametrize("bad", ["3", True, 0, -1])
    def test_bad_value_is_config_error(self, bad):
        from src.core.config_schema import ConfigError, parse_provider
        with pytest.raises(ConfigError, match="max_concurrent"):
            parse_provider({"type": "openai", "max_concurrent": bad})

    def test_positive_int_parses(self):
        from src.core.config_schema import parse_provider
        assert parse_provider({"type": "openai", "max_concurrent": 3}).max_concurrent == 3

    def test_absent_and_null_parse(self):
        from src.core.config_schema import parse_provider
        assert parse_provider({"type": "openai"}).max_concurrent is None
        assert parse_provider({"type": "openai", "max_concurrent": None}).max_concurrent is None

    def test_bad_value_in_a_section_names_the_provider(self):
        """Through parse_config the error is wrapped with the entry's name —
        an operator sees `providers.orange`, not a bare value complaint."""
        from src.core.config_schema import ConfigError, parse_config
        with pytest.raises(ConfigError) as exc_info:
            parse_config({
                "providers": {"orange": {"type": "openai", "max_concurrent": "3"}},
                "models": {"m": {"provider": "orange"}},
                "user_keys": {},
            })
        assert "providers.orange" in str(exc_info.value)
        assert "max_concurrent" in str(exc_info.value)


class TestModelInfoCompat:
    """model_info `compat` (manual-only pi-ai compat flags, rendered as stored):
    a mapping of string keys to bool/str/int scalars; anything else is warned
    about and dropped, soft like the other model_info keys."""

    def _model_info_config(self, compat):
        from src.core.config_schema import parse_config
        return parse_config({
            "providers": {"p": {"type": "openai"}},
            "models": {"m": {"provider": "p"}},
            "model_info": {"m": {"compat": compat}},
        })

    def test_scalar_compat_kept_verbatim(self):
        with patch("src.core.config_schema.logger") as mock_logger:
            config = self._model_info_config({"forceAdaptiveThinking": True,
                                              "supportsMidConvoEffort": True,
                                              "title": "deep thinker", "rank": 2})
        assert config.model_info["m"]["compat"] == {
            "forceAdaptiveThinking": True, "supportsMidConvoEffort": True,
            "title": "deep thinker", "rank": 2,
        }
        mock_logger.warning.assert_not_called()

    def test_non_scalar_values_dropped_with_warning(self):
        with patch("src.core.config_schema.logger") as mock_logger:
            config = self._model_info_config({"ok": True, "bad-list": [1], "bad-dict": {},
                                              "bad-none": None})
        assert config.model_info["m"]["compat"] == {"ok": True}
        warnings = [c[0][0] for c in mock_logger.warning.call_args_list]
        assert any("compat" in w and "bad-list" in w for w in warnings)

    def test_non_string_key_dropped(self):
        with patch("src.core.config_schema.logger"):
            config = self._model_info_config({"ok": 1, 3: "int key"})
        assert config.model_info["m"]["compat"] == {"ok": 1}

    def test_non_mapping_compat_dropped_with_warning(self):
        with patch("src.core.config_schema.logger") as mock_logger:
            config = self._model_info_config(["flag"])
        assert "compat" not in config.model_info["m"]
        assert mock_logger.warning.called


class TestAnthropicCompatAllowList:
    """compat on an anthropic-type model is filtered to dsh's offered set
    (ANTHROPIC_COMPAT_KEYS — the anthropic-messages ANTHROPIC_COMPAT_GATE
    `offer` keys): a key outside the set is warned and dropped, because dsh
    fails a model's resolution on a withheld key, an unknown key, or a key
    offered only by another protocol. On any other provider type compat is
    kept as stored (TestModelInfoCompat pins that branch)."""

    OFFERED = {
        "forceAdaptiveThinking": True,
        "supportsStrictTools": True,
        "supportsEagerToolInputStreaming": True,
        "supportsCacheControlOnTools": True,
        "supportsTemperature": True,
        "allowEmptySignature": True,
        "supportsLongCacheRetention": True,
    }
    DROPPED = {
        "supportsMidConvoEffort": True,      # withheld by the gate
        "supportsDeveloperRole": True,       # offered only on openai-completions
        "forceAdaptiveThinkng": True,        # typo
    }

    def _parse(self, provider_type="anthropic", model_info=None, models=None, providers=None):
        from src.core.config_schema import parse_config
        raw = {
            "providers": providers if providers is not None else {"p": {"type": provider_type}},
            "models": models if models is not None else {"m": {"provider": "p"}},
        }
        if model_info is not None:
            raw["model_info"] = model_info
        with patch("src.core.config_schema.logger") as mock_logger:
            config = parse_config(raw)
        return config, mock_logger

    @staticmethod
    def _warnings(mock_logger):
        return [c[0][0] for c in mock_logger.warning.call_args_list]

    @classmethod
    def _checklist_warnings(cls, mock_logger):
        return [w for w in cls._warnings(mock_logger) if "checklist" in w]

    def test_every_offered_key_kept_without_warning(self):
        entry = {"context_length": 524288, "max_completion_tokens": 131072,
                 "compat": dict(self.OFFERED)}
        config, mock_logger = self._parse(model_info={"m": entry})
        assert config.model_info["m"]["compat"] == self.OFFERED
        assert mock_logger.warning.call_args_list == []

    def test_withheld_foreign_and_typo_keys_dropped_with_warning(self):
        compat = {"forceAdaptiveThinking": True, **self.DROPPED}
        config, mock_logger = self._parse(
            model_info={"m": {"context_length": 1, "max_completion_tokens": 1,
                              "compat": compat}})
        assert config.model_info["m"]["compat"] == {"forceAdaptiveThinking": True}
        for key in self.DROPPED:
            assert any(key in w and "'m'" in w for w in self._warnings(mock_logger)), key

    def test_compat_whose_every_key_dropped_disappears(self):
        config, _ = self._parse(
            model_info={"m": {"context_length": 1, "max_completion_tokens": 1,
                              "compat": {"supportsMidConvoEffort": True}}})
        assert "compat" not in config.model_info["m"]

    def test_same_keys_on_openai_provider_kept_without_warning(self):
        compat = {"forceAdaptiveThinking": True, **self.DROPPED}
        config, mock_logger = self._parse(
            provider_type="openai", model_info={"m": {"compat": compat}})
        assert config.model_info["m"]["compat"] == compat
        assert mock_logger.warning.call_args_list == []

    # -- checklist: context_length / max_completion_tokens on anthropic models

    def test_missing_context_length_warns_once(self):
        _, mock_logger = self._parse(
            model_info={"m": {"max_completion_tokens": 4096}})
        checklist = self._checklist_warnings(mock_logger)
        assert len(checklist) == 1
        assert "'m'" in checklist[0] and "context_length" in checklist[0]

    def test_no_model_info_entry_warns_for_both_keys(self):
        _, mock_logger = self._parse()  # no model_info section at all
        checklist = self._checklist_warnings(mock_logger)
        assert len(checklist) == 2
        assert any("context_length" in w for w in checklist)
        assert any("max_completion_tokens" in w for w in checklist)

    def test_complete_anthropic_entry_no_checklist_warning(self):
        _, mock_logger = self._parse(
            model_info={"m": {"context_length": 524288, "max_completion_tokens": 131072}})
        assert self._checklist_warnings(mock_logger) == []

    def test_incomplete_entry_on_openai_no_checklist_warning(self):
        _, mock_logger = self._parse(
            provider_type="openai", model_info={"m": {"max_completion_tokens": 4096}})
        assert self._checklist_warnings(mock_logger) == []

    def test_orphan_entry_no_checklist_warning(self):
        # 'ghost' is not a models.yaml model: the existing orphan warning fires,
        # but the checklist iterates MODELS, never model_info entries.
        _, mock_logger = self._parse(
            model_info={"m": {"context_length": 1, "max_completion_tokens": 1},
                        "ghost": {"max_completion_tokens": 1}})
        assert self._checklist_warnings(mock_logger) == []
        assert any("ghost" in w for w in self._warnings(mock_logger))

    def test_provider_none_model_no_checklist_warning(self):
        _, mock_logger = self._parse(
            models={"m": {"provider": None}},
            model_info={"m": {"max_completion_tokens": 1}})
        assert self._checklist_warnings(mock_logger) == []

    def test_unknown_provider_model_no_checklist_warning(self):
        # 'ghost' provider is not in providers.yaml — _warn_dangling_references
        # owns that warning; the checklist treats the model as not typed.
        _, mock_logger = self._parse(
            models={"m": {"provider": "ghost"}},
            providers={"p": {"type": "anthropic"}},
            model_info={"m": {"max_completion_tokens": 1}})
        assert self._checklist_warnings(mock_logger) == []
        assert any("unknown provider 'ghost'" in w for w in self._warnings(mock_logger))

    def test_repo_configs_parse_with_compat_and_no_warnings(self):
        """The repo's own config files: deepseek-a/flash carries the
        production-settled compat line and numbers, and loading them logs no
        compat or checklist warning."""
        from src.core.config_schema import parse_config
        base = Path(__file__).resolve().parents[2] / "config"
        raw = {}
        for fname, key in (("providers.yaml", "providers"), ("models.yaml", "models"),
                           ("user_keys.yaml", "user_keys"), ("model_info.yaml", "model_info")):
            with open(base / fname) as f:
                loaded = yaml.safe_load(f) or {}
            raw[key] = loaded.get(key) or {}
        with patch("src.core.config_schema.logger") as mock_logger:
            config = parse_config(raw)
        entry = config.model_info["deepseek-a/flash"]
        assert entry["compat"] == {"forceAdaptiveThinking": True}
        assert entry["context_length"] == 524288
        assert entry["max_completion_tokens"] == 131072
        offenders = [w for w in self._warnings(mock_logger)
                     if "compat" in w or "checklist" in w]
        assert offenders == []


class TestCrossReferenceWarnings:
    """Cross-file references are a soft check in parse_config: a model whose
    provider is not a providers.yaml key and a key whose allowed_models names
    an unknown model each warn once — nothing dropped, no veto (the
    model_info orphan warning's class)."""

    def _parse(self, raw):
        from src.core.config_schema import parse_config
        with patch("src.core.config_schema.logger") as mock_logger:
            config = parse_config(raw)
        return config, mock_logger

    def test_unknown_provider_and_unknown_allowed_model_each_warn_once(self):
        config, mock_logger = self._parse({
            "providers": {"openai": {"type": "openai"}},
            "models": {
                "gpt-4": {"provider": "openai"},
                "ghost-backed": {"provider": "ghost"},
            },
            "user_keys": {
                "full-key": {"api_key": "k1", "allowed_models": ["gpt-4"]},
                "ghost-key": {"api_key": "k2", "allowed_models": ["ghost-model"]},
            },
        })
        # Soft: nothing dropped, no veto.
        assert "ghost-backed" in config.models
        assert "ghost-key" in config.user_keys
        warnings = [str(call) for call in mock_logger.warning.call_args_list]
        assert len(warnings) == 2
        assert any("ghost-backed" in w and "unknown provider 'ghost'" in w
                   for w in warnings)
        assert any("ghost-key" in w and "ghost-model" in w for w in warnings)

    def test_matching_references_do_not_warn(self):
        _, mock_logger = self._parse({
            "providers": {"openai": {"type": "openai"}},
            "models": {"gpt-4": {"provider": "openai"}},
            "user_keys": {"full-key": {"api_key": "k1", "allowed_models": ["gpt-4"]}},
        })
        assert mock_logger.warning.call_args_list == []

    def test_empty_referenced_section_does_not_warn(self):
        """A section's references are checked only when it is non-empty: an
        empty providers/models section never reaches parse_config in
        production (rejected before parsing), and warning per entry there
        would only re-describe the missing section."""
        # providers empty → the model's dangling provider is not warned.
        _, mock_logger = self._parse({
            "models": {"m": {"provider": "ghost"}},
        })
        assert mock_logger.warning.call_args_list == []
        # models empty → the key's dangling allowed_model is not warned.
        _, mock_logger = self._parse({
            "user_keys": {"k": {"api_key": "x", "allowed_models": ["ghost"]}},
        })
        assert mock_logger.warning.call_args_list == []


# ===================================================================
# Typed Settings snapshot
# ===================================================================

class TestSettings:
    """ConfigManager exposes a frozen, typed Settings built once at construction."""

    def _cm_with_env(self, env):
        with patch.dict("os.environ", env, clear=True):
            return _build_config_manager()

    def test_settings_object_exposed(self):
        """cm.settings is a Settings carrying the resolved env values."""
        from src.core.config_manager import Settings
        cm = self._cm_with_env({"QUEUE_WAIT_TIMEOUT": "7.5"})
        assert isinstance(cm.settings, Settings)
        assert cm.settings.queue_wait_timeout == 7.5

    def test_settings_defaults(self):
        """A bare Settings() carries the documented no-config defaults."""
        from src.core.config_manager import Settings
        s = Settings()
        assert s.queue_wait_timeout == 30.0
        assert s.stream_read_timeout == 300.0
        assert s.provider_max_retries == 3
        assert s.provider_retry_base_delay == 1.0
        assert s.provider_retry_max_delay == 30.0
        assert s.openai_connect_timeout == 60.0
        assert s.openai_transcription_timeout == 3600.0
        assert s.openai_embeddings_read_timeout == 30.0
        assert s.httpx_max_connections == 100
        assert s.default_stt_model == "stt/dummy"

    def test_settings_is_frozen(self):
        """Settings is frozen — a running process cannot mutate its knobs."""
        from dataclasses import FrozenInstanceError

        from src.core.config_manager import Settings
        s = Settings()
        with pytest.raises(FrozenInstanceError):
            s.queue_wait_timeout = 1.0

    def test_no_duck_typed_attribute_access(self):
        """Env-backed knobs are NOT attributes of ConfigManager anymore —
        only cm.settings.<name> resolves; the bare name is an AttributeError."""
        cm = _build_config_manager()
        assert cm.settings.stream_read_timeout == 300.0
        with pytest.raises(AttributeError):
            cm.stream_read_timeout

    def test_env_resolved_once_into_settings(self):
        """Settings is resolved at construction; a later env change is ignored."""
        with patch.dict("os.environ", {"STREAM_READ_TIMEOUT": "111"}, clear=True):
            cm = _build_config_manager()
        with patch.dict("os.environ", {"STREAM_READ_TIMEOUT": "999"}, clear=True):
            assert cm.settings.stream_read_timeout == 111.0

    def test_malformed_env_refuses_to_start(self):
        """A malformed numeric env var refuses to start (the module ARCH
        promises fail-fast) instead of silently degrading to the default."""
        with patch.dict("os.environ", {"QUEUE_WAIT_TIMEOUT": "not-a-number"}, clear=True):
            with pytest.raises(RuntimeError, match="QUEUE_WAIT_TIMEOUT"):
                _build_config_manager()


class TestPropertyGetters:
    """Env-backed settings are resolved once, when the ConfigManager is built."""

    def _cm_with_env(self, env):
        with patch.dict("os.environ", env, clear=True):
            return _build_config_manager()

    def test_httpx_max_connections_default(self):
        """httpx_max_connections returns default 100."""
        assert self._cm_with_env({}).settings.httpx_max_connections == 100

    def test_httpx_max_connections_from_env(self):
        """httpx_max_connections reads from env var."""
        assert self._cm_with_env({"HTTPX_MAX_CONNECTIONS": "200"}).settings.httpx_max_connections == 200

    def test_provider_max_retries_default(self):
        """provider_max_retries defaults to 3."""
        assert self._cm_with_env({}).settings.provider_max_retries == 3

    def test_provider_retry_base_delay_from_env(self):
        """provider_retry_base_delay reads env var as float."""
        assert self._cm_with_env({"PROVIDER_RETRY_BASE_DELAY": "2.5"}).settings.provider_retry_base_delay == 2.5

    def test_httpx_pool_timeout_default(self):
        """httpx_pool_timeout defaults to 5.0."""
        assert self._cm_with_env({}).settings.httpx_pool_timeout == 5.0

    def test_config_reload_interval_from_env(self):
        """config_reload_interval reads from env."""
        assert self._cm_with_env({"CONFIG_RELOAD_INTERVAL": "10"}).settings.config_reload_interval == 10

    def test_stream_read_timeout_default(self):
        """stream_read_timeout defaults to 300.0."""
        assert self._cm_with_env({}).settings.stream_read_timeout == 300.0

    def test_stream_read_timeout_from_env(self):
        """stream_read_timeout reads from env as float."""
        assert self._cm_with_env({"STREAM_READ_TIMEOUT": "600"}).settings.stream_read_timeout == 600.0

    def test_default_stt_model_default(self):
        """default_stt_model defaults to stt/dummy."""
        assert self._cm_with_env({}).settings.default_stt_model == "stt/dummy"

    def test_default_stt_model_from_env(self):
        """default_stt_model reads from env."""
        assert self._cm_with_env({"DEFAULT_STT_MODEL": "stt/custom"}).settings.default_stt_model == "stt/custom"

    def test_model_cache_settings(self):
        """Model-cache settings resolve from env with the documented defaults."""
        cm = self._cm_with_env({})
        assert cm.settings.model_cache_enabled is True
        assert cm.settings.model_cache_refresh_interval == 3600
        assert cm.settings.model_cache_path == "data/model_cache.json"

        cm = self._cm_with_env({"MODEL_CACHE_ENABLED": "false", "MODEL_CACHE_PATH": "/tmp/c.json"})
        assert cm.settings.model_cache_enabled is False
        assert cm.settings.model_cache_path == "/tmp/c.json"

    def test_model_cache_ttl_is_gone(self):
        """MODEL_CACHE_TTL was a dead knob (stale-if-error contradicts a TTL) — removed."""
        with pytest.raises(AttributeError):
            self._cm_with_env({"MODEL_CACHE_TTL": "1"}).settings.model_cache_ttl

    def test_usage_db_path_settings(self):
        """usage_db_path resolves from env with the documented default."""
        cm = self._cm_with_env({})
        assert cm.settings.usage_db_path == "data/usage.db"

        cm = self._cm_with_env({"USAGE_DB_PATH": "/tmp/usage.db"})
        assert cm.settings.usage_db_path == "/tmp/usage.db"


class TestEnvSettingsResolvedOnce:
    """Values are frozen at construction; a malformed value refuses to start."""

    def test_later_env_change_is_ignored(self):
        """Changing the env after construction does not change a resolved setting."""
        with patch.dict("os.environ", {"STREAM_READ_TIMEOUT": "111"}, clear=True):
            cm = _build_config_manager()
        with patch.dict("os.environ", {"STREAM_READ_TIMEOUT": "999"}, clear=True):
            assert cm.settings.stream_read_timeout == 111.0

    def test_malformed_number_names_variable_and_raw_value(self):
        """A malformed numeric env var raises RuntimeError naming the variable
        AND its raw value, so a crash log is enough to fix the .env line."""
        with patch.dict("os.environ", {"QUEUE_WAIT_TIMEOUT": "not-a-number"}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                _build_config_manager()
        message = str(exc_info.value)
        assert "QUEUE_WAIT_TIMEOUT" in message
        assert "'not-a-number'" in message

    def test_malformed_bool_refuses_to_start(self):
        """`MODEL_CACHE_ENABLED=yes` raises instead of reading as False —
        anything-not-`true` used to disable the model cache in silence."""
        with patch.dict("os.environ", {"MODEL_CACHE_ENABLED": "yes"}, clear=True):
            with pytest.raises(RuntimeError, match="MODEL_CACHE_ENABLED"):
                _build_config_manager()

    def test_every_malformed_variable_is_named_at_once(self):
        """One RuntimeError lists every malformed variable — the same
        collect-then-raise shape as parse_config, fix them all in one pass."""
        with patch.dict("os.environ", {"QUEUE_WAIT_TIMEOUT": "abc",
                                       "MODEL_CACHE_ENABLED": "yes"}, clear=True):
            with pytest.raises(RuntimeError) as exc_info:
                _build_config_manager()
        message = str(exc_info.value)
        assert "QUEUE_WAIT_TIMEOUT" in message and "'abc'" in message
        assert "MODEL_CACHE_ENABLED" in message and "'yes'" in message

    def test_bool_true_is_case_and_whitespace_tolerant(self):
        """` TRUE ` parses as True — the strip/lower tolerance is kept."""
        with patch.dict("os.environ", {"MODEL_CACHE_ENABLED": " TRUE "}, clear=True):
            cm = _build_config_manager()
        assert cm.settings.model_cache_enabled is True

    def test_empty_value_means_unset(self):
        """An empty value is an unset knob, not a malformed one: bool and
        numeric fields keep their defaults (`MODEL_CACHE_ENABLED=` in a .env
        file must not refuse to start)."""
        with patch.dict("os.environ", {"MODEL_CACHE_ENABLED": "",
                                       "QUEUE_WAIT_TIMEOUT": ""}, clear=True):
            cm = _build_config_manager()
        assert cm.settings.model_cache_enabled is True
        assert cm.settings.queue_wait_timeout == 30.0

    def test_unknown_attribute_still_raises(self):
        """Only cm.settings exists; anything else is a normal AttributeError."""
        cm = _build_config_manager()
        with pytest.raises(AttributeError):
            cm.definitely_not_a_setting


# ===================================================================
# _assert_config_complete (fail-fast validation)
# ===================================================================

class TestAssertConfigComplete:

    def test_empty_section_raises(self):
        """Empty providers section raises RuntimeError (fail-fast)."""
        from src.core.config_manager import ConfigManager
        with pytest.raises(RuntimeError, match="missing or empty"):
            ConfigManager._assert_config_complete({"providers": {}, "models": {"m": {}}, "user_keys": {"k": {}}})

    def test_missing_section_raises(self):
        """Missing user_keys section raises RuntimeError."""
        from src.core.config_manager import ConfigManager
        with pytest.raises(RuntimeError, match="user_keys"):
            ConfigManager._assert_config_complete({"providers": {"p": {}}, "models": {"m": {}}})

    def test_complete_config_passes(self):
        """All three non-empty sections pass validation."""
        ConfigManager._assert_config_complete(
            {"providers": {"p": {}}, "models": {"m": {}}, "user_keys": {"k": {}}}
        )

    def test_missing_sections_lists_all(self):
        """_missing_sections names every empty-or-absent section, not just the first."""
        assert ConfigManager._missing_sections({"providers": {}}) == ["providers", "models", "user_keys"]


# ===================================================================
# add_reload_callback
# ===================================================================

class TestAddReloadCallback:

    @pytest.mark.asyncio
    async def test_callback_called_on_reload(self):
        """Registered callback is called on successful reload."""
        cm = _build_config_manager()
        cb = AsyncMock()
        cm.add_reload_callback(cb, name="cb")

        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            await cm.reload_config()

        cb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Multiple registered callbacks are all called."""
        cm = _build_config_manager()
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        cm.add_reload_callback(cb1, name="cb1")
        cm.add_reload_callback(cb2, name="cb2")

        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            await cm.reload_config()

        cb1.assert_awaited_once()
        cb2.assert_awaited_once()


# ===================================================================
# add_post_swap_callback (two-phase reload)
# ===================================================================

class TestAddPostSwapCallback:
    """Post-swap callbacks run AFTER self.config is swapped and cannot abort it."""

    MODELS_V2 = "models:\n  gpt-4:\n    provider: openai\n  gpt-5:\n    provider: openai\n"

    @pytest.mark.asyncio
    async def test_runs_after_config_swap(self):
        """Inside a post-swap callback get_config() already returns the NEW
        config; inside a pre-swap callback it still returns the old one."""
        cm = _build_config_manager()
        seen = {}

        async def pre_cb(new_config):
            seen["pre"] = "gpt-5" in cm.get_config().models

        async def post_cb(new_config):
            seen["post"] = "gpt-5" in cm.get_config().models

        cm.add_reload_callback(pre_cb, name="pre")
        cm.add_post_swap_callback(post_cb, name="post")

        file_map = {
            "providers.yaml": PROVIDERS_YAML,
            "models.yaml": self.MODELS_V2,
            "user_keys.yaml": USER_KEYS_YAML,
        }
        with patch("builtins.open", side_effect=_multi_open(file_map)), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.APPLIED

        assert seen == {"pre": False, "post": True}

    @pytest.mark.asyncio
    async def test_post_swap_failure_returns_post_swap_failed_but_keeps_new_config(self):
        """A failing post-swap callback cannot un-publish the swap: the config
        is already live and there is nothing to roll back to. The reload still
        reports POST_SWAP_FAILED so _poll_once does not treat the on-disk
        state as consumed — the next tick retries the reload (cheap with
        provider instance reuse)."""
        cm = _build_config_manager()
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        cm.add_post_swap_callback(failing, name="failing_post")

        file_map = {
            "providers.yaml": PROVIDERS_YAML,
            "models.yaml": self.MODELS_V2,
            "user_keys.yaml": USER_KEYS_YAML,
        }
        with patch("builtins.open", side_effect=_multi_open(file_map)), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.POST_SWAP_FAILED

        failing.assert_awaited_once()
        # The swap itself stays published.
        assert "gpt-5" in cm.get_config().models

    @pytest.mark.asyncio
    async def test_post_swap_failure_never_logs_the_plain_success_line(self):
        """The reload_complete line is a WARNING carrying post_swap_failed=True
        when a post-swap callback raised. The retry loop reprints this line
        every poll interval, so an INFO 'Configuration reloaded' would read as
        'all good' forever while the router serves a half-applied state."""
        cm = _build_config_manager()
        cm.add_post_swap_callback(AsyncMock(side_effect=RuntimeError("boom")), name="failing_post")

        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger") as mock_logger:
            assert await cm.reload_config() is ReloadOutcome.POST_SWAP_FAILED

        completions = [
            call for call in mock_logger.warning.call_args_list
            if call.kwargs.get("extra", {}).get("config", {}).get("operation") == "reload_complete"
        ]
        assert len(completions) == 1, "the completion line must be logged once, at warning"
        assert completions[0].kwargs["extra"]["config"]["post_swap_failed"] is True
        # And never as the plain success line.
        assert not [
            call for call in mock_logger.info.call_args_list
            if call.kwargs.get("extra", {}).get("config", {}).get("operation") == "reload_complete"
        ]

    @pytest.mark.asyncio
    async def test_clean_reload_logs_the_success_line_at_info(self):
        """The failing branch above is only meaningful if the clean path still
        logs reload_complete at INFO with the flag false."""
        cm = _build_config_manager()
        cm.add_post_swap_callback(AsyncMock(), name="fine_post")

        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger") as mock_logger:
            assert await cm.reload_config() is ReloadOutcome.APPLIED

        completions = [
            call for call in mock_logger.info.call_args_list
            if call.kwargs.get("extra", {}).get("config", {}).get("operation") == "reload_complete"
        ]
        assert len(completions) == 1
        assert completions[0].kwargs["extra"]["config"]["post_swap_failed"] is False
        assert not mock_logger.warning.call_args_list

    @pytest.mark.asyncio
    async def test_pre_swap_failure_skips_post_swap(self):
        """A pre-swap (aborting) callback failure prevents the swap AND the
        post-swap callbacks from running at all."""
        cm = _build_config_manager()
        original_config = cm.get_config()
        post = AsyncMock()
        cm.add_reload_callback(AsyncMock(side_effect=RuntimeError("boom")), name="failing_pre")
        cm.add_post_swap_callback(post, name="post")

        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.REJECTED

        post.assert_not_called()
        assert cm.get_config() is original_config


# ===================================================================
# mtime bookkeeping: what each reload outcome commits
# ===================================================================

class TestReloadOutcomeMtimes:
    """_poll_once commits last_mtimes for APPLIED and REJECTED alike, and
    leaves them uncommitted only for POST_SWAP_FAILED.

    A rejection is final for those file bytes: parsing is deterministic and
    the env is frozen per process, so a retry only repeats the rejection —
    its one log line per file change is the signal. A post-swap failure is
    different: the config is already published and the derived cache must
    catch up, so the next poll retries.
    """

    @pytest.mark.asyncio
    async def test_reload_config_reports_success(self):
        """reload_config returns APPLIED when the new config is applied."""
        cm = _build_config_manager()
        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.APPLIED

    @pytest.mark.asyncio
    async def test_reload_config_reports_callback_failure(self):
        """reload_config returns REJECTED when a pre-swap callback vetoes the new config."""
        cm = _build_config_manager()
        cm.add_reload_callback(AsyncMock(side_effect=RuntimeError("boom")), name="failing")
        with patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.REJECTED

    @pytest.mark.asyncio
    async def test_reload_config_reports_partial_rejection(self):
        """reload_config returns REJECTED when the loaded config is incomplete."""
        cm = _build_config_manager()
        with patch("builtins.open", side_effect=_multi_open({"providers.yaml": PROVIDERS_YAML})), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.REJECTED

    @pytest.mark.asyncio
    async def test_config_error_reload_returns_rejected_and_commits_mtimes(self):
        """A ConfigError reload returns REJECTED and commits last_mtimes —
        the same bytes would give the same verdict on every retry."""
        cm = _build_config_manager()
        file_map = {**ALL_YAMLS, "providers.yaml": PROVIDERS_YAML + "    reasoning_dialect: typo\n"}

        with patch("os.path.getmtime", return_value=2000.0), \
             patch("builtins.open", side_effect=_multi_open(file_map)), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.REJECTED
            assert await cm._poll_once() is True    # change detected, reload re-attempted
            assert await cm._poll_once() is False   # the rejection is final for these bytes

        assert set(cm.last_mtimes.values()) == {2000.0}

    @pytest.mark.asyncio
    async def test_missing_section_reload_returns_rejected_and_commits_mtimes(self):
        """An incomplete on-disk config returns REJECTED and commits
        last_mtimes — the same finality as a ConfigError."""
        cm = _build_config_manager()

        with patch("os.path.getmtime", return_value=2000.0), \
             patch("builtins.open", side_effect=_multi_open({"providers.yaml": PROVIDERS_YAML})), \
             patch("src.core.config_manager.logger"):
            assert await cm.reload_config() is ReloadOutcome.REJECTED
            assert await cm._poll_once() is True    # change detected, reload re-attempted
            assert await cm._poll_once() is False   # the rejection is final for these bytes

        assert set(cm.last_mtimes.values()) == {2000.0}

    @pytest.mark.asyncio
    async def test_vetoed_reload_commits_mtimes(self):
        """A vetoed reload is final for those file bytes: last_mtimes are
        committed and the second _poll_once returns False (no retry)."""
        cm = _build_config_manager()
        cm.add_reload_callback(AsyncMock(side_effect=RuntimeError("boom")), name="failing")

        with patch("os.path.getmtime", return_value=2000.0), \
             patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            assert await cm._poll_once() is True    # change detected, reload attempted
            assert await cm._poll_once() is False   # rejected: final, not retried

        assert set(cm.last_mtimes.values()) == {2000.0}

    @pytest.mark.asyncio
    async def test_post_swap_failure_leaves_mtimes_uncommitted(self):
        """A post-swap failure is not 'applied': last_mtimes stay uncommitted
        so the next _poll_once retries the reload. Without this, a provider
        added by a reload whose publish callback failed would 404 until the
        file was touched again — the config swap kept, the cache half-applied,
        nothing retried."""
        cm = _build_config_manager()
        before = dict(cm.last_mtimes)
        cm.add_post_swap_callback(AsyncMock(side_effect=RuntimeError("boom")), name="failing_post")

        with patch("os.path.getmtime", return_value=2000.0), \
             patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            assert await cm._poll_once() is True   # change detected, reload attempted
            assert await cm._poll_once() is True   # still pending: retried

        assert cm.last_mtimes == before

    @pytest.mark.asyncio
    async def test_successful_reload_commits_mtimes(self):
        """A successful reload records the new mtimes so it is not repeated."""
        cm = _build_config_manager()

        with patch("os.path.getmtime", return_value=2000.0), \
             patch("builtins.open", side_effect=_multi_open(ALL_YAMLS)), \
             patch("src.core.config_manager.logger"):
            assert await cm._poll_once() is True    # change detected and applied
            assert await cm._poll_once() is False   # nothing left to do

        assert set(cm.last_mtimes.values()) == {2000.0}
