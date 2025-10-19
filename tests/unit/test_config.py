"""
ABOUTME: Unit tests for ReasoningBank configuration system
ABOUTME: Tests default settings, validation logic, and preset configurations
"""

import pytest
import os
import tempfile
from reasoningbank.config import (
    ReasoningBankConfig,
    get_config_for_paper_replication,
    get_config_for_claude,
    get_config_for_matts_parallel,
    get_config_for_matts_sequential
)


class TestReasoningBankConfigDefaults:
    """Tests for default ReasoningBankConfig values"""

    def test_default_llm_settings(self):
        """Test default LLM provider and model settings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            assert config.llm_provider == "anthropic"
            assert config.llm_model == "claude-3-5-sonnet-20241022"

    def test_temperature_settings_match_paper(self):
        """Test temperature settings match paper specifications (Appendix A.2)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Paper specifies exact temperature values
            assert config.agent_temperature == 0.7, "Agent temperature should be 0.7 (balanced exploration)"
            assert config.judge_temperature == 0.0, "Judge temperature should be 0.0 (deterministic)"
            assert config.extractor_temperature == 1.0, "Extractor temperature should be 1.0 (diverse)"
            assert config.selector_temperature == 0.0, "Selector temperature should be 0.0 (consistent)"

    def test_memory_extraction_limits_match_paper(self):
        """Test extraction limits match paper specifications (Appendix A.1)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Paper uses max 3 items per trajectory
            assert config.max_memory_items_per_trajectory == 3
            # MaTTS uses max 5 aggregated items
            assert config.max_memory_items_aggregated == 5

    def test_embedding_settings_match_paper(self):
        """Test embedding settings match paper implementation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Paper uses Gemini embeddings
            assert config.embedding_model == "gemini-embedding-001"
            assert config.embedding_dimension == 768, "Gemini embeddings are 768-dimensional"

    def test_retrieval_settings_match_paper(self):
        """Test retrieval settings match paper defaults (Appendix C.1)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Paper uses top-k=1 by default
            assert config.top_k_retrieval == 1

    def test_agent_settings_match_paper(self):
        """Test agent settings match paper specifications (Section 4.1)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Paper uses 30 steps maximum
            assert config.max_steps_per_task == 30
            assert config.react_format is True
            assert config.enable_memory_injection is True

    def test_matts_default_settings(self):
        """Test MaTTS default configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            assert config.enable_matts is False, "MaTTS should be disabled by default"
            assert config.matts_mode == "parallel"
            assert config.scaling_factor_k == 3
            assert config.use_self_contrast is True
            assert config.use_best_of_n is True

    def test_refinement_prompts_match_paper(self):
        """Test sequential refinement prompts match Figure 10"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Paper provides exact refinement prompts in Figure 10
            assert len(config.refinement_prompts) == 2
            assert "re-examine the previous trajectory" in config.refinement_prompts[0]
            assert "Let's check again" in config.refinement_prompts[1]
            assert "<think>...</think><action></action>" in config.refinement_prompts[0]


class TestReasoningBankConfigValidation:
    """Tests for configuration validation logic"""

    def test_validate_top_k_retrieval_minimum(self):
        """Test top_k_retrieval must be >= 1"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="top_k_retrieval must be >= 1"):
                ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json"),
                    top_k_retrieval=0
                )

    def test_validate_max_memory_items_per_trajectory_range(self):
        """Test max_memory_items_per_trajectory must be 1-3"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test invalid value (0)
            with pytest.raises(ValueError, match="max_memory_items_per_trajectory should be 1-3"):
                ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json"),
                    max_memory_items_per_trajectory=0
                )

            # Test invalid value (4)
            with pytest.raises(ValueError, match="max_memory_items_per_trajectory should be 1-3"):
                ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json"),
                    max_memory_items_per_trajectory=4
                )

            # Test valid values (1, 2, 3)
            for valid_value in [1, 2, 3]:
                config = ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, f"memory_{valid_value}.json"),
                    max_memory_items_per_trajectory=valid_value
                )
                assert config.max_memory_items_per_trajectory == valid_value

    def test_validate_scaling_factor_k_minimum(self):
        """Test scaling_factor_k must be >= 1"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="scaling_factor_k must be >= 1"):
                ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json"),
                    scaling_factor_k=0
                )

    def test_validate_agent_temperature_range(self):
        """Test agent_temperature must be in [0, 2]"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test invalid value (negative)
            with pytest.raises(ValueError, match="agent_temperature must be in"):
                ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json"),
                    agent_temperature=-0.1
                )

            # Test invalid value (too high)
            with pytest.raises(ValueError, match="agent_temperature must be in"):
                ReasoningBankConfig(
                    llm_api_key="test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json"),
                    agent_temperature=2.1
                )

            # Test valid boundary values
            config_min = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory_min.json"),
                agent_temperature=0.0
            )
            assert config_min.agent_temperature == 0.0

            config_max = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory_max.json"),
                agent_temperature=2.0
            )
            assert config_max.agent_temperature == 2.0

    def test_validate_api_key_required(self):
        """Test API key is required for initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clear any existing API key from environment
            original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

            try:
                with pytest.raises(ValueError, match="LLM API key not found"):
                    ReasoningBankConfig(
                        llm_api_key=None,
                        memory_bank_path=os.path.join(tmpdir, "memory.json")
                    )
            finally:
                # Restore original key
                if original_key:
                    os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_validate_method_callable(self):
        """Test validate() method can be called directly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Should not raise any exceptions with valid config
            config.validate()


class TestReasoningBankConfigPresets:
    """Tests for preset configuration functions"""

    def test_paper_replication_config(self):
        """Test get_config_for_paper_replication() creates correct configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_paper_replication()
            config.llm_api_key = "test-key"
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.validate()

            # Paper uses Gemini model
            assert config.llm_provider == "google"
            assert config.llm_model == "gemini-2.5-flash"

            # Paper temperature settings
            assert config.agent_temperature == 0.7
            assert config.judge_temperature == 0.0
            assert config.extractor_temperature == 1.0

            # Paper retrieval and agent settings
            assert config.top_k_retrieval == 1
            assert config.max_steps_per_task == 30
            assert config.environment == "browsergym"

    def test_claude_config(self):
        """Test get_config_for_claude() creates correct configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_claude()
            config.llm_api_key = "test-key"
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.validate()

            # Claude-specific settings
            assert config.llm_provider == "anthropic"
            assert config.llm_model == "claude-3-5-sonnet-20241022"

            # Paper-compliant settings
            assert config.agent_temperature == 0.7
            assert config.top_k_retrieval == 1
            assert config.max_steps_per_task == 30

    def test_matts_parallel_config(self):
        """Test get_config_for_matts_parallel() creates correct configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with default k=3
            config_k3 = get_config_for_matts_parallel()
            config_k3.llm_api_key = "test-key"
            config_k3.memory_bank_path = os.path.join(tmpdir, "memory_k3.json")
            config_k3.validate()

            assert config_k3.enable_matts is True
            assert config_k3.matts_mode == "parallel"
            assert config_k3.scaling_factor_k == 3

            # Test with custom k=5
            config_k5 = get_config_for_matts_parallel(k=5)
            config_k5.llm_api_key = "test-key"
            config_k5.memory_bank_path = os.path.join(tmpdir, "memory_k5.json")
            config_k5.validate()

            assert config_k5.scaling_factor_k == 5

            # Test with k=7 (paper uses k ∈ {3,5,7})
            config_k7 = get_config_for_matts_parallel(k=7)
            config_k7.llm_api_key = "test-key"
            config_k7.memory_bank_path = os.path.join(tmpdir, "memory_k7.json")
            config_k7.validate()

            assert config_k7.scaling_factor_k == 7

    def test_matts_sequential_config(self):
        """Test get_config_for_matts_sequential() creates correct configuration"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with default k=3
            config_k3 = get_config_for_matts_sequential()
            config_k3.llm_api_key = "test-key"
            config_k3.memory_bank_path = os.path.join(tmpdir, "memory_k3.json")
            config_k3.validate()

            assert config_k3.enable_matts is True
            assert config_k3.matts_mode == "sequential"
            assert config_k3.scaling_factor_k == 3

            # Test with custom k=5
            config_k5 = get_config_for_matts_sequential(k=5)
            config_k5.llm_api_key = "test-key"
            config_k5.memory_bank_path = os.path.join(tmpdir, "memory_k5.json")
            config_k5.validate()

            assert config_k5.scaling_factor_k == 5


class TestReasoningBankConfigFileHandling:
    """Tests for file path and directory handling"""

    def test_directory_creation_in_post_init(self):
        """Test that __post_init__ creates necessary directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create path with non-existent subdirectory
            memory_path = os.path.join(tmpdir, "subdir", "nested", "memory.json")

            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=memory_path
            )

            # Directory should have been created
            assert os.path.exists(os.path.dirname(memory_path))

    def test_default_paths(self):
        """Test default file paths are set correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Check default paths
            assert config.memory_bank_path.endswith("memory.json")
            assert config.embedding_cache_path == "./data/embeddings.json"
            assert config.log_file == "./data/reasoningbank.log"
            assert config.metrics_file == "./data/metrics.json"


class TestReasoningBankConfigEnvironmentVariables:
    """Tests for environment variable handling"""

    def test_api_key_from_environment(self):
        """Test API key can be loaded from environment variable"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set environment variable
            os.environ["ANTHROPIC_API_KEY"] = "env-test-key"

            try:
                config = ReasoningBankConfig(
                    memory_bank_path=os.path.join(tmpdir, "memory.json")
                )

                # Should use key from environment
                assert config.llm_api_key == "env-test-key"
            finally:
                # Clean up
                os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_api_key_set_in_environment_after_init(self):
        """Test that API key is set in environment during __post_init__"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clear any existing key
            original_key = os.environ.pop("ANTHROPIC_API_KEY", None)

            try:
                config = ReasoningBankConfig(
                    llm_api_key="init-test-key",
                    memory_bank_path=os.path.join(tmpdir, "memory.json")
                )

                # Key should now be in environment
                assert os.environ.get("ANTHROPIC_API_KEY") == "init-test-key"
            finally:
                # Restore original key
                if original_key:
                    os.environ["ANTHROPIC_API_KEY"] = original_key
                else:
                    os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.mark.unit
class TestReasoningBankConfigIntegration:
    """Integration tests for configuration system"""

    def test_full_configuration_lifecycle(self):
        """Test complete configuration creation, validation, and usage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create configuration
            config = ReasoningBankConfig(
                llm_provider="anthropic",
                llm_model="claude-3-5-sonnet-20241022",
                llm_api_key="test-key",
                agent_temperature=0.7,
                judge_temperature=0.0,
                extractor_temperature=1.0,
                selector_temperature=0.0,
                memory_bank_path=os.path.join(tmpdir, "memory.json"),
                embedding_cache_path=os.path.join(tmpdir, "embeddings.json"),
                top_k_retrieval=1,
                max_memory_items_per_trajectory=3,
                max_memory_items_aggregated=5,
                max_steps_per_task=30,
                enable_logging=False
            )

            # Validate configuration
            config.validate()

            # Check all settings are correct
            assert config.llm_provider == "anthropic"
            assert config.agent_temperature == 0.7
            assert config.judge_temperature == 0.0
            assert config.extractor_temperature == 1.0
            assert config.selector_temperature == 0.0
            assert config.max_memory_items_per_trajectory == 3
            assert config.max_steps_per_task == 30

    def test_configuration_modification(self):
        """Test configuration can be modified after creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=os.path.join(tmpdir, "memory.json")
            )

            # Modify configuration
            config.top_k_retrieval = 3
            config.max_steps_per_task = 50
            config.enable_matts = True
            config.scaling_factor_k = 5

            # Validate modified configuration
            config.validate()

            # Check modifications were applied
            assert config.top_k_retrieval == 3
            assert config.max_steps_per_task == 50
            assert config.enable_matts is True
            assert config.scaling_factor_k == 5
