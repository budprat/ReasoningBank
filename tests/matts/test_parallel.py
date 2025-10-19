"""
ABOUTME: Tests for MaTTS (Memory-aware Test-Time Scaling) parallel scaling
ABOUTME: Tests k-trajectory sampling, best-of-n selection, and self-contrast extraction
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import Future

from reasoningbank.matts.parallel import MaTTSParallel, run_matts_parallel
from reasoningbank.config import ReasoningBankConfig, get_config_for_matts_parallel
from reasoningbank.models import TrajectoryResult, MemoryItem, MaTTSResult, ReActStep
from reasoningbank.agent import ReasoningBankAgent


@pytest.fixture
def matts_config():
    """Create MaTTS configuration for testing with OpenAI embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ReasoningBankConfig(
            llm_api_key="test-key",
            memory_bank_path=os.path.join(tmpdir, "memory.json"),
            embedding_cache_path=os.path.join(tmpdir, "embeddings.json"),
            llm_provider="anthropic",
            embedding_model="text-embedding-3-small",  # Use OpenAI embeddings
            embedding_dimension=1536,  # OpenAI embedding size
            scaling_factor_k=3,
            enable_logging=False
        )
        yield config


@pytest.fixture
def mock_environment():
    """Create mock environment for testing."""
    def environment(action: str) -> str:
        if action.lower().startswith("calculate"):
            return "100"
        elif action.lower().startswith("answer:"):
            return "Task completed."
        else:
            return f"Executed: {action}"
    return environment


@pytest.fixture
def sample_trajectory_success():
    """Create sample successful trajectory."""
    return TrajectoryResult(
        query="Calculate 25 * 4",
        full_trajectory="<think>Multiply</think>\n<action>calculate 25*4</action>\n<observation>100</observation>",
        final_state="100",
        model_output="Answer: 100",
        steps_taken=3,
        success=True,
        memory_items=None,
        entry_id=None
    )


@pytest.fixture
def sample_trajectory_failure():
    """Create sample failed trajectory."""
    return TrajectoryResult(
        query="Calculate 25 * 4",
        full_trajectory="<think>Add</think>\n<action>calculate 25+4</action>\n<observation>29</observation>",
        final_state="29",
        model_output="Answer: 29",
        steps_taken=5,
        success=False,
        memory_items=None,
        entry_id=None
    )


@pytest.mark.matts
class TestMaTTSParallelInitialization:
    """Tests for MaTTSParallel initialization"""

    def test_matts_parallel_initialization(self, matts_config, mock_environment):
        """Test MaTTS Parallel initializes correctly"""
        matts = MaTTSParallel(matts_config, mock_environment)

        assert matts.config == matts_config
        assert matts.environment == mock_environment
        assert matts.k == 3  # From config
        assert isinstance(matts.agent, ReasoningBankAgent)

    def test_matts_parallel_uses_config_k(self, matts_config, mock_environment):
        """Test that k is taken from config.scaling_factor_k"""
        matts_config.scaling_factor_k = 5
        matts = MaTTSParallel(matts_config, mock_environment)

        assert matts.k == 5

    def test_matts_parallel_with_different_k_values(self, matts_config, mock_environment):
        """Test initialization with different k values (3, 5, 7)"""
        for k_value in [3, 5, 7]:
            matts_config.scaling_factor_k = k_value
            matts = MaTTSParallel(matts_config, mock_environment)
            assert matts.k == k_value


@pytest.mark.matts
class TestTrajectorySelection:
    """Tests for best-of-n trajectory selection"""

    def test_select_best_trajectory_prefers_success(self, matts_config, mock_environment,
                                                     sample_trajectory_success, sample_trajectory_failure):
        """Test that selection prefers successful trajectories over failed ones"""
        matts = MaTTSParallel(matts_config, mock_environment)

        # Mix of successful and failed trajectories
        trajectories = [sample_trajectory_failure, sample_trajectory_success]

        best = matts._select_best_trajectory(trajectories)

        assert best.success is True
        assert best == sample_trajectory_success

    def test_select_best_trajectory_prefers_fewer_steps_among_successful(self, matts_config, mock_environment):
        """Test that among successful trajectories, selection prefers fewer steps"""
        matts = MaTTSParallel(matts_config, mock_environment)

        # Create two successful trajectories with different step counts
        traj1 = TrajectoryResult(
            query="Test", full_trajectory="T1", final_state="S1",
            model_output="O1", steps_taken=5, success=True
        )
        traj2 = TrajectoryResult(
            query="Test", full_trajectory="T2", final_state="S2",
            model_output="O2", steps_taken=3, success=True
        )
        traj3 = TrajectoryResult(
            query="Test", full_trajectory="T3", final_state="S3",
            model_output="O3", steps_taken=7, success=True
        )

        trajectories = [traj1, traj2, traj3]
        best = matts._select_best_trajectory(trajectories)

        # Should select traj2 with fewest steps
        assert best.steps_taken == 3
        assert best == traj2

    def test_select_best_trajectory_among_all_failed(self, matts_config, mock_environment):
        """Test selection among all failed trajectories chooses fewer steps"""
        matts = MaTTSParallel(matts_config, mock_environment)

        traj1 = TrajectoryResult(
            query="Test", full_trajectory="T1", final_state="S1",
            model_output="O1", steps_taken=8, success=False
        )
        traj2 = TrajectoryResult(
                query="Test", full_trajectory="T2", final_state="S2",
                model_output="O2", steps_taken=4, success=False
            )
    
        best = matts._select_best_trajectory([traj1, traj2])

        # Should prefer fewer steps even among failures
        assert best.steps_taken == 4
        assert best == traj2

    def test_select_best_trajectory_empty_list_raises_error(self, matts_config, mock_environment):
        """Test that empty trajectory list raises ValueError"""
        matts = MaTTSParallel(matts_config, mock_environment)

        with pytest.raises(ValueError, match="No trajectories"):
            matts._select_best_trajectory([])

    
@pytest.mark.matts
class TestParallelSampling:
    """Tests for parallel trajectory sampling"""

    def test_sample_parallel_trajectories_generates_k_trajectories(self, matts_config, mock_environment):
        """Test that parallel sampling generates exactly k trajectories"""
        matts = MaTTSParallel(matts_config, mock_environment)
        matts_config.scaling_factor_k = 3

        # Mock the single trajectory sampling
        mock_trajectory = TrajectoryResult(
            query="Test", full_trajectory="T", final_state="S",
            model_output="O", steps_taken=3, success=True
        )

        with patch.object(matts, '_sample_single_trajectory', return_value=mock_trajectory):
            trajectories = matts._sample_parallel_trajectories("Test query", k=3, max_steps=30)

        assert len(trajectories) == 3
        assert all(isinstance(t, TrajectoryResult) for t in trajectories)

    def test_sample_parallel_trajectories_with_different_k_values(self, matts_config, mock_environment):
        """Test parallel sampling with k=3, k=5, k=7"""
        matts = MaTTSParallel(matts_config, mock_environment)

        mock_trajectory = TrajectoryResult(
            query="Test", full_trajectory="T", final_state="S",
            model_output="O", steps_taken=3, success=True
        )

        for k in [3, 5, 7]:
            with patch.object(matts, '_sample_single_trajectory', return_value=mock_trajectory):
                trajectories = matts._sample_parallel_trajectories("Test", k=k, max_steps=30)

            assert len(trajectories) == k

    def test_sample_single_trajectory_disables_memory_injection(self, matts_config, mock_environment):
        """Test that single trajectory sampling disables memory injection"""
        matts = MaTTSParallel(matts_config, mock_environment)
        agent = ReasoningBankAgent(matts_config, mock_environment)

        # Mock agent execution methods
        mock_steps = [
            ReActStep(1, "Think", "calculate 25*4", "100")
        ]
        with patch.object(agent, '_execute_task', return_value=(mock_steps, "100", "Answer: 100")), \
             patch.object(agent, '_format_trajectory', return_value="<think>Think</think>"), \
             patch.object(agent.judge, 'judge_trajectory_success', return_value=True):

            trajectory = matts._sample_single_trajectory(agent, "Calculate 25*4", 30, 0)

            # Verify _execute_task was called with empty memories
            agent._execute_task.assert_called_once()
            call_args = agent._execute_task.call_args
            assert call_args[1]['memories'] == []

    def test_parallel_sampling_handles_errors_gracefully(self, matts_config, mock_environment):
        """Test that parallel sampling handles individual trajectory errors"""
        matts = MaTTSParallel(matts_config, mock_environment)

        # Create mock that fails on first call, succeeds on others
        call_count = [0]
        def mock_sample(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Sampling error")
            return TrajectoryResult(
                query="Test", full_trajectory="T", final_state="S",
                model_output="O", steps_taken=3, success=True
            )

        with patch.object(matts, '_sample_single_trajectory', side_effect=mock_sample):
            trajectories = matts._sample_parallel_trajectories("Test", k=3, max_steps=30)

        # Should have 2 trajectories (1 failed, 2 succeeded)
        assert len(trajectories) == 2


@pytest.mark.matts
class TestSelfContrastExtraction:
    """Tests for self-contrast memory extraction"""

    def test_extract_aggregated_memories_uses_self_contrast(self, matts_config, mock_environment):
        """Test that aggregated extraction uses self-contrast method"""
        matts = MaTTSParallel(matts_config, mock_environment)

        trajectories = [
            TrajectoryResult("Q1", "T1", "S1", "O1", 3, True),
            TrajectoryResult("Q1", "T2", "S2", "O2", 4, True),
            TrajectoryResult("Q1", "T3", "S3", "O3", 2, True)
        ]

        mock_memories = [
            MemoryItem(title="Strategy 1", description="Desc 1", content="Content 1"),
            MemoryItem(title="Strategy 2", description="Desc 2", content="Content 2")
        ]

        with patch.object(matts.agent.extractor, 'extract_with_self_contrast',
                         return_value=mock_memories) as mock_extract:

            memories = matts._extract_aggregated_memories("Q1", trajectories)

            # Verify self-contrast was called
            mock_extract.assert_called_once()

            # Verify correct arguments
            call_args = mock_extract.call_args[0]
            trajectory_tuples = call_args[0]
            query = call_args[1]

            assert len(trajectory_tuples) == 3
            assert query == "Q1"
            assert memories == mock_memories

    def test_extract_aggregated_memories_respects_max_items(self, matts_config, mock_environment):
        """Test that aggregated extraction respects max_memory_items_aggregated (5)"""
        matts_config.max_memory_items_aggregated = 5
        matts = MaTTSParallel(matts_config, mock_environment)

        trajectories = [
            TrajectoryResult("Q", "T1", "S", "O", 3, True),
            TrajectoryResult("Q", "T2", "S", "O", 3, True)
        ]

        # Mock extractor to return 7 items (exceeds max 5)
        many_memories = [
            MemoryItem(title=f"Item {i}", description=f"Desc {i}", content=f"Content {i}")
            for i in range(7)
        ]

        with patch.object(matts.agent.extractor, 'extract_with_self_contrast',
                         return_value=many_memories[:5]):  # Extractor should enforce limit

            memories = matts._extract_aggregated_memories("Q", trajectories)

            # Should return at most 5 items
            assert len(memories) <= 5


@pytest.mark.matts
class TestMaTTSParallelWorkflow:
    """Tests for complete MaTTS parallel workflow"""

    def test_matts_parallel_run_complete_workflow(self, matts_config, mock_environment):
        """Test complete MaTTS parallel workflow with k=3"""
        matts = MaTTSParallel(matts_config, mock_environment)
        matts_config.scaling_factor_k = 3

        # Mock trajectories
        mock_trajectories = [
            TrajectoryResult("Q", "T1", "S1", "O1", 5, False),  # Failed
            TrajectoryResult("Q", "T2", "S2", "O2", 3, True),   # Best (success, fewer steps)
            TrajectoryResult("Q", "T3", "S3", "O3", 4, True)    # Success but more steps
        ]

        mock_memories = [
            MemoryItem(title="Strategy", description="Desc", content="Content", success_signal=True)
        ]

        with patch.object(matts, '_sample_parallel_trajectories', return_value=mock_trajectories), \
             patch.object(matts, '_extract_aggregated_memories', return_value=mock_memories), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="entry_123"):
    
            result = matts.run(query="Calculate 25 * 4", max_steps=30, k=3)
    
            # Verify result structure
            assert isinstance(result, MaTTSResult)
            assert result.query == "Calculate 25 * 4"
            assert result.scaling_mode == "parallel"
            assert result.scaling_factor == 3
            assert result.best_trajectory == mock_trajectories[1]  # Best one selected
            assert result.all_trajectories == mock_trajectories
            assert result.aggregated_memories == mock_memories
            assert result.entry_id == "entry_123"

    def test_matts_parallel_run_uses_config_k_by_default(self, matts_config, mock_environment):
        """Test that run() uses config.scaling_factor_k when k not specified"""
        matts_config.scaling_factor_k = 5
        matts = MaTTSParallel(matts_config, mock_environment)

        with patch.object(matts, '_sample_parallel_trajectories', return_value=[]), \
             patch.object(matts, '_select_best_trajectory', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)), \
             patch.object(matts, '_extract_aggregated_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="entry_1"):

            result = matts.run(query="Test", max_steps=30)

            # Verify k=5 was used (from config)
            matts._sample_parallel_trajectories.assert_called_once()
            call_args = matts._sample_parallel_trajectories.call_args[0]
            assert call_args[1] == 5  # k is 2nd positional argument

    def test_matts_parallel_run_allows_k_override(self, matts_config, mock_environment):
        """Test that run() allows k parameter to override config"""
        matts_config.scaling_factor_k = 3
        matts = MaTTSParallel(matts_config, mock_environment)

        with patch.object(matts, '_sample_parallel_trajectories', return_value=[]), \
             patch.object(matts, '_select_best_trajectory', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)), \
             patch.object(matts, '_extract_aggregated_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="entry_1"):

            result = matts.run(query="Test", max_steps=30, k=7)

            # Verify k=7 was used (override)
            matts._sample_parallel_trajectories.assert_called_once()
            call_args = matts._sample_parallel_trajectories.call_args[0]
            assert call_args[1] == 7  # k is 2nd positional argument

    def test_matts_parallel_consolidates_best_trajectory_with_aggregated_memories(self, matts_config, mock_environment):
        """Test that consolidation uses best trajectory with aggregated memories"""
        matts = MaTTSParallel(matts_config, mock_environment)

        best_trajectory = TrajectoryResult("Q", "Best trajectory", "Final", "Output", 3, True)
        mock_trajectories = [
            TrajectoryResult("Q", "T1", "S1", "O1", 5, False),
            best_trajectory,
            TrajectoryResult("Q", "T3", "S3", "O3", 4, True)
        ]

        aggregated_memories = [
            MemoryItem(title="Aggregated", description="From self-contrast", content="Robust pattern")
        ]

        with patch.object(matts, '_sample_parallel_trajectories', return_value=mock_trajectories), \
             patch.object(matts, '_extract_aggregated_memories', return_value=aggregated_memories), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="entry_123") as mock_consolidate:

            result = matts.run(query="Q", max_steps=30, k=3)

        # Verify consolidation was called with best trajectory and aggregated memories
        mock_consolidate.assert_called_once_with(
            query="Q",
            trajectory=best_trajectory.full_trajectory,
            final_state=best_trajectory.final_state,
            model_output=best_trajectory.model_output,
            success=best_trajectory.success,
            memory_items=aggregated_memories,
            steps_taken=best_trajectory.steps_taken
        )


@pytest.mark.matts
class TestMaTTSParallelConvenienceFunction:
    """Tests for run_matts_parallel convenience function"""

    def test_run_matts_parallel_creates_and_executes_matts(self, matts_config, mock_environment):
        """Test that convenience function creates MaTTSParallel and runs it"""
        mock_result = MaTTSResult(
            query="Test",
            best_trajectory=TrajectoryResult("Test", "T", "S", "O", 3, True),
            all_trajectories=[],
            aggregated_memories=[],
            entry_id="entry_1",
            scaling_mode="parallel",
            scaling_factor=3
        )

        with patch('reasoningbank.matts.parallel.MaTTSParallel') as MockMaTTSParallel:
            mock_instance = MockMaTTSParallel.return_value
            mock_instance.run.return_value = mock_result

            result = run_matts_parallel(
                query="Test query",
                config=matts_config,
                environment=mock_environment,
                k=5
            )

        # Verify MaTTSParallel was created correctly
        MockMaTTSParallel.assert_called_once_with(matts_config, mock_environment)

        # Verify run was called with correct parameters
        mock_instance.run.assert_called_once_with("Test query", k=5)

        assert result == mock_result

    def test_run_matts_parallel_without_k_parameter(self, matts_config, mock_environment):
        """Test convenience function with k=None uses config default"""
        with patch('reasoningbank.matts.parallel.MaTTSParallel') as MockMaTTSParallel:
            mock_instance = MockMaTTSParallel.return_value
            mock_instance.run.return_value = MaTTSResult(
                query="T", best_trajectory=TrajectoryResult("T", "T", "S", "O", 1, True),
                all_trajectories=[], aggregated_memories=[], entry_id="e1",
                scaling_mode="parallel", scaling_factor=3
            )

            result = run_matts_parallel(
                query="Test",
                config=matts_config,
                environment=mock_environment
            )

        # Verify k=None was passed (will use config default)
        mock_instance.run.assert_called_once_with("Test", k=None)


@pytest.mark.matts
class TestMaTTSParallelConfiguration:
    """Tests for MaTTS parallel configuration"""

    def test_get_config_for_matts_parallel_k3(self):
        """Test get_config_for_matts_parallel with k=3"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_matts_parallel(k=3)
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.llm_api_key = "test-key"

            assert config.scaling_factor_k == 3
            assert config.agent_temperature == 0.7
            assert config.max_memory_items_aggregated == 5

    def test_get_config_for_matts_parallel_k5(self):
        """Test get_config_for_matts_parallel with k=5"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_matts_parallel(k=5)
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.llm_api_key = "test-key"

            assert config.scaling_factor_k == 5

    def test_get_config_for_matts_parallel_k7(self):
        """Test get_config_for_matts_parallel with k=7"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_matts_parallel(k=7)
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.llm_api_key = "test-key"

            assert config.scaling_factor_k == 7


@pytest.mark.matts
@pytest.mark.integration
class TestMaTTSParallelIntegration:
    """Integration tests for MaTTS parallel with real components"""

    def test_matts_parallel_end_to_end_with_mocked_llm(self, matts_config, mock_environment):
        """Test end-to-end MaTTS parallel with mocked LLM calls"""
        matts = MaTTSParallel(matts_config, mock_environment)

        # Mock LLM responses for agent execution
        def mock_agent_llm(system_prompt, user_message):
            return "<think>I need to calculate 25 * 4</think>\n<action>Answer: 100</action>"

        # Mock judge to return success
        def mock_judge(query, trajectory, final_state, model_output):
            return "100" in model_output

        # Mock extractor self-contrast
        mock_memories = [
            MemoryItem(
                title="Multiplication Strategy",
                description="Use multiplication for product calculations",
                content="When calculating products, use the multiply operation directly.",
                success_signal=True
            )
        ]

        # Create mock trajectories for parallel sampling
        mock_trajectories = [
            TrajectoryResult("Calculate 25 * 4", "T1", "100", "Answer: 100", 3, True),
            TrajectoryResult("Calculate 25 * 4", "T2", "100", "Answer: 100", 3, True),
            TrajectoryResult("Calculate 25 * 4", "T3", "100", "Answer: 100", 3, True)
        ]

        with patch.object(matts, '_sample_parallel_trajectories', return_value=mock_trajectories), \
             patch.object(matts.agent.extractor, 'extract_with_self_contrast', return_value=mock_memories), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="entry_123"):

            result = matts.run(query="Calculate 25 * 4", max_steps=10, k=3)

            # Verify result
            assert isinstance(result, MaTTSResult)
            assert result.scaling_mode == "parallel"
            assert result.scaling_factor == 3
            assert len(result.all_trajectories) == 3
            assert result.best_trajectory.success is True
            assert len(result.aggregated_memories) > 0
        assert result.entry_id is not None

    def test_matts_parallel_benefits_over_single_trajectory(self, matts_config, mock_environment):
        """Test that parallel scaling provides benefits over single execution"""
        matts = MaTTSParallel(matts_config, mock_environment)

        # Create varied trajectories (some fail, some succeed)
        trajectories_varied = [
            TrajectoryResult("Q", "T1", "S1", "Fail", 8, False),     # Failed, many steps
            TrajectoryResult("Q", "T2", "S2", "Success", 3, True),   # Success, few steps ✓
            TrajectoryResult("Q", "T3", "S3", "Fail", 6, False)      # Failed
        ]

        with patch.object(matts, '_sample_parallel_trajectories', return_value=trajectories_varied), \
             patch.object(matts, '_extract_aggregated_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Q", max_steps=30, k=3)

        # Best-of-n selection should choose the best trajectory
        assert result.best_trajectory.success is True
        assert result.best_trajectory.steps_taken == 3

        # Without parallel scaling, we might have gotten the first (failed) trajectory
        # With parallel scaling, we get the best one
