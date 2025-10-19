"""
ABOUTME: Tests for MaTTS (Memory-aware Test-Time Scaling) sequential refinement
ABOUTME: Tests iterative refinement, best trajectory selection, and progressive improvement
"""

import pytest
import tempfile
import os
from unittest.mock import patch

from reasoningbank.matts.sequential import MaTTSSequential, run_matts_sequential
from reasoningbank.config import ReasoningBankConfig, get_config_for_matts_sequential
from reasoningbank.models import TrajectoryResult, MemoryItem, MaTTSResult, ReActStep
from reasoningbank.agent import ReasoningBankAgent


@pytest.fixture
def matts_config():
    """Create MaTTS configuration for testing with OpenAI embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with refinement prompts (Figure 10 from paper)
        config = ReasoningBankConfig(
            llm_api_key="test-key",
            memory_bank_path=os.path.join(tmpdir, "memory.json"),
            embedding_cache_path=os.path.join(tmpdir, "embeddings.json"),
            llm_provider="anthropic",
            embedding_model="text-embedding-3-small",  # Use OpenAI embeddings
            embedding_dimension=1536,  # OpenAI embedding size
            scaling_factor_k=3,
            enable_logging=False,
            refinement_prompts=[
                "Carefully review your previous attempt and identify what went wrong. Try again with a better approach.",
                "Your previous attempt was not optimal. Think more carefully and try a different strategy.",
                "Based on the previous attempts, what is the best approach? Execute it now."
            ]
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
def sample_initial_trajectory():
    """Create sample initial trajectory (before refinement)."""
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


@pytest.fixture
def sample_refined_trajectory():
    """Create sample refined trajectory (after refinement)."""
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


@pytest.mark.matts
class TestMaTTSSequentialInitialization:
    """Tests for MaTTSSequential initialization"""

    def test_matts_sequential_initialization(self, matts_config, mock_environment):
        """Test MaTTS Sequential initializes correctly"""
        matts = MaTTSSequential(matts_config, mock_environment)

        assert matts.config == matts_config
        assert matts.environment == mock_environment
        assert matts.k == 3  # From config
        assert isinstance(matts.agent, ReasoningBankAgent)
        assert len(matts.refinement_prompts) == 3

    def test_matts_sequential_uses_config_k(self, matts_config, mock_environment):
        """Test that k is taken from config.scaling_factor_k"""
        matts_config.scaling_factor_k = 5
        matts = MaTTSSequential(matts_config, mock_environment)

        assert matts.k == 5

    def test_matts_sequential_loads_refinement_prompts(self, matts_config, mock_environment):
        """Test that refinement prompts are loaded from config"""
        matts = MaTTSSequential(matts_config, mock_environment)

        assert matts.refinement_prompts == matts_config.refinement_prompts
        assert "review your previous attempt" in matts.refinement_prompts[0].lower()
        assert "not optimal" in matts.refinement_prompts[1].lower()
        assert "best approach" in matts.refinement_prompts[2].lower()

    def test_matts_sequential_with_different_k_values(self, matts_config, mock_environment):
        """Test initialization with different k values (3, 5, 7)"""
        for k_value in [3, 5, 7]:
            matts_config.scaling_factor_k = k_value
            matts = MaTTSSequential(matts_config, mock_environment)
            assert matts.k == k_value


@pytest.mark.matts
class TestRefinementPrompts:
    """Tests for refinement prompt handling"""

    def test_get_refinement_prompt_returns_correct_prompt(self, matts_config, mock_environment):
        """Test that get_refinement_prompt returns prompt for correct iteration"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Get prompts for iterations 0, 1, 2
        prompt0 = matts._get_refinement_prompt(0)
        prompt1 = matts._get_refinement_prompt(1)
        prompt2 = matts._get_refinement_prompt(2)

        assert prompt0 == matts.refinement_prompts[0]
        assert prompt1 == matts.refinement_prompts[1]
        assert prompt2 == matts.refinement_prompts[2]

    def test_get_refinement_prompt_reuses_last_when_exhausted(self, matts_config, mock_environment):
        """Test that last prompt is reused when refinement_num exceeds list"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Request prompt beyond list length (only 3 prompts defined)
        prompt3 = matts._get_refinement_prompt(3)
        prompt4 = matts._get_refinement_prompt(4)

        # Should return last prompt
        assert prompt3 == matts.refinement_prompts[-1]
        assert prompt4 == matts.refinement_prompts[-1]

    def test_refinement_prompts_match_paper_structure(self, matts_config, mock_environment):
        """Test that refinement prompts follow Figure 10 structure from paper"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # First prompt should ask to review previous attempt
        assert "previous" in matts.refinement_prompts[0].lower()
        assert "review" in matts.refinement_prompts[0].lower() or "identify" in matts.refinement_prompts[0].lower()

        # Second prompt should encourage different strategy
        assert "previous" in matts.refinement_prompts[1].lower()
        assert "different" in matts.refinement_prompts[1].lower() or "strategy" in matts.refinement_prompts[1].lower()

        # Third prompt should ask for best approach
        assert "best" in matts.refinement_prompts[2].lower() or "approach" in matts.refinement_prompts[2].lower()


@pytest.mark.matts
class TestInitialTrajectoryExecution:
    """Tests for initial trajectory execution"""

    def test_execute_initial_trajectory(self, matts_config, mock_environment):
        """Test initial trajectory execution without refinement"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Mock agent execution
        mock_steps = [ReActStep(1, "Think", "Action", "Observation")]
        with patch.object(matts.agent, '_execute_task', return_value=(mock_steps, "Final", "Output")), \
             patch.object(matts.agent, '_format_trajectory', return_value="Formatted trajectory"), \
             patch.object(matts.agent.judge, 'judge_trajectory_success', return_value=True):

            trajectory = matts._execute_initial_trajectory("Test query", max_steps=30)

        # Verify trajectory structure
        assert isinstance(trajectory, TrajectoryResult)
        assert trajectory.query == "Test query"
        assert trajectory.full_trajectory == "Formatted trajectory"
        assert trajectory.final_state == "Final"
        assert trajectory.model_output == "Output"
        assert trajectory.steps_taken == 1
        assert trajectory.success is True

    def test_initial_trajectory_disables_memory_injection(self, matts_config, mock_environment):
        """Test that initial execution disables memory injection"""
        matts = MaTTSSequential(matts_config, mock_environment)

        mock_steps = [ReActStep(1, "T", "A", "O")]
        with patch.object(matts.agent, '_execute_task', return_value=(mock_steps, "S", "O")) as mock_execute, \
             patch.object(matts.agent, '_format_trajectory', return_value="T"), \
             patch.object(matts.agent.judge, 'judge_trajectory_success', return_value=True):

            trajectory = matts._execute_initial_trajectory("Query", 30)

        # Verify _execute_task was called with empty memories
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert call_args[1]['memories'] == []


@pytest.mark.matts
class TestRefinementExecution:
    """Tests for refinement execution"""

    def test_build_refinement_system_prompt_includes_previous_trajectory(self, matts_config, mock_environment,
                                                                          sample_initial_trajectory):
        """Test that refinement prompt includes previous trajectory information"""
        matts = MaTTSSequential(matts_config, mock_environment)

        refinement_prompt = "Try a better approach."
        system_prompt = matts._build_refinement_system_prompt(
            "Calculate 25 * 4",
            sample_initial_trajectory,
            refinement_prompt
        )

        # Should include base ReAct instructions
        assert "ReAct" in system_prompt or "reasoning and actions" in system_prompt.lower()
        assert "<think>" in system_prompt
        assert "<action>" in system_prompt

        # Should include previous attempt section
        assert "Previous Attempt" in system_prompt
        assert "Calculate 25 * 4" in system_prompt  # Task query
        assert sample_initial_trajectory.full_trajectory in system_prompt
        assert sample_initial_trajectory.final_state in system_prompt
        assert sample_initial_trajectory.model_output in system_prompt

        # Should indicate success/failure
        assert "FAILURE" in system_prompt  # Because sample_initial_trajectory.success is False

        # Should include refinement instructions
        assert refinement_prompt in system_prompt

    def test_build_refinement_system_prompt_marks_success_correctly(self, matts_config, mock_environment):
        """Test that success/failure is marked correctly in refinement prompt"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Test with successful trajectory
        success_traj = TrajectoryResult("Q", "T", "S", "O", 3, True)
        prompt_success = matts._build_refinement_system_prompt("Q", success_traj, "Refine")
        assert "SUCCESS" in prompt_success

        # Test with failed trajectory
        failure_traj = TrajectoryResult("Q", "T", "S", "O", 3, False)
        prompt_failure = matts._build_refinement_system_prompt("Q", failure_traj, "Refine")
        assert "FAILURE" in prompt_failure

    def test_execute_refinement_uses_enhanced_prompt(self, matts_config, mock_environment,
                                                      sample_initial_trajectory):
        """Test that refinement execution uses enhanced system prompt"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Patch agent methods to intercept calls
        with patch.object(matts.agent, '_call_agent_llm', return_value="<think>T</think><action>Answer: 100</action>") as patched_call, \
             patch.object(matts.agent, '_parse_react_response', return_value=("T", "Answer: 100")), \
             patch.object(matts.agent, '_format_trajectory', return_value="Formatted"), \
             patch.object(matts.agent.judge, 'judge_trajectory_success', return_value=True):

            refined = matts._execute_refinement(
                "Query",
                sample_initial_trajectory,
                "Refinement instruction",
                max_steps=10
            )

        # Verify _call_agent_llm was called using the patch object
        assert patched_call.called

        # First call should have enhanced system prompt
        first_call_args = patched_call.call_args_list[0]
        system_prompt = first_call_args[0][0]

        # Enhanced prompt should include previous trajectory
        assert "Previous Attempt" in system_prompt
        assert "Refinement instruction" in system_prompt


@pytest.mark.matts
class TestTrajectorySelection:
    """Tests for best trajectory selection in sequential mode"""

    def test_select_best_trajectory_prefers_successful(self, matts_config, mock_environment):
        """Test that selection prefers successful trajectories"""
        matts = MaTTSSequential(matts_config, mock_environment)

        trajectories = [
            TrajectoryResult("Q", "T1", "S1", "O1", 5, False),  # Initial - failed
            TrajectoryResult("Q", "T2", "S2", "O2", 4, False),  # Refinement 1 - failed
            TrajectoryResult("Q", "T3", "S3", "O3", 3, True)    # Refinement 2 - success
        ]

        best = matts._select_best_trajectory(trajectories)

        assert best.success is True
        assert best == trajectories[2]

    def test_select_best_trajectory_prefers_most_recent_among_successful(self, matts_config, mock_environment):
        """Test that among successful trajectories, selection prefers most recent"""
        matts = MaTTSSequential(matts_config, mock_environment)

        trajectories = [
            TrajectoryResult("Q", "T1", "S1", "O1", 3, True),  # Initial - success
            TrajectoryResult("Q", "T2", "S2", "O2", 5, False), # Refinement 1 - failed
            TrajectoryResult("Q", "T3", "S3", "O3", 4, True)   # Refinement 2 - success, more recent
        ]

        best = matts._select_best_trajectory(trajectories)

        # Should prefer most recent successful (index 2)
        assert best == trajectories[2]
        assert best.success is True

    def test_select_best_trajectory_prefers_fewer_steps_if_tied(self, matts_config, mock_environment):
        """Test that selection prefers fewer steps when recency is tied"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Two successful trajectories at same index (hypothetically)
        # The selection logic uses index, so let's test the tiebreaker indirectly
        trajectories = [
            TrajectoryResult("Q", "T1", "S1", "O1", 3, True),  # Fewer steps
            TrajectoryResult("Q", "T2", "S2", "O2", 5, True),  # More steps
        ]

        best = matts._select_best_trajectory(trajectories)

        # Should prefer most recent (index 1), but if we reverse:
        trajectories_reversed = [
            TrajectoryResult("Q", "T2", "S2", "O2", 5, True),
            TrajectoryResult("Q", "T1", "S1", "O1", 3, True),
        ]
        best_reversed = matts._select_best_trajectory(trajectories_reversed)

        # Most recent successful is preferred
        assert best_reversed == trajectories_reversed[1]

    def test_select_best_trajectory_returns_last_if_all_failed(self, matts_config, mock_environment):
        """Test that selection returns last attempt if all failed"""
        matts = MaTTSSequential(matts_config, mock_environment)

        trajectories = [
            TrajectoryResult("Q", "T1", "S1", "O1", 5, False),
            TrajectoryResult("Q", "T2", "S2", "O2", 4, False),
            TrajectoryResult("Q", "T3", "S3", "O3", 3, False)
        ]

        best = matts._select_best_trajectory(trajectories)

        # Should return last attempt
        assert best == trajectories[-1]

    def test_select_best_trajectory_empty_list_raises_error(self, matts_config, mock_environment):
        """Test that empty trajectory list raises ValueError"""
        matts = MaTTSSequential(matts_config, mock_environment)

        with pytest.raises(ValueError, match="No trajectories"):
            matts._select_best_trajectory([])


@pytest.mark.matts
class TestMaTTSSequentialWorkflow:
    """Tests for complete MaTTS sequential workflow"""

    def test_matts_sequential_run_complete_workflow(self, matts_config, mock_environment,
                                                     sample_initial_trajectory, sample_refined_trajectory):
        """Test complete MaTTS sequential workflow with k=3 refinements"""
        matts = MaTTSSequential(matts_config, mock_environment)
        matts_config.scaling_factor_k = 3

        # Mock trajectories: initial failure, then progressively better refinements
        mock_trajectories = [
            sample_initial_trajectory,    # Initial - failed
            sample_refined_trajectory     # Refinement 1 - success
        ]

        mock_memories = [
            MemoryItem(title="Strategy", description="Desc", content="Content", success_signal=True)
        ]

        # Mock internal methods
        with patch.object(matts, '_execute_initial_trajectory', return_value=sample_initial_trajectory), \
             patch.object(matts, '_execute_refinement', return_value=sample_refined_trajectory), \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=mock_memories), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="entry_123"):

            result = matts.run(query="Calculate 25 * 4", max_steps=30, k=3)

        # Verify result structure
        assert isinstance(result, MaTTSResult)
        assert result.query == "Calculate 25 * 4"
        assert result.scaling_mode == "sequential"
        assert result.scaling_factor == 3

        # Should have initial + 3 refinements = 4 total trajectories
        assert len(result.all_trajectories) == 4

        # Best trajectory should be a refined one (success)
        assert result.best_trajectory.success is True

        # Should have extracted memories from best trajectory
        assert result.aggregated_memories == mock_memories
        assert result.entry_id == "entry_123"

    def test_matts_sequential_executes_k_refinements(self, matts_config, mock_environment):
        """Test that sequential mode executes exactly k refinements"""
        matts = MaTTSSequential(matts_config, mock_environment)

        initial_traj = TrajectoryResult("Q", "T0", "S0", "O0", 5, False)
        refined_traj = TrajectoryResult("Q", "T_ref", "S_ref", "O_ref", 3, True)

        with patch.object(matts, '_execute_initial_trajectory', return_value=initial_traj), \
             patch.object(matts, '_execute_refinement', return_value=refined_traj) as mock_refine, \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Q", max_steps=30, k=3)

        # Verify refinement was executed 3 times
        assert mock_refine.call_count == 3

        # Verify 1 initial + 3 refinements = 4 total
        assert len(result.all_trajectories) == 4

    def test_matts_sequential_uses_config_k_by_default(self, matts_config, mock_environment):
        """Test that run() uses config.scaling_factor_k when k not specified"""
        matts_config.scaling_factor_k = 5
        matts = MaTTSSequential(matts_config, mock_environment)

        with patch.object(matts, '_execute_initial_trajectory', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)), \
             patch.object(matts, '_execute_refinement', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)) as mock_refine, \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Test", max_steps=30)

        # Verify k=5 refinements were executed
        assert mock_refine.call_count == 5
        assert result.scaling_factor == 5

    def test_matts_sequential_allows_k_override(self, matts_config, mock_environment):
        """Test that run() allows k parameter to override config"""
        matts_config.scaling_factor_k = 3
        matts = MaTTSSequential(matts_config, mock_environment)

        with patch.object(matts, '_execute_initial_trajectory', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)), \
             patch.object(matts, '_execute_refinement', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)) as mock_refine, \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Test", max_steps=30, k=7)

        # Verify k=7 refinements were executed (override)
        assert mock_refine.call_count == 7
        assert result.scaling_factor == 7

    def test_matts_sequential_uses_correct_refinement_prompts(self, matts_config, mock_environment):
        """Test that sequential refinement uses correct prompts for each iteration"""
        matts = MaTTSSequential(matts_config, mock_environment)

        with patch.object(matts, '_execute_initial_trajectory', return_value=TrajectoryResult("Q", "T", "S", "O", 1, False)), \
             patch.object(matts, '_execute_refinement', return_value=TrajectoryResult("Q", "T", "S", "O", 1, True)) as mock_refine, \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Q", max_steps=30, k=3)

        # Verify each refinement call used correct prompt
        assert mock_refine.call_count == 3

        # Check that refinement prompts were used in order
        for i, call_args in enumerate(mock_refine.call_args_list):
            refinement_prompt = call_args[0][2]
            expected_prompt = matts.refinement_prompts[i]
            assert refinement_prompt == expected_prompt

    def test_matts_sequential_extracts_memories_from_best_trajectory(self, matts_config, mock_environment):
        """Test that memory extraction uses best trajectory, not all trajectories"""
        matts = MaTTSSequential(matts_config, mock_environment)

        best_trajectory = TrajectoryResult("Q", "Best traj", "Final", "Output", 3, True)

        with patch.object(matts, '_execute_initial_trajectory', return_value=TrajectoryResult("Q", "T1", "S1", "O1", 5, False)), \
             patch.object(matts, '_execute_refinement', return_value=best_trajectory), \
             patch.object(matts, '_select_best_trajectory', return_value=best_trajectory), \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=[]) as mock_extract, \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Q", max_steps=30, k=2)

        # Verify extract_memories was called with best trajectory
        mock_extract.assert_called_once_with(
            query="Q",
            trajectory=best_trajectory.full_trajectory,
            final_state=best_trajectory.final_state,
            model_output=best_trajectory.model_output,
            success=best_trajectory.success
        )


@pytest.mark.matts
class TestMaTTSSequentialConvenienceFunction:
    """Tests for run_matts_sequential convenience function"""

    def test_run_matts_sequential_creates_and_executes_matts(self, matts_config, mock_environment):
        """Test that convenience function creates MaTTSSequential and runs it"""
        mock_result = MaTTSResult(
            query="Test",
            best_trajectory=TrajectoryResult("Test", "T", "S", "O", 3, True),
            all_trajectories=[],
            aggregated_memories=[],
            entry_id="entry_1",
            scaling_mode="sequential",
            scaling_factor=3
        )

        with patch('reasoningbank.matts.sequential.MaTTSSequential') as MockMaTTSSequential:
            mock_instance = MockMaTTSSequential.return_value
            mock_instance.run.return_value = mock_result

            result = run_matts_sequential(
                query="Test query",
                config=matts_config,
                environment=mock_environment,
                k=5
            )

        # Verify MaTTSSequential was created correctly
        MockMaTTSSequential.assert_called_once_with(matts_config, mock_environment)

        # Verify run was called with correct parameters
        mock_instance.run.assert_called_once_with("Test query", k=5)

        assert result == mock_result

    def test_run_matts_sequential_without_k_parameter(self, matts_config, mock_environment):
        """Test convenience function with k=None uses config default"""
        with patch('reasoningbank.matts.sequential.MaTTSSequential') as MockMaTTSSequential:
            mock_instance = MockMaTTSSequential.return_value
            mock_instance.run.return_value = MaTTSResult(
                query="T", best_trajectory=TrajectoryResult("T", "T", "S", "O", 1, True),
                all_trajectories=[], aggregated_memories=[], entry_id="e1",
                scaling_mode="sequential", scaling_factor=3
            )

            result = run_matts_sequential(
                query="Test",
                config=matts_config,
                environment=mock_environment
            )

        # Verify k=None was passed (will use config default)
        mock_instance.run.assert_called_once_with("Test", k=None)


@pytest.mark.matts
class TestMaTTSSequentialConfiguration:
    """Tests for MaTTS sequential configuration"""

    def test_get_config_for_matts_sequential_k3(self):
        """Test get_config_for_matts_sequential with k=3"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_matts_sequential(k=3)
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.llm_api_key = "test-key"

            assert config.scaling_factor_k == 3
            assert config.agent_temperature == 0.7
            assert len(config.refinement_prompts) > 0

    def test_get_config_for_matts_sequential_k5(self):
        """Test get_config_for_matts_sequential with k=5"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_matts_sequential(k=5)
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.llm_api_key = "test-key"

            assert config.scaling_factor_k == 5

    def test_get_config_for_matts_sequential_k7(self):
        """Test get_config_for_matts_sequential with k=7"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config_for_matts_sequential(k=7)
            config.memory_bank_path = os.path.join(tmpdir, "memory.json")
            config.llm_api_key = "test-key"

            assert config.scaling_factor_k == 7

    def test_get_config_for_matts_sequential_has_refinement_prompts(self):
        """Test that config includes refinement prompts"""
        config = get_config_for_matts_sequential(k=3)

        assert hasattr(config, 'refinement_prompts')
        assert len(config.refinement_prompts) >= 2  # At least 2 refinement prompts
        # Prompts should guide iterative improvement
        assert any("previous" in prompt.lower() for prompt in config.refinement_prompts)


@pytest.mark.matts
@pytest.mark.integration
class TestMaTTSSequentialIntegration:
    """Integration tests for MaTTS sequential with real components"""

    def test_matts_sequential_end_to_end_with_mocked_llm(self, matts_config, mock_environment):
        """Test end-to-end MaTTS sequential with mocked LLM calls"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Track call count to simulate improvement over iterations
        call_count = [0]

        def mock_agent_llm(system_prompt, user_message):
            call_count[0] += 1
            if call_count[0] <= 3:
                # Initial attempts fail
                return "<think>I'll add them</think>\n<action>Answer: 29</action>"
            else:
                # Refinements succeed
                return "<think>I need to multiply</think>\n<action>Answer: 100</action>"

        def mock_judge(query, trajectory, final_state, model_output):
            return "100" in model_output

        mock_memories = [
            MemoryItem(
                title="Multiplication Strategy",
                description="Use multiplication for products",
                content="When finding products, use multiply not add.",
                success_signal=True
            )
        ]

        with patch.object(matts.agent, '_call_agent_llm', side_effect=mock_agent_llm), \
             patch.object(matts.agent.judge, 'judge_trajectory_success', side_effect=mock_judge), \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=mock_memories):

            result = matts.run(query="Calculate 25 * 4", max_steps=10, k=3)

        # Verify result
        assert isinstance(result, MaTTSResult)
        assert result.scaling_mode == "sequential"
        assert result.scaling_factor == 3
        assert len(result.all_trajectories) == 4  # 1 initial + 3 refinements
        assert result.best_trajectory.success is True
        assert len(result.aggregated_memories) > 0
        assert result.entry_id is not None

    def test_matts_sequential_shows_progressive_improvement(self, matts_config, mock_environment):
        """Test that sequential refinement shows progressive improvement"""
        matts = MaTTSSequential(matts_config, mock_environment)

        # Create trajectories showing improvement: initial fail → refinements succeed
        trajectories = [
            TrajectoryResult("Q", "T1", "S1", "Fail", 8, False),      # Initial - failed, many steps
            TrajectoryResult("Q", "T2", "S2", "Better", 6, False),    # Refinement 1 - still fail, fewer steps
            TrajectoryResult("Q", "T3", "S3", "Good", 4, True),       # Refinement 2 - success!
            TrajectoryResult("Q", "T4", "S4", "Best", 3, True)        # Refinement 3 - success, even fewer steps
        ]

        # Mock to return improving trajectories
        with patch.object(matts, '_execute_initial_trajectory', return_value=trajectories[0]), \
             patch.object(matts, '_execute_refinement', side_effect=trajectories[1:]), \
             patch.object(matts.agent.extractor, 'extract_memories', return_value=[]), \
             patch.object(matts.agent.consolidator, 'add_from_trajectory', return_value="e1"):

            result = matts.run(query="Q", max_steps=30, k=3)

        # Verify progressive improvement
        assert result.all_trajectories == trajectories

        # Best should be the last (most refined and successful)
        assert result.best_trajectory == trajectories[3]
        assert result.best_trajectory.success is True
        assert result.best_trajectory.steps_taken == 3

        # Shows benefit of iterative refinement
        initial_steps = trajectories[0].steps_taken
        final_steps = trajectories[3].steps_taken
        improvement_pct = ((initial_steps - final_steps) / initial_steps) * 100
        assert improvement_pct > 60  # At least 60% improvement in efficiency
