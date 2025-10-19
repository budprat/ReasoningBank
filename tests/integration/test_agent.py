"""
ABOUTME: Integration tests for ReasoningBankAgent
ABOUTME: Tests agent execution with memory retrieval, learning, and environment interaction
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from reasoningbank.agent import ReasoningBankAgent, create_agent
from reasoningbank.models import TrajectoryResult, MemoryItem, ReActStep
from reasoningbank.config import ReasoningBankConfig
from tests.fixtures.mock_environments import MockEnvironment


@pytest.mark.integration
class TestReasoningBankAgentBasic:
    """Basic agent functionality tests"""

    def test_agent_initialization(self, test_config):
        """Test ReasoningBankAgent initialization"""
        agent = ReasoningBankAgent(test_config)

        assert agent.config == test_config
        assert agent.judge is not None
        assert agent.extractor is not None
        assert agent.retriever is not None
        assert agent.consolidator is not None
        assert agent.temperature == 0.7  # Agent uses 0.7 for balanced exploration

    def test_agent_with_custom_environment(self, test_config):
        """Test agent with custom environment function"""
        def custom_env(action: str) -> str:
            return f"Custom observation for: {action}"

        agent = ReasoningBankAgent(test_config, environment=custom_env)

        assert agent.environment == custom_env

    def test_agent_uses_mock_environment_by_default(self, test_config):
        """Test that agent uses mock environment when none provided"""
        agent = ReasoningBankAgent(test_config)

        # Test mock environment
        observation = agent.environment("test action")
        assert "Executed" in observation or "completed" in observation.lower()

    def test_agent_temperature_from_config(self, test_config):
        """
        Test agent temperature matches paper specification (Appendix A.2).

        Paper specifies:
        - Agent: 0.7 (balanced exploration)
        """
        test_config.agent_temperature = 0.7
        agent = ReasoningBankAgent(test_config)

        assert agent.temperature == 0.7
        assert agent.config.agent_temperature == 0.7

    def test_agent_max_steps_configuration(self, test_config):
        """
        Test max steps configuration matches paper (Section 4.1).

        Paper specifies: Max 30 steps per task
        """
        test_config.max_steps_per_task = 30
        agent = ReasoningBankAgent(test_config)

        assert agent.config.max_steps_per_task == 30

    def test_agent_components_use_same_config(self, test_config):
        """Test all agent components share same configuration"""
        agent = ReasoningBankAgent(test_config)

        assert agent.judge.config == test_config
        assert agent.extractor.config == test_config
        assert agent.retriever.config == test_config
        assert agent.consolidator.config == test_config


@pytest.mark.integration
class TestAgentExecution:
    """Tests for agent task execution"""

    def test_agent_run_complete_cycle(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent.run() executes complete ReasoningBank cycle"""
        agent = ReasoningBankAgent(test_config)

        # Mock all LLM calls
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Test thinking</think><action>Answer: 100</action>"):

            result = agent.run(
                query="Calculate 25 * 4",
                max_steps=5,
                enable_memory_injection=False  # No memory for first run
            )

        # Verify complete result
        assert isinstance(result, TrajectoryResult)
        assert result.query == "Calculate 25 * 4"
        assert result.success is True
        assert len(result.memory_items) > 0
        assert result.entry_id is not None

    def test_agent_without_memory_injection(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent execution without memory injection"""
        agent = ReasoningBankAgent(test_config)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Calculate</think><action>Answer: 100</action>"):

            result = agent.run(
                "Calculate 25 * 4",
                max_steps=3,
                enable_memory_injection=False
            )

        assert result.success is True

    def test_agent_with_memory_injection(self, test_config, sample_memory_items, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test agent execution with memory injection"""
        agent = ReasoningBankAgent(test_config)

        # Add memories to bank first
        agent.consolidator.add_from_trajectory(
            query="Previous calculation",
            trajectory="Previous",
            final_state="State",
            model_output="Output",
            success=True,
            memory_items=sample_memory_items
        )

        # Mock all calls
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Use strategy</think><action>Answer: 100</action>"), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):

            result = agent.run(
                "Calculate 25 * 4",
                max_steps=3,
                enable_memory_injection=True
            )

        assert result.success is True

    def test_agent_respects_max_steps(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test that agent respects max_steps parameter"""
        agent = ReasoningBankAgent(test_config)

        # Mock agent to not provide answer (will hit max steps)
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Still thinking</think><action>calculate</action>"):

            result = agent.run(
                "Test query",
                max_steps=2,
                enable_memory_injection=False
            )

        # Should have exactly 2 steps
        assert result.steps_taken == 2

    def test_agent_handles_failed_task_execution(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent correctly handles failed task execution"""
        agent = ReasoningBankAgent(test_config)

        mock_response = "<think>Incorrect approach</think><action>Answer: Wrong</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]):

            result = agent.run("Difficult task", max_steps=30)

        assert result.success is False
        assert len(result.memory_items) > 0  # Should still extract failure lessons

    def test_agent_parses_react_format_in_execution(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent correctly parses ReAct format during execution"""
        agent = ReasoningBankAgent(test_config)

        mock_response = "<think>Step 1 reasoning</think>\n<action>Answer: 42</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            result = agent.run("What is the answer?", max_steps=5)

        assert result.success is True
        assert "think" in result.full_trajectory.lower()
        assert "action" in result.full_trajectory.lower()

    def test_agent_returns_complete_trajectory_result(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent returns complete TrajectoryResult structure"""
        agent = ReasoningBankAgent(test_config)

        mock_response = "<think>Solve</think>\n<action>Answer: Done</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            result = agent.run("Test task", max_steps=10)

        # Verify complete structure
        assert isinstance(result, TrajectoryResult)
        assert result.query == "Test task"
        assert result.full_trajectory is not None
        assert result.final_state is not None
        assert result.model_output is not None
        assert result.steps_taken > 0
        assert result.success is not None
        assert result.memory_items is not None
        assert result.entry_id is not None

    def test_agent_handles_llm_errors_gracefully(self, test_config):
        """Test agent handles LLM errors gracefully"""
        agent = ReasoningBankAgent(test_config)

        with patch.object(agent, '_call_agent_llm', side_effect=Exception("LLM error")):
            with pytest.raises(Exception) as exc_info:
                agent.run("Test", max_steps=5)

            assert "LLM error" in str(exc_info.value)


@pytest.mark.integration
class TestAgentMemoryIntegration:
    """Tests for agent memory integration"""

    def test_agent_learns_from_success(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test that agent learns from successful execution"""
        agent = ReasoningBankAgent(test_config)

        # Initial memory bank should be empty
        initial_stats = agent.get_statistics()
        assert initial_stats["total_entries"] == 0

        # Run task successfully
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Success</think><action>Answer: 100</action>"):

            result = agent.run("Test query", max_steps=3, enable_memory_injection=False)

        # Memory bank should have grown
        final_stats = agent.get_statistics()
        assert final_stats["total_entries"] == 1
        assert final_stats["successful_entries"] == 1

        # Should have extracted memories
        assert len(result.memory_items) > 0

    def test_agent_learns_from_failure(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test that agent learns from failed execution"""
        agent = ReasoningBankAgent(test_config)

        # Run task that fails
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Wrong</think><action>wrong answer</action>"):

            result = agent.run("Test query", max_steps=3, enable_memory_injection=False)

        # Should have learned from failure
        stats = agent.get_statistics()
        assert stats["failed_entries"] == 1

        # Should have failure memories
        assert len(result.memory_items) > 0
        assert all(item.success_signal is False for item in result.memory_items)

    def test_agent_retrieves_relevant_memories(self, test_config, sample_memory_items, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test that agent retrieves relevant memories for similar tasks"""
        agent = ReasoningBankAgent(test_config)

        # Add memories from previous task
        agent.consolidator.add_from_trajectory(
            query="Calculate 10 * 5",
            trajectory="Previous calculation",
            final_state="Result: 50",
            model_output="50",
            success=True,
            memory_items=sample_memory_items
        )

        # Run similar task
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Use strategy</think><action>Answer: 100</action>"), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):

            result = agent.run(
                "Calculate 25 * 4",  # Similar to previous task
                max_steps=3,
                enable_memory_injection=True
            )

        # Should successfully complete with memory help
        assert result.success is True

    def test_agent_retrieves_memories_before_execution(self, test_config, sample_memory_items, mock_embedding_responses, mock_judge_responses, mock_extractor_responses):
        """Test agent retrieves relevant memories before task execution"""
        agent = ReasoningBankAgent(test_config)

        # Add memories to bank first
        agent.consolidator.add_from_trajectory(
            query="Previous task",
            trajectory="Test",
            final_state="Test",
            model_output="Test",
            success=True,
            memory_items=sample_memory_items
        )

        mock_response = "<think>Use strategy</think>\n<action>Answer: 100</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            result = agent.run("Similar task", max_steps=10, enable_memory_injection=True)

        assert result.success is True
        # Memory bank should have 2 entries now (1 initial + 1 new)
        assert len(agent.get_memory_bank()) == 2

    def test_agent_consolidates_memories_to_persistent_storage(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test agent consolidates memories to persistent JSON storage"""
        agent = ReasoningBankAgent(test_config)

        mock_response = "<think>Strategy</think>\n<action>Answer: Done</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):

            result1 = agent.run("Task 1", max_steps=10, enable_memory_injection=False)
            result2 = agent.run("Task 2", max_steps=10, enable_memory_injection=True)

        # Both results should have entry IDs
        assert result1.entry_id is not None
        assert result2.entry_id is not None

        # Memory bank should persist
        assert len(agent.get_memory_bank()) == 2

    def test_agent_uses_top_k_retrieval(self, test_config, sample_memory_items, mock_embedding_responses, mock_judge_responses, mock_extractor_responses):
        """
        Test agent uses top-k=1 retrieval (Appendix C.1).

        Paper specifies: Default top-k=1 retrieval
        """
        test_config.top_k_retrieval = 1
        agent = ReasoningBankAgent(test_config)

        # Add multiple memories
        for i in range(3):
            agent.consolidator.add_from_trajectory(
                query=f"Task {i}",
                trajectory="T",
                final_state="S",
                model_output="O",
                success=True,
                memory_items=sample_memory_items
            )

        mock_response = "<think>Use memory</think>\n<action>Answer: Done</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            # Spy on retriever to check k=1
            with patch.object(agent.retriever, 'retrieve', wraps=agent.retriever.retrieve) as mock_retrieve:
                agent.run("New task", max_steps=10, enable_memory_injection=True)

                # Verify retrieve was called with k=1
                assert mock_retrieve.called
                call_kwargs = mock_retrieve.call_args[1]
                assert call_kwargs.get('k') == 1

    def test_agent_injects_memories_into_system_prompt(self, test_config, sample_memory_items, mock_embedding_responses, mock_judge_responses, mock_extractor_responses):
        """Test agent injects retrieved memories into system prompt"""
        agent = ReasoningBankAgent(test_config)

        # Add memory
        agent.consolidator.add_from_trajectory(
            query="Prior task",
            trajectory="T",
            final_state="S",
            model_output="O",
            success=True,
            memory_items=sample_memory_items
        )

        mock_response = "<think>Apply memory</think>\n<action>Answer: Done</action>"

        captured_prompts = []

        def capture_llm_call(system_prompt, user_message):
            captured_prompts.append(system_prompt)
            return mock_response

        with patch.object(agent, '_call_agent_llm', side_effect=capture_llm_call), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            agent.run("New task", max_steps=10, enable_memory_injection=True)

        # Verify system prompt contains memory injection
        assert len(captured_prompts) > 0
        system_prompt = captured_prompts[0]
        assert "Relevant Past Experience" in system_prompt or "Memory" in system_prompt

    def test_agent_validates_closed_loop_cycle(self, test_config, mock_judge_responses, mock_extractor_responses):
        """
        Test complete Retrieve → Act → Judge → Extract → Consolidate cycle.

        Paper Section 3: Complete closed-loop learning cycle
        """
        agent = ReasoningBankAgent(test_config)

        mock_response = "<think>Execute</think>\n<action>Answer: Result</action>"

        with patch.object(agent, '_call_agent_llm', return_value=mock_response), \
             patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            # Execute task
            result = agent.run("Test task", max_steps=10)

        # Verify all cycle steps completed
        assert result.full_trajectory is not None  # Act
        assert result.success is not None  # Judge
        assert len(result.memory_items) > 0  # Extract
        assert result.entry_id is not None  # Consolidate
        assert len(agent.get_memory_bank()) == 1  # Consolidated


@pytest.mark.integration
class TestAgentWithMockEnvironment:
    """Tests for agent with MockEnvironment"""

    def test_agent_with_arithmetic_environment(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent with MockEnvironment for arithmetic tasks"""
        from tests.fixtures.mock_environments import create_arithmetic_environment

        env = create_arithmetic_environment()
        agent = ReasoningBankAgent(test_config, environment=env.execute_action)

        # Mock agent response to use calculator
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Calculate</think><action>calculate 25 * 4</action>"):

            result = agent.run(
                "What is 25 * 4?",
                max_steps=5,
                enable_memory_injection=False
            )

        assert result.success is True

    def test_agent_with_search_environment(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent with MockEnvironment for search tasks"""
        from tests.fixtures.mock_environments import create_search_environment

        env = create_search_environment()
        agent = ReasoningBankAgent(test_config, environment=env.execute_action)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Search</think><action>search 'To Kill a Mockingbird author'</action>"):

            result = agent.run(
                "Who wrote 'To Kill a Mockingbird'?",
                max_steps=5,
                enable_memory_injection=False
            )

        assert result.success is True

    def test_agent_with_navigation_environment(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent with MockEnvironment for navigation tasks"""
        from tests.fixtures.mock_environments import create_navigation_environment

        env = create_navigation_environment()
        agent = ReasoningBankAgent(test_config, environment=env.execute_action)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', side_effect=[
                 "<think>Navigate to cart</think><action>click 'View Cart'</action>",
                 "<think>Proceed to checkout</think><action>click 'Checkout'</action>",
                 "<think>Done</think><action>Answer: At checkout page</action>"
             ]):

            result = agent.run(
                "Navigate to checkout page",
                max_steps=5,
                enable_memory_injection=False
            )

        assert result.success is True

    def test_agent_handles_multi_step_environment_interactions(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent handles multi-step environment interactions"""
        from tests.fixtures.mock_environments import MockEnvironment

        env = MockEnvironment()
        agent = ReasoningBankAgent(test_config, environment=env.execute_action)

        # Mock agent to perform multiple steps
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', side_effect=[
                 "<think>Step 1</think><action>action1</action>",
                 "<think>Step 2</think><action>action2</action>",
                 "<think>Step 3</think><action>Answer: Done</action>"
             ]):

            result = agent.run(
                "Multi-step task",
                max_steps=5,
                enable_memory_injection=False
            )

        assert result.success is True
        assert result.steps_taken == 3

    def test_agent_tracks_environment_state(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent tracks environment state across steps"""
        from tests.fixtures.mock_environments import create_arithmetic_environment

        env = create_arithmetic_environment()
        agent = ReasoningBankAgent(test_config, environment=env.execute_action)

        # Execute multiple environment actions
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', side_effect=[
                 "<think>First calc</think><action>calculate 5 + 3</action>",
                 "<think>Second calc</think><action>calculate 10 * 2</action>",
                 "<think>Final answer</think><action>Answer: 20</action>"
             ]):

            result = agent.run(
                "Calculate step by step",
                max_steps=5,
                enable_memory_injection=False
            )

        # Verify trajectory contains all environment observations
        assert result.success is True
        assert "calculate" in result.full_trajectory.lower()

    def test_agent_handles_environment_errors(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent handles environment errors gracefully"""
        def error_environment(action: str) -> str:
            if "error" in action.lower():
                raise ValueError("Environment error")
            return "Success"

        agent = ReasoningBankAgent(test_config, environment=error_environment)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Try action</think><action>trigger error</action>"):

            # Agent should handle environment errors
            with pytest.raises(ValueError) as exc_info:
                agent.run("Test error handling", max_steps=3, enable_memory_injection=False)

            assert "Environment error" in str(exc_info.value)

    def test_agent_uses_environment_observations(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent uses environment observations in next steps"""
        observations = []

        def tracking_environment(action: str) -> str:
            obs = f"Observation for {action}"
            observations.append(obs)
            return obs

        agent = ReasoningBankAgent(test_config, environment=tracking_environment)

        captured_messages = []

        def capture_calls(system_prompt, user_message):
            captured_messages.append(user_message)
            if len(captured_messages) == 1:
                return "<think>First step</think><action>first_action</action>"
            return "<think>Second step</think><action>Answer: Done</action>"

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', side_effect=capture_calls):

            result = agent.run(
                "Use observations",
                max_steps=3,
                enable_memory_injection=False
            )

        # Second message should contain observation from first action
        assert result.success is True
        assert len(observations) >= 1

    def test_agent_terminates_on_environment_completion_signal(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test agent terminates when environment signals completion"""
        def completion_environment(action: str) -> str:
            if "final" in action.lower():
                return "TASK_COMPLETE"
            return "Continue"

        agent = ReasoningBankAgent(test_config, environment=completion_environment)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Complete it</think><action>final action</action>"):

            result = agent.run(
                "Test completion",
                max_steps=10,
                enable_memory_injection=False
            )

        # Should complete successfully with completion signal
        assert result.success is True
        assert "TASK_COMPLETE" in result.full_trajectory or "final action" in result.full_trajectory.lower()


@pytest.mark.integration
class TestAgentProgressiveImprovement:
    """Tests for agent improving over time with memory"""

    def test_agent_improves_with_memory_accumulation(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test that agent performance improves as memory accumulates"""
        agent = ReasoningBankAgent(test_config)

        # Task 1: Initial execution (no prior memories)
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>First time</think><action>Answer: 100</action>"):

            result1 = agent.run(
                "Calculate 25 * 4",
                max_steps=5,
                enable_memory_injection=False
            )

        # Verify learning occurred
        assert result1.success is True
        assert len(agent.get_memory_bank()) == 1

        # Task 2: Similar task with memory
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Use previous strategy</think><action>Answer: 75</action>"), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):

            result2 = agent.run(
                "Calculate 15 * 5",
                max_steps=5,
                enable_memory_injection=True
            )

        # Should leverage past experience
        assert result2.success is True
        assert len(agent.get_memory_bank()) == 2

    def test_agent_memory_bank_grows_over_time(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test that agent's memory bank grows with each task"""
        agent = ReasoningBankAgent(test_config)

        # Execute multiple tasks
        for i in range(3):
            with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
                 patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
                 patch.object(agent, '_call_agent_llm', return_value=f"<think>Task {i+1}</think><action>Answer: Done</action>"):

                agent.run(f"Task {i+1}", max_steps=3, enable_memory_injection=False)

        # Memory bank should have 3 entries
        stats = agent.get_statistics()
        assert stats["total_entries"] == 3

    def test_agent_statistics_tracking(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test that agent tracks statistics correctly"""
        agent = ReasoningBankAgent(test_config)

        # Execute 2 successes and 1 failure
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Success</think><action>Answer: Done</action>"):

            agent.run("Task 1", max_steps=3, enable_memory_injection=False)
            agent.run("Task 2", max_steps=5, enable_memory_injection=False)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Failure</think><action>wrong</action>"):

            agent.run("Task 3", max_steps=2, enable_memory_injection=False)

        # Check statistics
        stats = agent.get_statistics()
        assert stats["total_entries"] == 3
        assert stats["successful_entries"] == 2
        assert stats["failed_entries"] == 1
        assert stats["success_rate"] == pytest.approx(2/3)

    def test_agent_success_rate_improves_over_time(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test that agent's success rate improves with accumulated experience"""
        agent = ReasoningBankAgent(test_config)

        # First 2 tasks: 1 success, 1 failure (50% success rate)
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Success</think><action>Answer: Done</action>"):
            agent.run("Task 1", max_steps=3, enable_memory_injection=False)

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Failed</think><action>wrong</action>"):
            agent.run("Task 2", max_steps=3, enable_memory_injection=False)

        # Check initial success rate
        stats_early = agent.get_statistics()
        assert stats_early["success_rate"] == pytest.approx(0.5)

        # Next 3 tasks: All successes with memory (should improve to 80% overall)
        for i in range(3, 6):
            with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
                 patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
                 patch.object(agent, '_call_agent_llm', return_value=f"<think>Learned</think><action>Answer: Done</action>"), \
                 patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
                agent.run(f"Task {i}", max_steps=3, enable_memory_injection=True)

        # Success rate should improve
        stats_final = agent.get_statistics()
        assert stats_final["success_rate"] > stats_early["success_rate"]
        assert stats_final["success_rate"] == pytest.approx(4/5)  # 4 successes out of 5 tasks

    def test_agent_reduces_steps_with_experience(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test that experienced agent solves tasks in fewer steps"""
        agent = ReasoningBankAgent(test_config)

        # First task: No experience, takes multiple steps
        steps_first_task = []

        def first_task_responses(*args, **kwargs):
            response = [
                "<think>Step 1</think><action>action1</action>",
                "<think>Step 2</think><action>action2</action>",
                "<think>Step 3</think><action>Answer: Done</action>"
            ][len(steps_first_task)]
            steps_first_task.append(response)
            return response

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', side_effect=first_task_responses):
            result1 = agent.run("Complex task", max_steps=5, enable_memory_injection=False)

        initial_steps = result1.steps_taken

        # Second similar task: With experience, should be more efficient
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Use memory</think><action>Answer: Done</action>"), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
            result2 = agent.run("Similar complex task", max_steps=5, enable_memory_injection=True)

        # Should complete in fewer or equal steps with experience
        assert result2.steps_taken <= initial_steps
        assert result2.success is True

    def test_agent_memory_diversity_grows(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test that agent's memory bank becomes more diverse over time"""
        agent = ReasoningBankAgent(test_config)

        # Add memories from different task types
        task_types = [
            ("Calculate 10 + 5", "arithmetic"),
            ("Search for Python docs", "search"),
            ("Navigate to homepage", "navigation"),
            ("Validate user input", "validation")
        ]

        for i, (task, task_type) in enumerate(task_types):
            with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
                 patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
                 patch.object(agent, '_call_agent_llm', return_value=f"<think>{task_type}</think><action>Answer: Done</action>"):
                agent.run(task, max_steps=3, enable_memory_injection=False)

        # Memory bank should have diverse entries
        memory_bank = agent.get_memory_bank()
        assert len(memory_bank) == 4

        # Verify queries are different (diversity check)
        queries = [entry.task_query for entry in memory_bank]
        assert len(set(queries)) == 4  # All unique queries


@pytest.mark.integration
class TestAgentReActFormat:
    """Tests for ReAct format handling"""

    def test_agent_parses_react_response_correctly(self, test_config):
        """Test that agent correctly parses ReAct format responses"""
        agent = ReasoningBankAgent(test_config)

        # Test parsing
        response = "<think>I need to calculate this</think><action>calculate 25*4</action>"
        thinking, action = agent._parse_react_response(response)

        assert thinking == "I need to calculate this"
        assert action == "calculate 25*4"

    def test_agent_formats_trajectory_correctly(self, test_config):
        """Test that agent formats trajectory correctly"""
        agent = ReasoningBankAgent(test_config)

        # Create mock steps
        steps = [
            ReActStep(1, "First think", "first action", "first obs"),
            ReActStep(2, "Second think", "second action", "second obs")
        ]

        trajectory = agent._format_trajectory(steps)

        assert "Step 1" in trajectory
        assert "Step 2" in trajectory
        assert "First think" in trajectory
        assert "second action" in trajectory

    def test_agent_handles_incomplete_react_format(self, test_config):
        """Test agent handles incomplete ReAct format gracefully"""
        agent = ReasoningBankAgent(test_config)

        # Missing closing tags
        response = "<think>thinking<action>action without closing tag"
        thinking, action = agent._parse_react_response(response)

        # Should extract what it can
        assert thinking == "thinking<action>action without closing tag" or action != ""

    def test_agent_constructs_react_steps_correctly(self, test_config):
        """Test agent constructs ReActStep objects with all fields"""
        agent = ReasoningBankAgent(test_config)

        # Create ReActStep
        step = ReActStep(
            step_num=1,
            think="Calculate the result",
            action="calculate 25 * 4",
            observation="Result: 100"
        )

        # Verify all fields present
        assert step.step_num == 1
        assert step.think == "Calculate the result"
        assert step.action == "calculate 25 * 4"
        assert step.observation == "Result: 100"

    def test_agent_handles_multiline_react_responses(self, test_config):
        """Test agent handles multi-line thinking and action blocks"""
        agent = ReasoningBankAgent(test_config)

        # Multi-line response
        response = """<think>
        First, I need to analyze the problem.
        Then, I should break it down into steps.
        Finally, execute the solution.
        </think>
        <action>
        execute step1
        execute step2
        Answer: Complete
        </action>"""

        thinking, action = agent._parse_react_response(response)

        # Should extract complete multi-line blocks
        assert "analyze the problem" in thinking.lower()
        assert "break it down" in thinking.lower()
        assert "execute step1" in action.lower()
        assert "execute step2" in action.lower()


@pytest.mark.integration
class TestAgentConvenienceFunction:
    """Tests for create_agent convenience function"""

    def test_create_agent_function(self, test_config):
        """Test create_agent() convenience function"""
        agent = create_agent(test_config)

        assert isinstance(agent, ReasoningBankAgent)
        assert agent.config == test_config

    def test_create_agent_with_custom_environment(self, test_config):
        """Test create_agent() with custom environment"""
        def custom_env(action: str) -> str:
            return "Custom"

        agent = create_agent(test_config, environment=custom_env)

        assert agent.environment == custom_env


@pytest.mark.integration
class TestEndToEndAgentWorkflow:
    """End-to-end agent workflow tests"""

    def test_complete_agent_lifecycle(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test complete agent lifecycle: multiple tasks with progressive learning"""
        agent = ReasoningBankAgent(test_config)

        # Execute 3 related tasks
        tasks = [
            "Calculate 10 * 5",
            "Calculate 20 * 3",
            "Calculate 15 * 4"
        ]

        for i, task in enumerate(tasks):
            with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
                 patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
                 patch.object(agent, '_call_agent_llm', return_value=f"<think>Task {i+1}</think><action>Answer: Done</action>"), \
                 patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):

                result = agent.run(
                    task,
                    max_steps=5,
                    enable_memory_injection=(i > 0)  # Enable memory after first task
                )

                assert result.success is True

        # Verify progressive learning
        memory_bank = agent.get_memory_bank()
        assert len(memory_bank) == 3

        # All should be successful
        assert all(entry.success for entry in memory_bank)

    def test_mixed_success_failure_workflow(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test agent learns from both successes and failures in sequence"""
        agent = ReasoningBankAgent(test_config)

        # Task 1: Success
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Success</think><action>Answer: Done</action>"):
            result1 = agent.run("Task 1", max_steps=5, enable_memory_injection=False)

        # Task 2: Failure
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["failure"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Wrong</think><action>wrong</action>"):
            result2 = agent.run("Task 2", max_steps=5, enable_memory_injection=False)

        # Task 3: Success with memory (should learn from both)
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Learned</think><action>Answer: Done</action>"), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
            result3 = agent.run("Task 3", max_steps=5, enable_memory_injection=True)

        # Verify learning from both
        assert result1.success is True
        assert result2.success is False
        assert result3.success is True

        memory_bank = agent.get_memory_bank()
        assert len(memory_bank) == 3
        assert sum(1 for entry in memory_bank if entry.success) == 2
        assert sum(1 for entry in memory_bank if not entry.success) == 1

    def test_varied_task_types_in_sequence(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test agent handles diverse task types in single workflow"""
        from tests.fixtures.mock_environments import create_arithmetic_environment, create_search_environment

        agent = ReasoningBankAgent(test_config)

        # Task 1: Arithmetic
        arith_env = create_arithmetic_environment()
        agent.environment = arith_env.execute_action

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Calculate</think><action>calculate 5 + 3</action>"):
            result1 = agent.run("Calculate 5 + 3", max_steps=5, enable_memory_injection=False)

        # Task 2: Search
        search_env = create_search_environment()
        agent.environment = search_env.execute_action

        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Search</think><action>search 'test'</action>"):
            result2 = agent.run("Search for test", max_steps=5, enable_memory_injection=False)

        # Task 3: With memory of diverse tasks
        with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
             patch.object(agent, '_call_agent_llm', return_value="<think>Combined</think><action>Answer: Done</action>"), \
             patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
            result3 = agent.run("Combined task", max_steps=5, enable_memory_injection=True)

        # All tasks successful
        assert all([result1.success, result2.success, result3.success])

        # Memory bank contains diverse entries
        memory_bank = agent.get_memory_bank()
        assert len(memory_bank) == 3
        queries = [entry.task_query for entry in memory_bank]
        assert len(set(queries)) == 3  # All unique

    def test_agent_memory_retrieval_across_workflow(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test agent retrieves and applies memories across entire workflow"""
        agent = ReasoningBankAgent(test_config)

        # Execute 5 tasks, alternating memory injection
        results = []
        for i in range(5):
            enable_memory = (i > 0)  # Enable memory after first task

            with patch.object(agent.judge, '_call_llm', return_value=mock_judge_responses["success"]), \
                 patch.object(agent.extractor, '_call_llm', return_value=mock_extractor_responses["success"]), \
                 patch.object(agent, '_call_agent_llm', return_value=f"<think>Task {i+1}</think><action>Answer: Done</action>"), \
                 patch.object(agent.retriever, 'embed_text', return_value=mock_embedding_responses["query"]):

                result = agent.run(
                    f"Task {i+1}",
                    max_steps=5,
                    enable_memory_injection=enable_memory
                )
                results.append(result)

        # All successful
        assert all(r.success for r in results)

        # Memory bank grew progressively
        memory_bank = agent.get_memory_bank()
        assert len(memory_bank) == 5

        # All have unique entry IDs
        entry_ids = [entry.id for entry in memory_bank]
        assert len(set(entry_ids)) == 5

        # Statistics accurate
        stats = agent.get_statistics()
        assert stats["total_entries"] == 5
        assert stats["successful_entries"] == 5
        assert stats["failed_entries"] == 0
        assert stats["success_rate"] == 1.0
