"""
ABOUTME: Unit tests for TrajectoryJudge (LLM-as-a-Judge) component
ABOUTME: Tests trajectory success/failure classification with temperature=0.0
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from reasoningbank.judge import TrajectoryJudge, judge_trajectory_success, GOOGLE_AVAILABLE
from reasoningbank.config import ReasoningBankConfig


class TestTrajectoryJudgeInitialization:
    """Tests for TrajectoryJudge initialization"""

    def test_judge_initialization_anthropic(self, test_config):
        """Test TrajectoryJudge initialization with Anthropic"""
        test_config.llm_provider = "anthropic"
        test_config.llm_model = "claude-3-5-sonnet-20241022"

        judge = TrajectoryJudge(test_config)

        assert judge.config == test_config
        assert judge.llm_provider == "anthropic"
        assert judge.llm_model == "claude-3-5-sonnet-20241022"
        assert judge.temperature == 0.0, "Judge temperature must be 0.0 for deterministic judgments"
        assert judge.client is not None

    def test_judge_initialization_openai(self, test_config):
        """Test TrajectoryJudge initialization with OpenAI"""
        test_config.llm_provider = "openai"
        test_config.llm_model = "gpt-4"

        judge = TrajectoryJudge(test_config)

        assert judge.llm_provider == "openai"
        assert judge.llm_model == "gpt-4"
        assert judge.temperature == 0.0

    @pytest.mark.skipif(not GOOGLE_AVAILABLE, reason="google-generativeai not installed")
    def test_judge_initialization_google(self, test_config):
        """Test TrajectoryJudge initialization with Google"""
        test_config.llm_provider = "google"
        test_config.llm_model = "gemini-2.5-flash"

        judge = TrajectoryJudge(test_config)

        assert judge.llm_provider == "google"
        assert judge.llm_model == "gemini-2.5-flash"
        assert judge.temperature == 0.0

    def test_judge_temperature_from_config(self, test_config):
        """Test that judge uses judge_temperature from config"""
        # Paper specifies judge_temperature=0.0 (Appendix A.2)
        test_config.judge_temperature = 0.0

        judge = TrajectoryJudge(test_config)

        assert judge.temperature == 0.0

    def test_unsupported_provider_raises_error(self, test_config):
        """Test that unsupported provider raises ValueError"""
        test_config.llm_provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            TrajectoryJudge(test_config)


class TestJudgePromptBuilding:
    """Tests for judge prompt construction (Figure 9)"""

    def test_build_judge_prompt_structure(self, test_config):
        """Test judge prompt matches Figure 9 structure"""
        judge = TrajectoryJudge(test_config)

        prompt = judge._build_judge_prompt(
            query="What is 25 * 4?",
            trajectory="<think>Calculate</think><action>calculate 25*4</action><observation>100</observation>",
            final_state="Result: 100",
            model_output="100"
        )

        # Check prompt contains required elements from Figure 9
        assert "expert in evaluating the performance" in prompt
        assert "web navigation agent" in prompt
        assert "three types of tasks" in prompt.lower()
        assert "Information seeking" in prompt
        assert "Site navigation" in prompt
        assert "Content modification" in prompt

    def test_build_judge_prompt_format_requirements(self, test_config):
        """Test prompt includes format requirements (Thoughts + Status)"""
        judge = TrajectoryJudge(test_config)

        prompt = judge._build_judge_prompt(
            query="Test query",
            trajectory="Test trajectory",
            final_state="Test state",
            model_output="Test output"
        )

        # Paper requires specific output format
        assert "Thoughts:" in prompt
        assert "Status:" in prompt
        assert '"success"' in prompt or '"failure"' in prompt

    def test_build_judge_prompt_includes_all_inputs(self, test_config):
        """Test prompt includes query, trajectory, final_state, and model_output"""
        judge = TrajectoryJudge(test_config)

        query = "Calculate 25 * 4 + 15"
        trajectory = "<think>Step 1</think><action>calculate</action>"
        final_state = "Result: 115"
        model_output = "The answer is 115"

        prompt = judge._build_judge_prompt(query, trajectory, final_state, model_output)

        # All inputs should be present in prompt
        assert query in prompt
        assert trajectory in prompt
        assert final_state in prompt
        assert model_output in prompt

    def test_build_judge_prompt_handles_empty_output(self, test_config):
        """Test prompt handles None or empty model_output"""
        judge = TrajectoryJudge(test_config)

        # Test with None output
        prompt = judge._build_judge_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output=None
        )

        assert "N/A" in prompt

        # Test with empty string
        prompt = judge._build_judge_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output=""
        )

        assert "N/A" in prompt


class TestJudgmentParsing:
    """Tests for parsing judge responses"""

    def test_parse_judgment_success(self, test_config):
        """Test parsing success responses"""
        judge = TrajectoryJudge(test_config)

        # Test various success formats
        success_responses = [
            "Thoughts: Task completed correctly\nStatus: success",
            "Status: success",
            "SUCCESS",
            "The task was a success",
            "  success  ",  # With whitespace
        ]

        for response in success_responses:
            result = judge._parse_judgment(response)
            assert result is True, f"Failed to parse success from: {response}"

    def test_parse_judgment_failure(self, test_config):
        """Test parsing failure responses"""
        judge = TrajectoryJudge(test_config)

        # Test various failure formats
        failure_responses = [
            "Thoughts: Task failed\nStatus: failure",
            "Status: failure",
            "FAILURE",
            "The task was a failure",
            "Task failed",
            "  failure  ",  # With whitespace
        ]

        for response in failure_responses:
            result = judge._parse_judgment(response)
            assert result is False, f"Failed to parse failure from: {response}"

    def test_parse_judgment_ambiguous_raises_error(self, test_config):
        """Test that ambiguous responses raise ValueError"""
        judge = TrajectoryJudge(test_config)

        ambiguous_responses = [
            "Maybe?",
            "Uncertain",
            "Not sure",
            "Unknown",
            "",  # Empty response
        ]

        for response in ambiguous_responses:
            with pytest.raises(ValueError, match="Cannot parse judge response"):
                judge._parse_judgment(response)

    def test_parse_judgment_case_insensitive(self, test_config):
        """Test parsing is case-insensitive"""
        judge = TrajectoryJudge(test_config)

        # Test mixed case
        assert judge._parse_judgment("SUCCESS") is True
        assert judge._parse_judgment("Success") is True
        assert judge._parse_judgment("success") is True

        assert judge._parse_judgment("FAILURE") is False
        assert judge._parse_judgment("Failure") is False
        assert judge._parse_judgment("failure") is False


class TestJudgeTrajectorySuccess:
    """Tests for main judge_trajectory_success method"""

    def test_judge_successful_trajectory_with_mock(self, test_config, mock_judge_responses):
        """Test judging a successful trajectory with mocked LLM"""
        judge = TrajectoryJudge(test_config)

        # Mock LLM call to return success
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["success"]):
            success = judge.judge_trajectory_success(
                query="What is 25 * 4?",
                trajectory="<think>Calculate</think><action>calculate 25*4</action>",
                final_state="Result: 100",
                model_output="100"
            )

        assert success is True

    def test_judge_failed_trajectory_with_mock(self, test_config, mock_judge_responses):
        """Test judging a failed trajectory with mocked LLM"""
        judge = TrajectoryJudge(test_config)

        # Mock LLM call to return failure
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["failure"]):
            success = judge.judge_trajectory_success(
                query="What is 25 * 4?",
                trajectory="<think>Wrong</think><action>calculate 25+4</action>",
                final_state="Result: 29",
                model_output="29"
            )

        assert success is False

    def test_judge_uses_correct_temperature(self, test_config):
        """Test that judge calls LLM with temperature=0.0"""
        judge = TrajectoryJudge(test_config)

        # Mock _call_llm to capture the call
        with patch.object(judge, '_call_llm', return_value="Status: success") as mock_call:
            judge.judge_trajectory_success(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test"
            )

            # Verify _call_llm was called
            assert mock_call.called
            assert judge.temperature == 0.0

    def test_judge_constructs_correct_prompt(self, test_config, sample_successful_trajectory):
        """Test that judge constructs prompt correctly"""
        judge = TrajectoryJudge(test_config)

        with patch.object(judge, '_call_llm', return_value="Status: success") as mock_call:
            judge.judge_trajectory_success(
                query=sample_successful_trajectory["query"],
                trajectory=sample_successful_trajectory["trajectory"],
                final_state=sample_successful_trajectory["final_state"],
                model_output=sample_successful_trajectory["model_output"]
            )

            # Get the prompt that was passed to _call_llm
            call_args = mock_call.call_args[0]
            prompt = call_args[0]

            # Verify prompt contains all inputs
            assert sample_successful_trajectory["query"] in prompt
            assert sample_successful_trajectory["trajectory"] in prompt
            assert sample_successful_trajectory["final_state"] in prompt


class TestJudgeWithConfidence:
    """Tests for judge_with_confidence method"""

    def test_judge_with_confidence_unanimous_success(self, test_config):
        """Test confidence judgment with unanimous success votes"""
        judge = TrajectoryJudge(test_config)

        # Mock judge_trajectory_success to always return True
        with patch.object(judge, 'judge_trajectory_success', return_value=True):
            majority_vote, confidence = judge.judge_with_confidence(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                num_samples=3
            )

        assert majority_vote is True
        assert confidence == 1.0, "Unanimous success should have 100% confidence"

    def test_judge_with_confidence_unanimous_failure(self, test_config):
        """Test confidence judgment with unanimous failure votes"""
        judge = TrajectoryJudge(test_config)

        # Mock judge_trajectory_success to always return False
        with patch.object(judge, 'judge_trajectory_success', return_value=False):
            majority_vote, confidence = judge.judge_with_confidence(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                num_samples=3
            )

        assert majority_vote is False
        assert confidence == 1.0, "Unanimous failure should have 100% confidence"

    def test_judge_with_confidence_majority_success(self, test_config):
        """Test confidence judgment with majority success (2/3)"""
        judge = TrajectoryJudge(test_config)

        # Mock to return [True, True, False]
        mock_results = [True, True, False]
        with patch.object(judge, 'judge_trajectory_success', side_effect=mock_results):
            majority_vote, confidence = judge.judge_with_confidence(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                num_samples=3
            )

        assert majority_vote is True
        assert confidence == pytest.approx(2/3), "2/3 success should have 66.7% confidence"

    def test_judge_with_confidence_majority_failure(self, test_config):
        """Test confidence judgment with majority failure (1/3 success)"""
        judge = TrajectoryJudge(test_config)

        # Mock to return [True, False, False]
        mock_results = [True, False, False]
        with patch.object(judge, 'judge_trajectory_success', side_effect=mock_results):
            majority_vote, confidence = judge.judge_with_confidence(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                num_samples=3
            )

        assert majority_vote is False
        assert confidence == pytest.approx(2/3), "2/3 failure should have 66.7% confidence"

    def test_judge_with_confidence_handles_parse_errors(self, test_config):
        """Test confidence judgment treats parse errors as failures"""
        judge = TrajectoryJudge(test_config)

        # Mock to raise ValueError (parse error)
        with patch.object(judge, 'judge_trajectory_success', side_effect=ValueError("Cannot parse")):
            majority_vote, confidence = judge.judge_with_confidence(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                num_samples=3
            )

        assert majority_vote is False, "Parse errors should be treated as failures"
        assert confidence == 1.0, "All parse errors should give unanimous failure"

    def test_judge_with_confidence_custom_samples(self, test_config):
        """Test confidence judgment with custom number of samples"""
        judge = TrajectoryJudge(test_config)

        # Test with 5 samples
        with patch.object(judge, 'judge_trajectory_success', return_value=True):
            majority_vote, confidence = judge.judge_with_confidence(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                num_samples=5
            )

        assert majority_vote is True
        assert confidence == 1.0


class TestConvenienceFunction:
    """Tests for standalone judge_trajectory_success function"""

    def test_convenience_function_returns_judgment(self, test_config):
        """Test convenience function creates judge and returns result"""
        with patch('reasoningbank.judge.TrajectoryJudge') as MockJudge:
            # Setup mock
            mock_judge_instance = MockJudge.return_value
            mock_judge_instance.judge_trajectory_success.return_value = True

            # Call convenience function
            result = judge_trajectory_success(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                config=test_config
            )

            # Verify it created a judge and called the method
            MockJudge.assert_called_once_with(test_config)
            mock_judge_instance.judge_trajectory_success.assert_called_once()
            assert result is True


class TestJudgeLLMIntegration:
    """Tests for LLM provider integration"""

    def test_call_llm_anthropic_format(self, test_config):
        """Test _call_llm with Anthropic provider"""
        test_config.llm_provider = "anthropic"
        judge = TrajectoryJudge(test_config)

        # Mock Anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Status: success")]
        judge.client.messages.create = MagicMock(return_value=mock_response)

        response = judge._call_llm("Test prompt")

        # Verify call format
        judge.client.messages.create.assert_called_once()
        call_kwargs = judge.client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 10  # Small max_tokens for judge
        assert response == "Status: success"

    def test_call_llm_openai_format(self, test_config):
        """Test _call_llm with OpenAI provider"""
        test_config.llm_provider = "openai"
        judge = TrajectoryJudge(test_config)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Status: failure"))]
        judge.client.chat.completions.create = MagicMock(return_value=mock_response)

        response = judge._call_llm("Test prompt")

        # Verify call format
        judge.client.chat.completions.create.assert_called_once()
        call_kwargs = judge.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 10
        assert response == "Status: failure"

    @pytest.mark.skipif(not GOOGLE_AVAILABLE, reason="google-generativeai not installed")
    def test_call_llm_google_format(self, test_config):
        """Test _call_llm with Google provider"""
        test_config.llm_provider = "google"
        judge = TrajectoryJudge(test_config)

        # Mock Google client
        mock_response = MagicMock()
        mock_response.text = "Status: success"
        judge.client.generate_content = MagicMock(return_value=mock_response)

        response = judge._call_llm("Test prompt")

        # Verify call was made
        judge.client.generate_content.assert_called_once()
        assert response == "Status: success"


@pytest.mark.unit
class TestJudgeIntegration:
    """Integration tests for judge component"""

    def test_full_judge_workflow_success(self, test_config, sample_successful_trajectory):
        """Test complete workflow for successful trajectory"""
        judge = TrajectoryJudge(test_config)

        # Mock LLM response
        with patch.object(judge, '_call_llm', return_value="Thoughts: Correct\nStatus: success"):
            success = judge.judge_trajectory_success(
                query=sample_successful_trajectory["query"],
                trajectory=sample_successful_trajectory["trajectory"],
                final_state=sample_successful_trajectory["final_state"],
                model_output=sample_successful_trajectory["model_output"]
            )

        assert success is True

    def test_full_judge_workflow_failure(self, test_config, sample_failed_trajectory):
        """Test complete workflow for failed trajectory"""
        judge = TrajectoryJudge(test_config)

        # Mock LLM response
        with patch.object(judge, '_call_llm', return_value="Thoughts: Incorrect\nStatus: failure"):
            success = judge.judge_trajectory_success(
                query=sample_failed_trajectory["query"],
                trajectory=sample_failed_trajectory["trajectory"],
                final_state=sample_failed_trajectory["final_state"],
                model_output=sample_failed_trajectory["model_output"]
            )

        assert success is False

    def test_judge_deterministic_with_temperature_zero(self, test_config):
        """Test that temperature=0.0 produces deterministic results"""
        judge = TrajectoryJudge(test_config)

        # Verify temperature is 0.0
        assert judge.temperature == 0.0, "Paper requires temperature=0.0 for deterministic judge"

        # Mock consistent responses
        with patch.object(judge, '_call_llm', return_value="Status: success"):
            results = []
            for _ in range(3):
                success = judge.judge_trajectory_success(
                    query="Test",
                    trajectory="Test",
                    final_state="Test",
                    model_output="Test"
                )
                results.append(success)

        # All results should be identical (deterministic)
        assert all(r == results[0] for r in results), "Results should be deterministic with temperature=0.0"
