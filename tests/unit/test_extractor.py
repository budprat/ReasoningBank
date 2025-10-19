"""
ABOUTME: Unit tests for MemoryExtractor component
ABOUTME: Tests memory extraction from successes (strategies) and failures (lessons)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from reasoningbank.extractor import MemoryExtractor, extract_memories
from reasoningbank.models import MemoryItem
from reasoningbank.config import ReasoningBankConfig


class TestMemoryExtractorInitialization:
    """Tests for MemoryExtractor initialization"""

    def test_extractor_initialization_anthropic(self, test_config):
        """Test MemoryExtractor initialization with Anthropic"""
        test_config.llm_provider = "anthropic"
        test_config.llm_model = "claude-3-5-sonnet-20241022"

        extractor = MemoryExtractor(test_config)

        assert extractor.config == test_config
        assert extractor.llm_provider == "anthropic"
        assert extractor.llm_model == "claude-3-5-sonnet-20241022"
        assert extractor.temperature == 1.0, "Extractor temperature must be 1.0 for diverse extraction"
        assert extractor.max_items == 3, "Paper uses max 3 items per trajectory"
        assert extractor.client is not None

    def test_extractor_temperature_from_config(self, test_config):
        """Test that extractor uses extractor_temperature from config"""
        # Paper specifies extractor_temperature=1.0 (Appendix A.2)
        test_config.extractor_temperature = 1.0

        extractor = MemoryExtractor(test_config)

        assert extractor.temperature == 1.0

    def test_extractor_max_items_from_config(self, test_config):
        """Test that extractor respects max_memory_items_per_trajectory"""
        test_config.max_memory_items_per_trajectory = 3

        extractor = MemoryExtractor(test_config)

        assert extractor.max_items == 3

    def test_unsupported_provider_raises_error(self, test_config):
        """Test that unsupported provider raises ValueError"""
        test_config.llm_provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            MemoryExtractor(test_config)


class TestSuccessPromptBuilding:
    """Tests for success extraction prompt (Figure 8)"""

    def test_build_success_prompt_structure(self, test_config):
        """Test success prompt matches Figure 8 structure"""
        extractor = MemoryExtractor(test_config)

        prompt = extractor._build_success_prompt(
            query="Test query",
            trajectory="Test trajectory",
            final_state="Test state",
            model_output="Test output"
        )

        # Check prompt contains required elements from Figure 8
        assert "expert in web navigation" in prompt
        assert "successfully accomplished the task" in prompt
        assert "extract and summarize useful insights" in prompt
        assert "memory items" in prompt

    def test_build_success_prompt_guidelines(self, test_config):
        """Test success prompt includes extraction guidelines"""
        extractor = MemoryExtractor(test_config)

        prompt = extractor._build_success_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test"
        )

        # Paper specifies specific guidelines
        assert "Guidelines" in prompt
        assert "generalizable" in prompt.lower()
        assert "must not repeat similar or overlapping items" in prompt

    def test_build_success_prompt_max_items(self, test_config):
        """Test success prompt includes max_items limit"""
        test_config.max_memory_items_per_trajectory = 3
        extractor = MemoryExtractor(test_config)

        prompt = extractor._build_success_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test"
        )

        # Should mention the max items limit
        assert "at most 3" in prompt or "3 memory items" in prompt

    def test_build_success_prompt_markdown_format(self, test_config):
        """Test success prompt specifies Markdown output format"""
        extractor = MemoryExtractor(test_config)

        prompt = extractor._build_success_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test"
        )

        # Paper requires Markdown format (Figure 8)
        assert "# Memory Item" in prompt
        assert "## Title" in prompt
        assert "## Description" in prompt
        assert "## Content" in prompt

    def test_build_success_prompt_includes_inputs(self, test_config):
        """Test success prompt includes query and trajectory"""
        extractor = MemoryExtractor(test_config)

        query = "Calculate 25 * 4"
        trajectory = "<think>Multiply</think><action>calculate 25*4</action>"

        prompt = extractor._build_success_prompt(
            query=query,
            trajectory=trajectory,
            final_state="Result: 100",
            model_output="100"
        )

        assert query in prompt
        assert trajectory in prompt


class TestFailurePromptBuilding:
    """Tests for failure extraction prompt (Figure 8)"""

    def test_build_failure_prompt_structure(self, test_config):
        """Test failure prompt matches Figure 8 structure"""
        extractor = MemoryExtractor(test_config)

        prompt = extractor._build_failure_prompt(
            query="Test query",
            trajectory="Test trajectory",
            final_state="Test state",
            model_output="Test output"
        )

        # Check prompt contains required elements for failure extraction
        assert "expert in web navigation" in prompt
        assert "attempted to resolve the task but failed" in prompt
        assert "extract and summarize useful insights" in prompt

    def test_build_failure_prompt_preventative_focus(self, test_config):
        """Test failure prompt focuses on preventative lessons"""
        extractor = MemoryExtractor(test_config)

        prompt = extractor._build_failure_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test"
        )

        # Failure prompt should focus on learning from mistakes
        assert "why the trajectory failed" in prompt.lower()
        assert "lessons" in prompt.lower() or "prevent" in prompt.lower()

    def test_build_failure_prompt_different_from_success(self, test_config):
        """Test failure prompt differs from success prompt"""
        extractor = MemoryExtractor(test_config)

        success_prompt = extractor._build_success_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test"
        )

        failure_prompt = extractor._build_failure_prompt(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test"
        )

        # Prompts should be different
        assert success_prompt != failure_prompt
        assert "successfully accomplished" in success_prompt
        assert "failed" in failure_prompt.lower()


class TestMarkdownParsing:
    """Tests for parsing Markdown extraction responses"""

    def test_parse_single_memory_item(self, test_config):
        """Test parsing single memory item from Markdown"""
        extractor = MemoryExtractor(test_config)

        response = """# Memory Item 1
## Title Test Title
## Description Test description
## Content Test content explaining the insight learned."""

        items = extractor._parse_extraction_response(
            response,
            success_signal=True,
            source_task_id="test_001"
        )

        assert len(items) == 1
        assert items[0].title == "Test Title"
        assert items[0].description == "Test description"
        assert items[0].content == "Test content explaining the insight learned."
        assert items[0].success_signal is True
        assert items[0].source_task_id == "test_001"

    def test_parse_multiple_memory_items(self, test_config):
        """Test parsing multiple memory items from Markdown"""
        extractor = MemoryExtractor(test_config)

        response = """# Memory Item 1
## Title First Title
## Description First description
## Content First content

# Memory Item 2
## Title Second Title
## Description Second description
## Content Second content"""

        items = extractor._parse_extraction_response(
            response,
            success_signal=True
        )

        assert len(items) == 2
        assert items[0].title == "First Title"
        assert items[1].title == "Second Title"

    def test_parse_with_title_on_next_line(self, test_config):
        """Test parsing when title/description/content are on next line"""
        extractor = MemoryExtractor(test_config)

        response = """# Memory Item 1
## Title
Step-by-step Calculation
## Description
Break down complex calculations into steps
## Content
When solving arithmetic, calculate operations one at a time."""

        items = extractor._parse_extraction_response(
            response,
            success_signal=True
        )

        assert len(items) == 1
        assert items[0].title == "Step-by-step Calculation"
        assert items[0].description == "Break down complex calculations into steps"
        assert "calculate operations one at a time" in items[0].content

    def test_parse_with_code_fences(self, test_config):
        """Test parsing with markdown code fences"""
        extractor = MemoryExtractor(test_config)

        response = """```
# Memory Item 1
## Title Test Title
## Description Test description
## Content Test content
```"""

        items = extractor._parse_extraction_response(
            response,
            success_signal=True
        )

        assert len(items) == 1
        assert items[0].title == "Test Title"

    def test_parse_empty_response_raises_error(self, test_config):
        """Test that empty response raises ValueError"""
        extractor = MemoryExtractor(test_config)

        with pytest.raises(ValueError, match="No valid memory items found"):
            extractor._parse_extraction_response("", success_signal=True)

    def test_parse_invalid_format_raises_error(self, test_config):
        """Test that invalid format raises ValueError"""
        extractor = MemoryExtractor(test_config)

        # Missing required sections
        response = "# Memory Item 1\n## Title Test"  # Missing description and content

        with pytest.raises(ValueError, match="No valid memory items found"):
            extractor._parse_extraction_response(response, success_signal=True)

    def test_parse_with_multiline_content(self, test_config):
        """Test parsing content that spans multiple lines"""
        extractor = MemoryExtractor(test_config)

        response = """# Memory Item 1
## Title Multi-line Content
## Description This has multiple lines of content
## Content
This is the first line.
This is the second line.
This is the third line."""

        items = extractor._parse_extraction_response(
            response,
            success_signal=True
        )

        assert len(items) == 1
        assert "first line" in items[0].content
        assert "second line" in items[0].content
        assert "third line" in items[0].content


class TestMemoryExtraction:
    """Tests for extract_memories method"""

    def test_extract_memories_success_trajectory(self, test_config, mock_extractor_responses):
        """Test extracting memories from successful trajectory"""
        extractor = MemoryExtractor(test_config)

        # Mock LLM call to return success memories
        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["success"]):
            items = extractor.extract_memories(
                query="What is 25 * 4?",
                trajectory="<think>Calculate</think>",
                final_state="Result: 100",
                model_output="100",
                success=True,
                source_task_id="test_001"
            )

        assert len(items) > 0
        assert all(isinstance(item, MemoryItem) for item in items)
        assert all(item.success_signal is True for item in items)
        assert all(item.source_task_id == "test_001" for item in items)

    def test_extract_memories_failure_trajectory(self, test_config, mock_extractor_responses):
        """Test extracting memories from failed trajectory"""
        extractor = MemoryExtractor(test_config)

        # Mock LLM call to return failure memories
        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["failure"]):
            items = extractor.extract_memories(
                query="What is 25 * 4?",
                trajectory="<think>Wrong approach</think>",
                final_state="Error",
                model_output="Error",
                success=False,
                source_task_id="test_002"
            )

        assert len(items) > 0
        assert all(item.success_signal is False for item in items)
        assert all(item.source_task_id == "test_002" for item in items)

    def test_extract_memories_enforces_max_items(self, test_config):
        """Test that extraction enforces max_items limit (3)"""
        test_config.max_memory_items_per_trajectory = 3
        extractor = MemoryExtractor(test_config)

        # Mock response with 5 memory items
        response_with_5_items = """# Memory Item 1
## Title Title 1
## Description Desc 1
## Content Content 1

# Memory Item 2
## Title Title 2
## Description Desc 2
## Content Content 2

# Memory Item 3
## Title Title 3
## Description Desc 3
## Content Content 3

# Memory Item 4
## Title Title 4
## Description Desc 4
## Content Content 4

# Memory Item 5
## Title Title 5
## Description Desc 5
## Content Content 5"""

        with patch.object(extractor, '_call_llm', return_value=response_with_5_items):
            items = extractor.extract_memories(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                success=True
            )

        # Should only return first 3 items
        assert len(items) == 3
        assert items[0].title == "Title 1"
        assert items[1].title == "Title 2"
        assert items[2].title == "Title 3"

    def test_extract_memories_uses_correct_prompt(self, test_config):
        """Test that extract_memories uses correct prompt based on success/failure"""
        extractor = MemoryExtractor(test_config)

        # Mock both prompt builders and LLM call
        with patch.object(extractor, '_build_success_prompt', return_value="success prompt") as mock_success, \
             patch.object(extractor, '_build_failure_prompt', return_value="failure prompt") as mock_failure, \
             patch.object(extractor, '_call_llm', return_value="# Memory Item 1\n## Title T\n## Description D\n## Content C"):

            # Test success case
            extractor.extract_memories(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                success=True
            )
            mock_success.assert_called_once()
            mock_failure.assert_not_called()

            # Reset mocks
            mock_success.reset_mock()
            mock_failure.reset_mock()

            # Test failure case
            extractor.extract_memories(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                success=False
            )
            mock_failure.assert_called_once()
            mock_success.assert_not_called()

    def test_extract_memories_uses_temperature_1_0(self, test_config):
        """Test that extractor uses temperature=1.0 for diversity"""
        extractor = MemoryExtractor(test_config)

        assert extractor.temperature == 1.0, "Paper requires temperature=1.0 for diverse extraction"


class TestSelfContrastExtraction:
    """Tests for self-contrast extraction (MaTTS parallel)"""

    def test_extract_with_self_contrast_multiple_trajectories(self, test_config):
        """Test self-contrast extraction with multiple trajectories"""
        extractor = MemoryExtractor(test_config)

        # Prepare multiple trajectories (success and failure)
        trajectories = [
            ("Trajectory 1", "State 1", "Output 1", "Query", True),
            ("Trajectory 2", "State 2", "Output 2", "Query", True),
            ("Trajectory 3", "State 3", "Output 3", "Query", False)
        ]

        # Mock LLM response
        mock_response = """# Memory Item 1
## Title Systematic Approach
## Description Successful trajectories follow systematic steps
## Content Comparing successful and failed attempts reveals importance of methodology."""

        with patch.object(extractor, '_call_llm', return_value=mock_response):
            items = extractor.extract_with_self_contrast(
                trajectories=trajectories,
                query="Test query",
                source_task_id="matts_001"
            )

        assert len(items) > 0
        assert all(isinstance(item, MemoryItem) for item in items)
        assert items[0].source_task_id == "matts_001"

    def test_self_contrast_enforces_max_aggregated(self, test_config):
        """Test self-contrast enforces max_memory_items_aggregated limit (5)"""
        test_config.max_memory_items_aggregated = 5
        extractor = MemoryExtractor(test_config)

        trajectories = [
            ("T1", "S1", "O1", "Q", True),
            ("T2", "S2", "O2", "Q", True),
            ("T3", "S3", "O3", "Q", False)
        ]

        # Mock response with 7 items
        response_with_7_items = "\n\n".join([
            f"# Memory Item {i}\n## Title Title {i}\n## Description Desc {i}\n## Content Content {i}"
            for i in range(1, 8)
        ])

        with patch.object(extractor, '_call_llm', return_value=response_with_7_items):
            items = extractor.extract_with_self_contrast(
                trajectories=trajectories,
                query="Test"
            )

        # Should only return first 5 items (max_memory_items_aggregated)
        assert len(items) == 5

    def test_build_self_contrast_prompt_structure(self, test_config):
        """Test self-contrast prompt includes all trajectories"""
        extractor = MemoryExtractor(test_config)

        trajectories = [
            ("Traj1", "State1", "Out1", "Q", True),
            ("Traj2", "State2", "Out2", "Q", False)
        ]

        prompt = extractor._build_self_contrast_prompt(trajectories, "Test query")

        # Check prompt structure
        assert "multiple trajectories" in prompt
        assert "compare and contrast" in prompt.lower()
        assert "self-contrast reasoning" in prompt.lower()
        assert "Trajectory 1 (SUCCESS)" in prompt
        assert "Trajectory 2 (FAILURE)" in prompt

    def test_self_contrast_prompt_includes_guidelines(self, test_config):
        """Test self-contrast prompt includes comparison guidelines"""
        extractor = MemoryExtractor(test_config)

        trajectories = [("T", "S", "O", "Q", True)]
        prompt = extractor._build_self_contrast_prompt(trajectories, "Query")

        # Paper specifies specific comparison guidelines
        assert "consistently led to success" in prompt.lower()
        assert "mistakes or inefficiencies" in prompt.lower()
        assert "generalizable" in prompt.lower()

    def test_self_contrast_prompt_max_items(self, test_config):
        """Test self-contrast prompt uses max_memory_items_aggregated"""
        test_config.max_memory_items_aggregated = 5
        extractor = MemoryExtractor(test_config)

        trajectories = [("T", "S", "O", "Q", True)]
        prompt = extractor._build_self_contrast_prompt(trajectories, "Query")

        # Should mention aggregated limit
        assert "at most 5" in prompt or "5 memory items" in prompt


class TestConvenienceFunction:
    """Tests for standalone extract_memories function"""

    def test_convenience_function_returns_memories(self, test_config):
        """Test convenience function creates extractor and returns memories"""
        with patch('reasoningbank.extractor.MemoryExtractor') as MockExtractor:
            # Setup mock
            mock_extractor_instance = MockExtractor.return_value
            mock_item = MemoryItem(
                title="Test",
                description="Test",
                content="Test"
            )
            mock_extractor_instance.extract_memories.return_value = [mock_item]

            # Call convenience function
            result = extract_memories(
                query="Test",
                trajectory="Test",
                final_state="Test",
                model_output="Test",
                success=True,
                config=test_config,
                source_task_id="test_001"
            )

            # Verify it created an extractor and called the method
            MockExtractor.assert_called_once_with(test_config)
            mock_extractor_instance.extract_memories.assert_called_once()
            assert len(result) == 1
            assert result[0].title == "Test"


class TestExtractorLLMIntegration:
    """Tests for LLM provider integration"""

    def test_call_llm_anthropic_format(self, test_config):
        """Test _call_llm with Anthropic provider"""
        test_config.llm_provider = "anthropic"
        extractor = MemoryExtractor(test_config)

        # Mock Anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="# Memory Item 1\n## Title T\n## Description D\n## Content C")]
        extractor.client.messages.create = MagicMock(return_value=mock_response)

        response = extractor._call_llm("Test prompt")

        # Verify call format
        extractor.client.messages.create.assert_called_once()
        call_kwargs = extractor.client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0  # Extractor uses temperature=1.0
        assert call_kwargs["max_tokens"] == 2000  # Room for multiple items
        assert "# Memory Item 1" in response


@pytest.mark.unit
class TestExtractorIntegration:
    """Integration tests for extractor component"""

    def test_full_extraction_workflow_success(self, test_config, sample_successful_trajectory, mock_extractor_responses):
        """Test complete extraction workflow for successful trajectory"""
        extractor = MemoryExtractor(test_config)

        # Mock LLM response
        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["success"]):
            items = extractor.extract_memories(
                query=sample_successful_trajectory["query"],
                trajectory=sample_successful_trajectory["trajectory"],
                final_state=sample_successful_trajectory["final_state"],
                model_output=sample_successful_trajectory["model_output"],
                success=True,
                source_task_id="integration_test"
            )

        # Verify extraction results
        assert len(items) > 0
        assert all(isinstance(item, MemoryItem) for item in items)
        assert all(item.success_signal is True for item in items)
        assert all(item.title and item.description and item.content for item in items)

    def test_full_extraction_workflow_failure(self, test_config, sample_failed_trajectory, mock_extractor_responses):
        """Test complete extraction workflow for failed trajectory"""
        extractor = MemoryExtractor(test_config)

        # Mock LLM response
        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["failure"]):
            items = extractor.extract_memories(
                query=sample_failed_trajectory["query"],
                trajectory=sample_failed_trajectory["trajectory"],
                final_state=sample_failed_trajectory["final_state"],
                model_output=sample_failed_trajectory["model_output"],
                success=False,
                source_task_id="integration_test"
            )

        # Verify extraction results
        assert len(items) > 0
        assert all(item.success_signal is False for item in items)

    def test_extraction_diverse_with_temperature_1_0(self, test_config):
        """Test that temperature=1.0 allows diverse extraction"""
        extractor = MemoryExtractor(test_config)

        # Verify temperature is 1.0
        assert extractor.temperature == 1.0, "Paper requires temperature=1.0 for diverse extraction"
