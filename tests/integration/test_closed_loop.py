"""
ABOUTME: Integration tests for complete ReasoningBank closed-loop learning cycle
ABOUTME: Tests Retrieve → Act → Judge → Extract → Consolidate workflow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from reasoningbank.judge import TrajectoryJudge
from reasoningbank.extractor import MemoryExtractor
from reasoningbank.retriever import MemoryRetriever
from reasoningbank.consolidator import MemoryConsolidator
from reasoningbank.models import MemoryEntry, MemoryItem, TrajectoryResult
from reasoningbank.config import ReasoningBankConfig


@pytest.mark.integration
class TestClosedLoopBasic:
    """Basic closed-loop workflow tests"""

    def test_judge_extract_consolidate_workflow(self, test_config, sample_successful_trajectory, mock_judge_responses, mock_extractor_responses):
        """Test Judge → Extract → Consolidate workflow"""
        # Initialize components
        judge = TrajectoryJudge(test_config)
        extractor = MemoryExtractor(test_config)
        consolidator = MemoryConsolidator(test_config)

        # Step 1: Judge trajectory
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["success"]):
            success = judge.judge_trajectory_success(
                query=sample_successful_trajectory["query"],
                trajectory=sample_successful_trajectory["trajectory"],
                final_state=sample_successful_trajectory["final_state"],
                model_output=sample_successful_trajectory["model_output"]
            )

        assert success is True

        # Step 2: Extract memories
        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["success"]):
            memory_items = extractor.extract_memories(
                query=sample_successful_trajectory["query"],
                trajectory=sample_successful_trajectory["trajectory"],
                final_state=sample_successful_trajectory["final_state"],
                model_output=sample_successful_trajectory["model_output"],
                success=success,
                source_task_id="test_001"
            )

        assert len(memory_items) > 0

        # Step 3: Consolidate into memory bank
        entry_id = consolidator.add_from_trajectory(
            query=sample_successful_trajectory["query"],
            trajectory=sample_successful_trajectory["trajectory"],
            final_state=sample_successful_trajectory["final_state"],
            model_output=sample_successful_trajectory["model_output"],
            success=success,
            memory_items=memory_items,
            steps_taken=3
        )

        # Verify consolidation
        assert entry_id is not None
        retrieved_entry = consolidator.get_entry(entry_id)
        assert retrieved_entry is not None
        assert len(retrieved_entry.memory_items) > 0

    def test_retrieve_from_consolidated_memories(self, test_config, sample_memory_items, mock_embedding_responses):
        """Test retrieving memories after consolidation"""
        # Initialize components
        consolidator = MemoryConsolidator(test_config)
        retriever = MemoryRetriever(test_config)

        # Add memories to bank
        consolidator.add_from_trajectory(
            query="Calculate arithmetic",
            trajectory="Test trajectory",
            final_state="Test state",
            model_output="Test output",
            success=True,
            memory_items=sample_memory_items
        )

        # Retrieve relevant memories
        with patch.object(retriever, 'embed_text', side_effect=[
            mock_embedding_responses["query"],
            mock_embedding_responses["memory1"],
            mock_embedding_responses["memory2"],
            mock_embedding_responses["memory3"]
        ]):
            retrieved = retriever.retrieve(
                "Calculate 25 * 4",
                consolidator.get_all_entries(),
                k=1
            )

        assert len(retrieved) > 0
        assert isinstance(retrieved[0], MemoryItem)


@pytest.mark.integration
class TestFullClosedLoop:
    """Complete closed-loop learning tests"""

    def test_complete_learning_cycle(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test complete cycle: Task 1 → Learn → Task 2 (with retrieval)"""
        # Initialize all components
        judge = TrajectoryJudge(test_config)
        extractor = MemoryExtractor(test_config)
        consolidator = MemoryConsolidator(test_config)
        retriever = MemoryRetriever(test_config)

        # ===== TASK 1: First execution (no prior memories) =====
        task1_query = "Calculate 25 * 4"
        task1_trajectory = "<think>Use order of operations</think><action>calculate 25*4</action><observation>100</observation>"
        task1_final = "Result: 100"
        task1_output = "100"

        # Judge Task 1
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["success"]):
            task1_success = judge.judge_trajectory_success(
                task1_query, task1_trajectory, task1_final, task1_output
            )

        assert task1_success is True

        # Extract memories from Task 1
        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["success"]):
            task1_memories = extractor.extract_memories(
                task1_query, task1_trajectory, task1_final, task1_output,
                success=task1_success,
                source_task_id="task_001"
            )

        assert len(task1_memories) > 0

        # Consolidate Task 1 memories
        entry1_id = consolidator.add_from_trajectory(
            task1_query, task1_trajectory, task1_final, task1_output,
            success=task1_success,
            memory_items=task1_memories,
            steps_taken=3
        )

        # ===== TASK 2: Second execution (with retrieval) =====
        task2_query = "Calculate 15 * 3"

        # Retrieve relevant memories for Task 2
        with patch.object(retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
            retrieved_memories = retriever.retrieve(
                task2_query,
                consolidator.get_all_entries(),
                k=1
            )

        # Should have retrieved memories from Task 1
        assert len(retrieved_memories) > 0

        # Verify memory bank has grown
        stats = consolidator.get_statistics()
        assert stats["total_entries"] == 1
        assert stats["successful_entries"] == 1

    def test_learning_from_success_and_failure(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test learning from both successful and failed trajectories"""
        judge = TrajectoryJudge(test_config)
        extractor = MemoryExtractor(test_config)
        consolidator = MemoryConsolidator(test_config)

        # ===== SUCCESSFUL TRAJECTORY =====
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["success"]):
            success_judgment = judge.judge_trajectory_success(
                "Query 1", "Traj 1", "State 1", "Output 1"
            )

        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["success"]):
            success_memories = extractor.extract_memories(
                "Query 1", "Traj 1", "State 1", "Output 1",
                success=success_judgment
            )

        consolidator.add_from_trajectory(
            "Query 1", "Traj 1", "State 1", "Output 1",
            success=True,
            memory_items=success_memories
        )

        # ===== FAILED TRAJECTORY =====
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["failure"]):
            failure_judgment = judge.judge_trajectory_success(
                "Query 2", "Traj 2", "State 2", "Output 2"
            )

        with patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["failure"]):
            failure_memories = extractor.extract_memories(
                "Query 2", "Traj 2", "State 2", "Output 2",
                success=failure_judgment
            )

        consolidator.add_from_trajectory(
            "Query 2", "Traj 2", "State 2", "Output 2",
            success=False,
            memory_items=failure_memories
        )

        # Verify both types of memories are stored
        success_entries = consolidator.get_success_entries()
        failure_entries = consolidator.get_failure_entries()

        assert len(success_entries) == 1
        assert len(failure_entries) == 1

        # Verify different types of memory items
        success_items = success_entries[0].memory_items
        failure_items = failure_entries[0].memory_items

        assert all(item.success_signal is True for item in success_items)
        assert all(item.success_signal is False for item in failure_items)


@pytest.mark.integration
class TestMemoryInjection:
    """Tests for memory injection into agent context"""

    def test_memory_injection_improves_context(self, test_config, sample_memory_items):
        """Test that retrieved memories enhance agent context"""
        consolidator = MemoryConsolidator(test_config)
        retriever = MemoryRetriever(test_config)

        # Add memories to bank
        consolidator.add_from_trajectory(
            query="Arithmetic calculation",
            trajectory="Test",
            final_state="Test",
            model_output="Test",
            success=True,
            memory_items=sample_memory_items
        )

        # Retrieve for similar task
        with patch.object(retriever, 'embed_text', return_value=[0.1] * 768):
            retrieved = retriever.retrieve(
                "Calculate 50 * 2",
                consolidator.get_all_entries(),
                k=1
            )

        # Format memories for injection
        memory_context = "\n".join([
            f"Memory: {item.title}\nDescription: {item.description}\nContent: {item.content}"
            for item in retrieved
        ])

        # Verify memory context is non-empty and useful
        assert len(memory_context) > 0
        assert "Memory:" in memory_context

    def test_retrieval_filters_by_relevance(self, test_config, mock_embedding_responses):
        """Test that retrieval selects most relevant memories"""
        consolidator = MemoryConsolidator(test_config)
        retriever = MemoryRetriever(test_config)

        # Add diverse memories
        memory1 = MemoryItem(
            title="Arithmetic Strategy",
            description="Arithmetic operations",
            content="Use order of operations for calculations"
        )
        memory2 = MemoryItem(
            title="Navigation Strategy",
            description="Web navigation",
            content="Click links to navigate pages"
        )

        consolidator.add_from_trajectory(
            "Calculate", "T1", "S1", "O1", True, [memory1]
        )
        consolidator.add_from_trajectory(
            "Navigate", "T2", "S2", "O2", True, [memory2]
        )

        # Retrieve for arithmetic query
        def mock_embed(text):
            if "arithmetic" in text.lower() or "calculate" in text.lower():
                return [1.0, 0.0, 0.0]  # Arithmetic cluster
            else:
                return [0.0, 1.0, 0.0]  # Navigation cluster

        with patch.object(retriever, 'embed_text', side_effect=mock_embed):
            retrieved = retriever.retrieve(
                "Calculate 25 * 4",
                consolidator.get_all_entries(),
                k=1
            )

        # Should retrieve arithmetic memory
        assert len(retrieved) == 1
        assert "Arithmetic" in retrieved[0].title


@pytest.mark.integration
class TestPersistenceAcrossSessions:
    """Tests for persistence across sessions"""

    def test_memories_persist_across_sessions(self, test_config, sample_memory_items):
        """Test that memories persist across consolidator instances"""
        # Session 1: Create and save memories
        consolidator1 = MemoryConsolidator(test_config)
        entry_id = consolidator1.add_from_trajectory(
            query="Persistent query",
            trajectory="Test",
            final_state="Test",
            model_output="Test",
            success=True,
            memory_items=sample_memory_items
        )

        # Session 2: Load and verify
        consolidator2 = MemoryConsolidator(test_config)

        retrieved_entry = consolidator2.get_entry(entry_id)
        assert retrieved_entry is not None
        assert len(retrieved_entry.memory_items) == len(sample_memory_items)

    def test_retrieval_works_after_reload(self, test_config, mock_embedding_responses):
        """Test that retrieval works after reloading memory bank"""
        # Session 1: Add memories
        consolidator1 = MemoryConsolidator(test_config)
        memory = MemoryItem(
            title="Test Memory",
            description="Test",
            content="Test content"
        )
        consolidator1.add_from_trajectory(
            "Test query", "T", "S", "O", True, [memory]
        )

        # Session 2: Reload and retrieve
        consolidator2 = MemoryConsolidator(test_config)
        retriever = MemoryRetriever(test_config)

        with patch.object(retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
            retrieved = retriever.retrieve(
                "Test query",
                consolidator2.get_all_entries(),
                k=1
            )

        assert len(retrieved) > 0


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end workflow tests"""

    def test_multi_task_learning_workflow(self, test_config, mock_judge_responses, mock_extractor_responses, mock_embedding_responses):
        """Test multi-task learning scenario"""
        # Initialize components
        judge = TrajectoryJudge(test_config)
        extractor = MemoryExtractor(test_config)
        consolidator = MemoryConsolidator(test_config)
        retriever = MemoryRetriever(test_config)

        # Execute 3 tasks and learn from each
        tasks = [
            ("Calculate 25 * 4", "<think>Multiply</think>", "100", "100", True),
            ("Calculate 50 / 2", "<think>Divide</think>", "25", "25", True),
            ("Navigate home", "<think>Click</think>", "Home page", "Success", True)
        ]

        for i, (query, traj, state, output, expected_success) in enumerate(tasks):
            # Judge
            judge_response = mock_judge_responses["success"] if expected_success else mock_judge_responses["failure"]
            with patch.object(judge, '_call_llm', return_value=judge_response):
                success = judge.judge_trajectory_success(query, traj, state, output)

            # Extract
            extract_response = mock_extractor_responses["success"] if expected_success else mock_extractor_responses["failure"]
            with patch.object(extractor, '_call_llm', return_value=extract_response):
                memories = extractor.extract_memories(
                    query, traj, state, output, success, source_task_id=f"task_{i+1}"
                )

            # Consolidate
            consolidator.add_from_trajectory(
                query, traj, state, output, success, memories
            )

        # Verify learning progress
        stats = consolidator.get_statistics()
        assert stats["total_entries"] == 3
        assert stats["successful_entries"] == 3

        # Retrieve for new task
        with patch.object(retriever, 'embed_text', return_value=mock_embedding_responses["query"]):
            retrieved = retriever.retrieve(
                "Calculate 10 * 5",  # Similar to task 1
                consolidator.get_all_entries(),
                k=2
            )

        assert len(retrieved) > 0

    def test_closed_loop_with_statistics_tracking(self, test_config, mock_judge_responses, mock_extractor_responses):
        """Test closed loop with statistics tracking"""
        judge = TrajectoryJudge(test_config)
        extractor = MemoryExtractor(test_config)
        consolidator = MemoryConsolidator(test_config)

        # Initial stats
        initial_stats = consolidator.get_statistics()
        assert initial_stats["total_entries"] == 0

        # Add successful task
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["success"]), \
             patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["success"]):

            success = judge.judge_trajectory_success("Q1", "T1", "S1", "O1")
            memories = extractor.extract_memories("Q1", "T1", "S1", "O1", success)
            consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, memories, steps_taken=5)

        # Add failed task
        with patch.object(judge, '_call_llm', return_value=mock_judge_responses["failure"]), \
             patch.object(extractor, '_call_llm', return_value=mock_extractor_responses["failure"]):

            success = judge.judge_trajectory_success("Q2", "T2", "S2", "O2")
            memories = extractor.extract_memories("Q2", "T2", "S2", "O2", success)
            consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", False, memories, steps_taken=3)

        # Final stats
        final_stats = consolidator.get_statistics()
        assert final_stats["total_entries"] == 2
        assert final_stats["successful_entries"] == 1
        assert final_stats["failed_entries"] == 1
        assert final_stats["success_rate"] == 0.5
        assert final_stats["avg_steps_per_task"] == 4.0  # (5 + 3) / 2
