"""
ABOUTME: Unit tests for ReasoningBank data models
ABOUTME: Tests MemoryItem, MemoryEntry, TrajectoryResult, ReActStep, MaTTSResult
"""

import pytest
from reasoningbank.models import MemoryItem, MemoryEntry, TrajectoryResult, ReActStep, MaTTSResult
import time
import json


class TestMemoryItem:
    """Tests for MemoryItem data model"""

    def test_memory_item_creation(self):
        """Test basic MemoryItem creation"""
        item = MemoryItem(
            title="Test Title",
            description="Test description",
            content="Test content with detailed information"
        )

        assert item.title == "Test Title"
        assert item.description == "Test description"
        assert item.content == "Test content with detailed information"
        assert item.source_task_id is None
        assert item.success_signal is None
        assert item.extraction_timestamp is None

    def test_memory_item_with_optional_fields(self):
        """Test MemoryItem with all optional fields"""
        timestamp = time.time()
        item = MemoryItem(
            title="Test Title",
            description="Test description",
            content="Test content",
            source_task_id="task_123",
            success_signal=True,
            extraction_timestamp=timestamp
        )

        assert item.source_task_id == "task_123"
        assert item.success_signal is True
        assert item.extraction_timestamp == timestamp

    def test_memory_item_to_dict(self):
        """Test MemoryItem serialization to dict"""
        item = MemoryItem(
            title="Test Title",
            description="Test description",
            content="Test content"
        )

        item_dict = item.to_dict()

        assert isinstance(item_dict, dict)
        assert item_dict["title"] == "Test Title"
        assert item_dict["description"] == "Test description"
        assert item_dict["content"] == "Test content"

    def test_memory_item_from_dict(self):
        """Test MemoryItem deserialization from dict"""
        data = {
            "title": "Test Title",
            "description": "Test description",
            "content": "Test content",
            "source_task_id": "task_123",
            "success_signal": True,
            "extraction_timestamp": time.time()
        }

        item = MemoryItem.from_dict(data)

        assert item.title == data["title"]
        assert item.description == data["description"]
        assert item.content == data["content"]
        assert item.source_task_id == data["source_task_id"]
        assert item.success_signal == data["success_signal"]

    def test_memory_item_str_representation(self):
        """Test MemoryItem string representation"""
        item = MemoryItem(
            title="Test Title",
            description="Test description",
            content="Test content",
            success_signal=True
        )

        str_repr = str(item)
        assert "Test Title" in str_repr
        assert "success=True" in str_repr


class TestMemoryEntry:
    """Tests for MemoryEntry data model"""

    def test_memory_entry_creation(self):
        """Test basic MemoryEntry creation"""
        item = MemoryItem(
            title="Test",
            description="Test",
            content="Test"
        )

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test query",
            trajectory="Test trajectory",
            success=True,
            memory_items=[item]
        )

        assert entry.id == "entry_001"
        assert entry.task_query == "Test query"
        assert entry.trajectory == "Test trajectory"
        assert entry.success is True
        assert len(entry.memory_items) == 1
        assert entry.timestamp is not None

    def test_memory_entry_with_optional_fields(self):
        """Test MemoryEntry with all optional fields"""
        item = MemoryItem(
            title="Test",
            description="Test",
            content="Test"
        )

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test query",
            trajectory="Test trajectory",
            success=True,
            memory_items=[item],
            final_state="Final state",
            model_output="Model output",
            steps_taken=5
        )

        assert entry.final_state == "Final state"
        assert entry.model_output == "Model output"
        assert entry.steps_taken == 5

    def test_memory_entry_to_dict(self):
        """Test MemoryEntry serialization"""
        item = MemoryItem(
            title="Test",
            description="Test",
            content="Test"
        )

        entry = MemoryEntry(
            id="entry_001",
            task_query="Test query",
            trajectory="Test trajectory",
            success=True,
            memory_items=[item]
        )

        entry_dict = entry.to_dict()

        assert isinstance(entry_dict, dict)
        assert entry_dict["id"] == "entry_001"
        assert isinstance(entry_dict["memory_items"], list)
        assert len(entry_dict["memory_items"]) == 1

    def test_memory_entry_from_dict(self):
        """Test MemoryEntry deserialization"""
        data = {
            "id": "entry_001",
            "task_query": "Test query",
            "trajectory": "Test trajectory",
            "success": True,
            "memory_items": [
                {
                    "title": "Test",
                    "description": "Test",
                    "content": "Test",
                    "source_task_id": None,
                    "success_signal": None,
                    "extraction_timestamp": None
                }
            ],
            "timestamp": time.time()
        }

        entry = MemoryEntry.from_dict(data)

        assert entry.id == "entry_001"
        assert len(entry.memory_items) == 1
        assert isinstance(entry.memory_items[0], MemoryItem)


class TestTrajectoryResult:
    """Tests for TrajectoryResult data model"""

    def test_trajectory_result_creation(self):
        """Test basic TrajectoryResult creation"""
        result = TrajectoryResult(
            query="Test query",
            full_trajectory="Test trajectory",
            final_state="Final state",
            model_output="Output",
            steps_taken=5
        )

        assert result.query == "Test query"
        assert result.full_trajectory == "Test trajectory"
        assert result.final_state == "Final state"
        assert result.model_output == "Output"
        assert result.steps_taken == 5
        assert result.success is None
        assert result.memory_items is None

    def test_trajectory_result_with_memory(self):
        """Test TrajectoryResult with memory items"""
        item = MemoryItem(
            title="Test",
            description="Test",
            content="Test"
        )

        result = TrajectoryResult(
            query="Test query",
            full_trajectory="Test trajectory",
            final_state="Final state",
            model_output="Output",
            steps_taken=5,
            success=True,
            memory_items=[item],
            entry_id="entry_001"
        )

        assert result.success is True
        assert len(result.memory_items) == 1
        assert result.entry_id == "entry_001"

    def test_trajectory_result_to_dict(self):
        """Test TrajectoryResult serialization"""
        result = TrajectoryResult(
            query="Test query",
            full_trajectory="Test trajectory",
            final_state="Final state",
            model_output="Output",
            steps_taken=5
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["query"] == "Test query"
        assert result_dict["steps_taken"] == 5


class TestReActStep:
    """Tests for ReActStep data model"""

    def test_react_step_creation(self):
        """Test ReActStep creation"""
        step = ReActStep(
            step_num=1,
            think="Thinking process",
            action="Action taken",
            observation="Observation received"
        )

        assert step.step_num == 1
        assert step.think == "Thinking process"
        assert step.action == "Action taken"
        assert step.observation == "Observation received"

    def test_react_step_to_string(self):
        """Test ReActStep string formatting"""
        step = ReActStep(
            step_num=1,
            think="Calculate 25 * 4",
            action="calculate 25 * 4",
            observation="Result: 100"
        )

        step_str = step.to_string()

        assert "<think>Calculate 25 * 4</think>" in step_str
        assert "<action>calculate 25 * 4</action>" in step_str
        assert "<observation>Result: 100</observation>" in step_str


class TestMaTTSResult:
    """Tests for MaTTSResult data model"""

    def test_matts_result_creation(self):
        """Test MaTTSResult creation"""
        traj1 = TrajectoryResult(
            query="Test",
            full_trajectory="Traj 1",
            final_state="State 1",
            model_output="Output 1",
            steps_taken=3,
            success=True
        )

        traj2 = TrajectoryResult(
            query="Test",
            full_trajectory="Traj 2",
            final_state="State 2",
            model_output="Output 2",
            steps_taken=4,
            success=False
        )

        item = MemoryItem(
            title="Test",
            description="Test",
            content="Test"
        )

        result = MaTTSResult(
            query="Test query",
            best_trajectory=traj1,
            all_trajectories=[traj1, traj2],
            aggregated_memories=[item],
            entry_id="entry_001",
            scaling_mode="parallel",
            scaling_factor=3
        )

        assert result.query == "Test query"
        assert result.best_trajectory.success is True
        assert len(result.all_trajectories) == 2
        assert len(result.aggregated_memories) == 1
        assert result.scaling_mode == "parallel"
        assert result.scaling_factor == 3

    def test_matts_result_to_dict(self):
        """Test MaTTSResult serialization"""
        traj = TrajectoryResult(
            query="Test",
            full_trajectory="Traj",
            final_state="State",
            model_output="Output",
            steps_taken=3
        )

        result = MaTTSResult(
            query="Test query",
            best_trajectory=traj,
            all_trajectories=[traj],
            aggregated_memories=[],
            entry_id="entry_001",
            scaling_mode="parallel",
            scaling_factor=3
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["scaling_mode"] == "parallel"
        assert result_dict["scaling_factor"] == 3
        assert isinstance(result_dict["best_trajectory"], dict)
        assert isinstance(result_dict["all_trajectories"], list)


@pytest.mark.unit
class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_memory_item_json_roundtrip(self):
        """Test MemoryItem JSON serialization roundtrip"""
        original = MemoryItem(
            title="Test Title",
            description="Test description",
            content="Test content",
            source_task_id="task_123",
            success_signal=True,
            extraction_timestamp=time.time()
        )

        # Serialize to JSON
        json_str = json.dumps(original.to_dict())

        # Deserialize from JSON
        data = json.loads(json_str)
        restored = MemoryItem.from_dict(data)

        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.content == original.content
        assert restored.source_task_id == original.source_task_id
        assert restored.success_signal == original.success_signal

    def test_memory_entry_json_roundtrip(self):
        """Test MemoryEntry JSON serialization roundtrip"""
        item = MemoryItem(
            title="Test",
            description="Test",
            content="Test"
        )

        original = MemoryEntry(
            id="entry_001",
            task_query="Test query",
            trajectory="Test trajectory",
            success=True,
            memory_items=[item]
        )

        # Serialize to JSON
        json_str = json.dumps(original.to_dict())

        # Deserialize from JSON
        data = json.loads(json_str)
        restored = MemoryEntry.from_dict(data)

        assert restored.id == original.id
        assert restored.task_query == original.task_query
        assert len(restored.memory_items) == len(original.memory_items)
