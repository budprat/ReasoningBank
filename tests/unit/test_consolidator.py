"""
ABOUTME: Unit tests for MemoryConsolidator component
ABOUTME: Tests persistent storage, querying, and management of memory bank
"""

import pytest
import os
import json
import tempfile
import time
from reasoningbank.consolidator import MemoryConsolidator, create_consolidator
from reasoningbank.models import MemoryEntry, MemoryItem, TrajectoryResult
from reasoningbank.config import ReasoningBankConfig


class TestMemoryConsolidatorInitialization:
    """Tests for MemoryConsolidator initialization"""

    def test_consolidator_initialization(self, test_config):
        """Test MemoryConsolidator initialization"""
        consolidator = MemoryConsolidator(test_config)

        assert consolidator.config == test_config
        assert consolidator.bank_path == test_config.memory_bank_path
        assert isinstance(consolidator.memory_bank, list)

    def test_consolidator_creates_data_directory(self):
        """Test that consolidator creates data directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = os.path.join(tmpdir, "nested", "dir", "memory.json")

            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=bank_path,
                enable_logging=False
            )

            consolidator = MemoryConsolidator(config)

            # Directory should have been created
            assert os.path.exists(os.path.dirname(bank_path))

    def test_consolidator_loads_existing_bank_on_init(self):
        """Test that consolidator automatically loads existing memory bank"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = os.path.join(tmpdir, "memory.json")

            # Create existing memory bank file
            existing_data = [{
                "id": "entry_001",
                "task_query": "Test query",
                "trajectory": "Test trajectory",
                "success": True,
                "memory_items": [{
                    "title": "Test",
                    "description": "Test",
                    "content": "Test",
                    "source_task_id": None,
                    "success_signal": True,
                    "extraction_timestamp": None
                }],
                "timestamp": time.time()
            }]

            os.makedirs(os.path.dirname(bank_path), exist_ok=True)
            with open(bank_path, 'w') as f:
                json.dump(existing_data, f)

            # Initialize consolidator
            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=bank_path,
                enable_logging=False
            )
            consolidator = MemoryConsolidator(config)

            # Should have loaded existing entry
            assert len(consolidator.memory_bank) == 1
            assert consolidator.memory_bank[0].id == "entry_001"


class TestLoadSaveOperations:
    """Tests for load/save functionality"""

    def test_save_creates_json_file(self, test_config):
        """Test that save() creates JSON file"""
        consolidator = MemoryConsolidator(test_config)

        # Add an entry
        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.memory_bank.append(MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[item]
        ))

        # Save
        consolidator.save()

        # Verify file exists
        assert os.path.exists(test_config.memory_bank_path)

    def test_load_reads_json_file(self, test_config):
        """Test that load() reads JSON file correctly"""
        consolidator = MemoryConsolidator(test_config)

        # Create and save entry
        item = MemoryItem(title="LoadTest", description="Test", content="Test")
        entry = MemoryEntry(
            id="entry_load",
            task_query="Load test",
            trajectory="Test",
            success=True,
            memory_items=[item]
        )
        consolidator.memory_bank = [entry]
        consolidator.save()

        # Create new consolidator (should load from disk)
        consolidator2 = MemoryConsolidator(test_config)

        assert len(consolidator2.memory_bank) == 1
        assert consolidator2.memory_bank[0].id == "entry_load"
        assert consolidator2.memory_bank[0].memory_items[0].title == "LoadTest"

    def test_load_with_missing_file(self, test_config):
        """Test load() with non-existent file initializes empty bank"""
        # Ensure file doesn't exist
        if os.path.exists(test_config.memory_bank_path):
            os.remove(test_config.memory_bank_path)

        consolidator = MemoryConsolidator(test_config)

        # Should have empty memory bank
        assert consolidator.memory_bank == []

    def test_load_with_corrupted_file(self):
        """Test load() handles corrupted JSON gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = os.path.join(tmpdir, "memory.json")

            # Create corrupted JSON file
            os.makedirs(os.path.dirname(bank_path), exist_ok=True)
            with open(bank_path, 'w') as f:
                f.write("{corrupted json [[[")

            config = ReasoningBankConfig(
                llm_api_key="test-key",
                memory_bank_path=bank_path,
                enable_logging=False
            )

            # Should initialize with empty bank (no error)
            consolidator = MemoryConsolidator(config)

            assert consolidator.memory_bank == []

    def test_save_formats_json_with_indent(self, test_config):
        """Test that save() formats JSON with indentation"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.memory_bank.append(MemoryEntry(
            id="entry_001",
            task_query="Test",
            trajectory="Test",
            success=True,
            memory_items=[item]
        ))

        consolidator.save()

        # Read file and check formatting
        with open(test_config.memory_bank_path, 'r') as f:
            content = f.read()

        # Should have indentation (pretty-printed)
        assert "  " in content or "\n" in content


class TestAddingEntries:
    """Tests for adding entries to memory bank"""

    def test_add_entry_with_trajectory_result(self, test_config):
        """Test add_entry() with TrajectoryResult"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        trajectory_result = TrajectoryResult(
            query="Test query",
            full_trajectory="Test trajectory",
            final_state="Final state",
            model_output="Output",
            steps_taken=5,
            success=True,
            memory_items=[item]
        )

        entry_id = consolidator.add_entry(trajectory_result, [item])

        # Should have added entry
        assert len(consolidator.memory_bank) == 1
        assert entry_id is not None
        assert consolidator.memory_bank[0].id == entry_id

    def test_add_from_trajectory_components(self, test_config):
        """Test add_from_trajectory() with direct components"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")

        entry_id = consolidator.add_from_trajectory(
            query="Test query",
            trajectory="Test trajectory",
            final_state="Final state",
            model_output="Output",
            success=True,
            memory_items=[item],
            steps_taken=3
        )

        # Should have added entry
        assert len(consolidator.memory_bank) == 1
        assert consolidator.memory_bank[0].id == entry_id
        assert consolidator.memory_bank[0].task_query == "Test query"

    def test_add_entry_generates_unique_id(self, test_config):
        """Test that each added entry gets a unique ID"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")

        id1 = consolidator.add_from_trajectory(
            query="Query 1",
            trajectory="Traj 1",
            final_state="State 1",
            model_output="Output 1",
            success=True,
            memory_items=[item]
        )

        id2 = consolidator.add_from_trajectory(
            query="Query 2",
            trajectory="Traj 2",
            final_state="State 2",
            model_output="Output 2",
            success=True,
            memory_items=[item]
        )

        # IDs should be different
        assert id1 != id2

    def test_add_entry_saves_automatically(self, test_config):
        """Test that add_entry() automatically saves to disk"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test",
            success=True,
            memory_items=[item]
        )

        # File should exist after add
        assert os.path.exists(test_config.memory_bank_path)

        # Should be able to load in new consolidator
        consolidator2 = MemoryConsolidator(test_config)
        assert len(consolidator2.memory_bank) == 1


class TestGettingEntries:
    """Tests for getting entries from memory bank"""

    def test_get_entry_by_id(self, test_config):
        """Test get_entry() retrieves entry by ID"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        entry_id = consolidator.add_from_trajectory(
            query="Test",
            trajectory="Test",
            final_state="Test",
            model_output="Test",
            success=True,
            memory_items=[item]
        )

        # Retrieve by ID
        retrieved = consolidator.get_entry(entry_id)

        assert retrieved is not None
        assert retrieved.id == entry_id

    def test_get_entry_returns_none_for_missing(self, test_config):
        """Test get_entry() returns None for non-existent ID"""
        consolidator = MemoryConsolidator(test_config)

        retrieved = consolidator.get_entry("nonexistent_id")

        assert retrieved is None

    def test_get_all_entries(self, test_config):
        """Test get_all_entries() returns all entries"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item])
        consolidator.add_from_trajectory("Q3", "T3", "S3", "O3", False, [item])

        all_entries = consolidator.get_all_entries()

        assert len(all_entries) == 3

    def test_get_success_entries(self, test_config):
        """Test get_success_entries() filters successful entries"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item])
        consolidator.add_from_trajectory("Q3", "T3", "S3", "O3", False, [item])

        success_entries = consolidator.get_success_entries()

        assert len(success_entries) == 2
        assert all(entry.success for entry in success_entries)

    def test_get_failure_entries(self, test_config):
        """Test get_failure_entries() filters failed entries"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", False, [item])
        consolidator.add_from_trajectory("Q3", "T3", "S3", "O3", False, [item])

        failure_entries = consolidator.get_failure_entries()

        assert len(failure_entries) == 2
        assert all(not entry.success for entry in failure_entries)


class TestSearchFunctionality:
    """Tests for searching entries"""

    def test_search_by_query_substring(self, test_config):
        """Test search_entries() with query substring"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Calculate 25 * 4", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Navigate to checkout", "T2", "S2", "O2", True, [item])
        consolidator.add_from_trajectory("Calculate average", "T3", "S3", "O3", True, [item])

        results = consolidator.search_entries(query_substring="calculate")

        assert len(results) == 2
        assert all("calculate" in entry.task_query.lower() for entry in results)

    def test_search_by_success(self, test_config):
        """Test search_entries() with success filter"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", False, [item])
        consolidator.add_from_trajectory("Q3", "T3", "S3", "O3", True, [item])

        # Search for successful only
        results = consolidator.search_entries(success=True)
        assert len(results) == 2
        assert all(entry.success for entry in results)

        # Search for failures only
        results = consolidator.search_entries(success=False)
        assert len(results) == 1
        assert all(not entry.success for entry in results)

    def test_search_by_timestamp_range(self, test_config):
        """Test search_entries() with timestamp filters"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")

        # Add entries with different timestamps
        now = time.time()
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        time.sleep(0.01)  # Small delay
        mid_time = time.time()
        time.sleep(0.01)
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item])

        # Search for entries after mid_time
        results = consolidator.search_entries(min_timestamp=mid_time)
        assert len(results) == 1

        # Search for entries before mid_time
        results = consolidator.search_entries(max_timestamp=mid_time)
        assert len(results) >= 1

    def test_search_with_combined_filters(self, test_config):
        """Test search_entries() with multiple filters"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Calculate 25", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Calculate 50", "T2", "S2", "O2", False, [item])
        consolidator.add_from_trajectory("Navigate home", "T3", "S3", "O3", True, [item])

        # Search for successful calculations
        results = consolidator.search_entries(
            query_substring="calculate",
            success=True
        )

        assert len(results) == 1
        assert "calculate" in results[0].task_query.lower()
        assert results[0].success is True


class TestStatistics:
    """Tests for memory bank statistics"""

    def test_get_statistics_with_entries(self, test_config):
        """Test get_statistics() calculates correctly"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item], steps_taken=5)
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item, item], steps_taken=3)
        consolidator.add_from_trajectory("Q3", "T3", "S3", "O3", False, [item], steps_taken=7)

        stats = consolidator.get_statistics()

        assert stats["total_entries"] == 3
        assert stats["successful_entries"] == 2
        assert stats["failed_entries"] == 1
        assert stats["success_rate"] == pytest.approx(2/3)
        assert stats["total_memory_items"] == 4  # 1 + 2 + 1
        assert stats["avg_items_per_entry"] == pytest.approx(4/3)
        assert stats["avg_steps_per_task"] == pytest.approx((5 + 3 + 7) / 3)

    def test_get_statistics_empty_bank(self, test_config):
        """Test get_statistics() with empty memory bank"""
        consolidator = MemoryConsolidator(test_config)

        stats = consolidator.get_statistics()

        assert stats["total_entries"] == 0
        assert stats["successful_entries"] == 0
        assert stats["failed_entries"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["total_memory_items"] == 0
        assert stats["avg_items_per_entry"] == 0.0

    def test_statistics_includes_timestamps(self, test_config):
        """Test that statistics include oldest/newest timestamps"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        time.sleep(0.01)
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item])

        stats = consolidator.get_statistics()

        assert "oldest_entry" in stats
        assert "newest_entry" in stats
        assert stats["oldest_entry"] < stats["newest_entry"]


class TestManagementOperations:
    """Tests for memory bank management operations"""

    def test_clear_removes_all_entries(self, test_config):
        """Test clear() removes all entries"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item])

        assert len(consolidator.memory_bank) == 2

        consolidator.clear()

        assert len(consolidator.memory_bank) == 0

    def test_remove_entry_by_id(self, test_config):
        """Test remove_entry() removes specific entry"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        id1 = consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])
        id2 = consolidator.add_from_trajectory("Q2", "T2", "S2", "O2", True, [item])

        # Remove first entry
        removed = consolidator.remove_entry(id1)

        assert removed is True
        assert len(consolidator.memory_bank) == 1
        assert consolidator.memory_bank[0].id == id2

    def test_remove_entry_returns_false_for_missing(self, test_config):
        """Test remove_entry() returns False for non-existent ID"""
        consolidator = MemoryConsolidator(test_config)

        removed = consolidator.remove_entry("nonexistent_id")

        assert removed is False

    def test_export_to_file(self, test_config):
        """Test export_to_file() creates export"""
        consolidator = MemoryConsolidator(test_config)

        item = MemoryItem(title="Test", description="Test", content="Test")
        consolidator.add_from_trajectory("Q1", "T1", "S1", "O1", True, [item])

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as export_file:
            export_path = export_file.name

        try:
            consolidator.export_to_file(export_path)

            # Verify export file exists and contains data
            assert os.path.exists(export_path)
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data) == 1
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

    def test_import_from_file_replace_mode(self, test_config):
        """Test import_from_file() with replace mode"""
        consolidator = MemoryConsolidator(test_config)

        # Add initial entry
        item = MemoryItem(title="Original", description="Test", content="Test")
        consolidator.add_from_trajectory("Original", "T1", "S1", "O1", True, [item])

        # Create import file
        import_item = MemoryItem(title="Imported", description="Test", content="Test")
        import_entry = MemoryEntry(
            id="import_001",
            task_query="Imported query",
            trajectory="Test",
            success=True,
            memory_items=[import_item]
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as import_file:
            import_path = import_file.name
            json.dump([import_entry.to_dict()], import_file)

        try:
            # Import (replace mode)
            consolidator.import_from_file(import_path, merge=False)

            # Should have replaced original with imported
            assert len(consolidator.memory_bank) == 1
            assert consolidator.memory_bank[0].task_query == "Imported query"
        finally:
            if os.path.exists(import_path):
                os.remove(import_path)

    def test_import_from_file_merge_mode(self, test_config):
        """Test import_from_file() with merge mode"""
        consolidator = MemoryConsolidator(test_config)

        # Add initial entry
        item = MemoryItem(title="Original", description="Test", content="Test")
        original_id = consolidator.add_from_trajectory("Original", "T1", "S1", "O1", True, [item])

        # Create import file with different ID
        import_item = MemoryItem(title="Imported", description="Test", content="Test")
        import_entry = MemoryEntry(
            id="import_001",
            task_query="Imported query",
            trajectory="Test",
            success=True,
            memory_items=[import_item]
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as import_file:
            import_path = import_file.name
            json.dump([import_entry.to_dict()], import_file)

        try:
            # Import (merge mode)
            consolidator.import_from_file(import_path, merge=True)

            # Should have both entries
            assert len(consolidator.memory_bank) == 2
        finally:
            if os.path.exists(import_path):
                os.remove(import_path)

    def test_import_avoids_duplicate_ids_in_merge(self, test_config):
        """Test import_from_file() avoids duplicate IDs when merging"""
        consolidator = MemoryConsolidator(test_config)

        # Add initial entry
        item = MemoryItem(title="Original", description="Test", content="Test")
        entry_id = consolidator.add_from_trajectory("Original", "T1", "S1", "O1", True, [item])

        # Create import file with SAME ID
        import_entry = consolidator.memory_bank[0].to_dict()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as import_file:
            import_path = import_file.name
            json.dump([import_entry], import_file)

        try:
            # Import (merge mode) - should not duplicate
            consolidator.import_from_file(import_path, merge=True)

            # Should still have only 1 entry (no duplicate)
            assert len(consolidator.memory_bank) == 1
        finally:
            if os.path.exists(import_path):
                os.remove(import_path)


class TestConvenienceFunction:
    """Tests for standalone create_consolidator function"""

    def test_create_consolidator_function(self, test_config):
        """Test create_consolidator() convenience function"""
        consolidator = create_consolidator(test_config)

        assert isinstance(consolidator, MemoryConsolidator)
        assert consolidator.config == test_config


@pytest.mark.unit
class TestConsolidatorIntegration:
    """Integration tests for consolidator component"""

    def test_full_consolidation_workflow(self, test_config):
        """Test complete consolidation workflow"""
        # Initialize consolidator
        consolidator = MemoryConsolidator(test_config)

        # Add multiple entries
        item1 = MemoryItem(title="Item1", description="Test", content="Test")
        item2 = MemoryItem(title="Item2", description="Test", content="Test")

        id1 = consolidator.add_from_trajectory(
            "Query 1", "Traj 1", "State 1", "Output 1", True, [item1], steps_taken=5
        )
        id2 = consolidator.add_from_trajectory(
            "Query 2", "Traj 2", "State 2", "Output 2", False, [item2], steps_taken=3
        )

        # Verify entries
        assert len(consolidator.memory_bank) == 2

        # Get statistics
        stats = consolidator.get_statistics()
        assert stats["total_entries"] == 2
        assert stats["success_rate"] == 0.5

        # Search entries
        results = consolidator.search_entries(success=True)
        assert len(results) == 1

        # Remove one entry
        consolidator.remove_entry(id1)
        assert len(consolidator.memory_bank) == 1

        # Verify persistence
        consolidator2 = MemoryConsolidator(test_config)
        assert len(consolidator2.memory_bank) == 1

    def test_cross_session_persistence(self, test_config):
        """Test that data persists across consolidator instances"""
        # Session 1: Add entries
        consolidator1 = MemoryConsolidator(test_config)
        item = MemoryItem(title="Persistent", description="Test", content="Test")
        entry_id = consolidator1.add_from_trajectory(
            "Persistent query", "Traj", "State", "Output", True, [item]
        )

        # Session 2: Load and verify
        consolidator2 = MemoryConsolidator(test_config)
        retrieved = consolidator2.get_entry(entry_id)

        assert retrieved is not None
        assert retrieved.task_query == "Persistent query"
        assert retrieved.memory_items[0].title == "Persistent"
