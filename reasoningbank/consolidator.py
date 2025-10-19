"""
ABOUTME: Memory consolidation implementation for persistent storage
ABOUTME: Handles loading, saving, and managing the memory bank with JSON persistence
"""

from typing import List, Optional, Dict, Any
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .config import ReasoningBankConfig
from .models import MemoryEntry, MemoryItem, TrajectoryResult


class MemoryConsolidator:
    """
    Persistent storage and management for ReasoningBank memory system.

    Handles:
    - Loading/saving memory bank from/to JSON
    - Adding new memory entries
    - Deduplication and quality control
    - Memory bank statistics and querying
    """

    def __init__(self, config: ReasoningBankConfig):
        """
        Initialize the memory consolidator with configuration.

        Args:
            config: ReasoningBankConfig with storage paths
        """
        self.config = config
        self.bank_path = config.memory_bank_path
        self.memory_bank: List[MemoryEntry] = []

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.bank_path), exist_ok=True)

        # Load existing memory bank
        self.load()

    def load(self) -> None:
        """Load memory bank from disk."""
        if os.path.exists(self.bank_path):
            try:
                with open(self.bank_path, 'r') as f:
                    data = json.load(f)

                # Convert dict data to MemoryEntry objects
                self.memory_bank = [
                    MemoryEntry.from_dict(entry_data)
                    for entry_data in data
                ]

                if self.config.enable_logging:
                    print(f"Loaded {len(self.memory_bank)} entries from memory bank")

            except Exception as e:
                if self.config.enable_logging:
                    print(f"Error loading memory bank: {e}")
                self.memory_bank = []
        else:
            self.memory_bank = []

    def save(self) -> None:
        """Save memory bank to disk."""
        try:
            # Convert MemoryEntry objects to dicts
            data = [entry.to_dict() for entry in self.memory_bank]

            # Write to file with pretty formatting
            with open(self.bank_path, 'w') as f:
                json.dump(data, f, indent=2)

            if self.config.enable_logging:
                print(f"Saved {len(self.memory_bank)} entries to memory bank")

        except Exception as e:
            if self.config.enable_logging:
                print(f"Error saving memory bank: {e}")
            raise

    def add_entry(
        self,
        trajectory_result: TrajectoryResult,
        memory_items: List[MemoryItem]
    ) -> str:
        """
        Add a new memory entry to the bank.

        Args:
            trajectory_result: Result from agent execution
            memory_items: Extracted memory items

        Returns:
            str: Entry ID
        """
        # Generate unique entry ID
        entry_id = str(uuid.uuid4())

        # Create memory entry
        entry = MemoryEntry(
            id=entry_id,
            task_query=trajectory_result.query,
            trajectory=trajectory_result.full_trajectory,
            success=trajectory_result.success,
            memory_items=memory_items,
            timestamp=datetime.now().timestamp(),
            final_state=trajectory_result.final_state,
            model_output=trajectory_result.model_output,
            steps_taken=trajectory_result.steps_taken
        )

        # Add to memory bank
        self.memory_bank.append(entry)

        # Save to disk
        self.save()

        return entry_id

    def add_from_trajectory(
        self,
        query: str,
        trajectory: str,
        final_state: str,
        model_output: str,
        success: bool,
        memory_items: List[MemoryItem],
        steps_taken: Optional[int] = None
    ) -> str:
        """
        Add entry directly from trajectory components.

        Args:
            query: Task query
            trajectory: Full execution trace
            final_state: Final environment state
            model_output: Final model output
            success: Success/failure signal
            memory_items: Extracted memory items
            steps_taken: Number of steps taken

        Returns:
            str: Entry ID
        """
        # Create trajectory result
        trajectory_result = TrajectoryResult(
            query=query,
            full_trajectory=trajectory,
            final_state=final_state,
            model_output=model_output,
            steps_taken=steps_taken or 0,
            success=success,
            memory_items=memory_items
        )

        return self.add_entry(trajectory_result, memory_items)

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get memory entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            Optional[MemoryEntry]: Entry if found, None otherwise
        """
        for entry in self.memory_bank:
            if entry.id == entry_id:
                return entry
        return None

    def get_all_entries(self) -> List[MemoryEntry]:
        """
        Get all memory entries.

        Returns:
            List[MemoryEntry]: All entries in memory bank
        """
        return self.memory_bank.copy()

    def get_success_entries(self) -> List[MemoryEntry]:
        """
        Get all successful entries.

        Returns:
            List[MemoryEntry]: Successful entries only
        """
        return [entry for entry in self.memory_bank if entry.success]

    def get_failure_entries(self) -> List[MemoryEntry]:
        """
        Get all failed entries.

        Returns:
            List[MemoryEntry]: Failed entries only
        """
        return [entry for entry in self.memory_bank if not entry.success]

    def search_entries(
        self,
        query_substring: Optional[str] = None,
        success: Optional[bool] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None
    ) -> List[MemoryEntry]:
        """
        Search entries with filters.

        Args:
            query_substring: Filter by task query substring
            success: Filter by success/failure
            min_timestamp: Minimum timestamp
            max_timestamp: Maximum timestamp

        Returns:
            List[MemoryEntry]: Filtered entries
        """
        results = self.memory_bank

        # Apply filters
        if query_substring is not None:
            results = [
                entry for entry in results
                if query_substring.lower() in entry.task_query.lower()
            ]

        if success is not None:
            results = [entry for entry in results if entry.success == success]

        if min_timestamp is not None:
            results = [entry for entry in results if entry.timestamp >= min_timestamp]

        if max_timestamp is not None:
            results = [entry for entry in results if entry.timestamp <= max_timestamp]

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory bank statistics.

        Returns:
            Dict[str, Any]: Statistics including counts, success rate, etc.
        """
        total_entries = len(self.memory_bank)

        if total_entries == 0:
            return {
                "total_entries": 0,
                "successful_entries": 0,
                "failed_entries": 0,
                "success_rate": 0.0,
                "total_memory_items": 0,
                "avg_items_per_entry": 0.0,
                "avg_steps_per_task": 0.0
            }

        successful = sum(1 for entry in self.memory_bank if entry.success)
        failed = total_entries - successful

        total_memory_items = sum(
            len(entry.memory_items) for entry in self.memory_bank
        )

        total_steps = sum(
            entry.steps_taken for entry in self.memory_bank
            if entry.steps_taken is not None
        )
        entries_with_steps = sum(
            1 for entry in self.memory_bank
            if entry.steps_taken is not None
        )

        return {
            "total_entries": total_entries,
            "successful_entries": successful,
            "failed_entries": failed,
            "success_rate": successful / total_entries,
            "total_memory_items": total_memory_items,
            "avg_items_per_entry": total_memory_items / total_entries,
            "avg_steps_per_task": total_steps / entries_with_steps if entries_with_steps > 0 else 0.0,
            "oldest_entry": min(entry.timestamp for entry in self.memory_bank),
            "newest_entry": max(entry.timestamp for entry in self.memory_bank)
        }

    def clear(self) -> None:
        """Clear all entries from memory bank."""
        self.memory_bank = []
        self.save()

    def remove_entry(self, entry_id: str) -> bool:
        """
        Remove an entry from memory bank.

        Args:
            entry_id: Entry identifier

        Returns:
            bool: True if removed, False if not found
        """
        original_length = len(self.memory_bank)
        self.memory_bank = [
            entry for entry in self.memory_bank
            if entry.id != entry_id
        ]

        if len(self.memory_bank) < original_length:
            self.save()
            return True
        return False

    def export_to_file(self, export_path: str) -> None:
        """
        Export memory bank to a different file.

        Args:
            export_path: Path to export file
        """
        data = [entry.to_dict() for entry in self.memory_bank]

        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_file(self, import_path: str, merge: bool = False) -> None:
        """
        Import memory bank from file.

        Args:
            import_path: Path to import file
            merge: If True, merge with existing entries. If False, replace.
        """
        with open(import_path, 'r') as f:
            data = json.load(f)

        imported_entries = [
            MemoryEntry.from_dict(entry_data)
            for entry_data in data
        ]

        if merge:
            # Merge with existing entries (avoid duplicates by ID)
            existing_ids = {entry.id for entry in self.memory_bank}
            new_entries = [
                entry for entry in imported_entries
                if entry.id not in existing_ids
            ]
            self.memory_bank.extend(new_entries)
        else:
            # Replace existing entries
            self.memory_bank = imported_entries

        self.save()


def create_consolidator(config: ReasoningBankConfig) -> MemoryConsolidator:
    """
    Convenience function to create and initialize a memory consolidator.

    Args:
        config: ReasoningBankConfig

    Returns:
        MemoryConsolidator: Initialized consolidator
    """
    return MemoryConsolidator(config)
