# ReasoningBank Architecture

## System Overview

ReasoningBank is a self-evolving AI agent framework that learns from experience through a closed-loop memory system. The architecture enables agents to extract reasoning strategies from both successful and failed attempts, building a persistent knowledge base that improves performance over time.

## Core Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ReasoningBank System                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                            │
│  │   User      │                                                            │
│  │   Query     │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │                ReasoningBankAgent                       │                │
│  │                                                         │                │
│  │  ┌───────────────────────────────────────────────┐    │                │
│  │  │          Closed-Loop Learning Cycle           │    │                │
│  │  │                                               │    │                │
│  │  │   1. RETRIEVE ──► 2. ACT ──► 3. JUDGE       │    │                │
│  │  │        ▲                           │         │    │                │
│  │  │        │                           ▼         │    │                │
│  │  │   5. CONSOLIDATE ◄── 4. EXTRACT             │    │                │
│  │  │                                               │    │                │
│  │  └───────────────────────────────────────────────┘    │                │
│  └─────────────────────────────────────────────────────────┘                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Core Components                               │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │  Retriever   │  │    Judge     │  │  Extractor   │             │   │
│  │  │              │  │              │  │              │             │   │
│  │  │ • Embedding  │  │ • Success/   │  │ • Dual-      │             │   │
│  │  │   Search     │  │   Failure    │  │   Prompt     │             │   │
│  │  │ • Similarity │  │   Detection  │  │ • Strategy   │             │   │
│  │  │   Ranking    │  │ • Binary     │  │   Mining     │             │   │
│  │  │ • Top-k      │  │   Signal     │  │ • Lesson     │             │   │
│  │  │   Selection  │  │              │  │   Extraction │             │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │   │
│  │         │                  │                  │                      │   │
│  │         ▼                  ▼                  ▼                      │   │
│  │  ┌───────────────────────────────────────────────────────┐         │   │
│  │  │                   Consolidator                        │         │   │
│  │  │                                                       │         │   │
│  │  │  • Memory Bank Management                            │         │   │
│  │  │  • Simple Addition (No Deduplication)                │         │   │
│  │  │  • Persistent Storage (JSON)                         │         │   │
│  │  │  • Import/Export Support                             │         │   │
│  │  └───────────────────────────────────────────────────────┘         │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │         ┌─────────────────────────────────────┐                    │   │
│  │         │        Memory Bank Storage          │                    │   │
│  │         │                                     │                    │   │
│  │         │  📁 memory_bank.json               │                    │   │
│  │         │  • MemoryEntry objects             │                    │   │
│  │         │  • MemoryItem collections          │                    │   │
│  │         │  • Embeddings cache                │                    │   │
│  │         └─────────────────────────────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Flow

### 1. Core Agent Execution Flow

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  1. RETRIEVE (Retriever)        │
│                                 │
│  Input: query, memory_bank      │
│  Process:                       │
│  • Generate query embedding     │
│  • Compute cosine similarity    │
│  • Rank all memories            │
│  • Return top-k relevant        │
│  Output: List[MemoryItem]       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. ACT (Agent Executor)        │
│                                 │
│  Input: query, memories         │
│  Process:                       │
│  • Augment prompt with memories │
│  • ReAct loop execution         │
│  • Environment interaction      │
│  • Track trajectory steps       │
│  Output: trajectory, state      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. JUDGE (Judge)               │
│                                 │
│  Input: query, trajectory       │
│  Process:                       │
│  • Analyze final state          │
│  • Determine success/failure    │
│  • Binary classification        │
│  Output: bool (success)         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. EXTRACT (Extractor)         │
│                                 │
│  Input: trajectory, success     │
│  Process:                       │
│  • Dual-prompt extraction       │
│  • Success → strategies         │
│  • Failure → lessons            │
│  • Structure as MemoryItems     │
│  Output: List[MemoryItem]       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  5. CONSOLIDATE (Consolidator)  │
│                                 │
│  Input: memory_items            │
│  Process:                       │
│  • Create MemoryEntry           │
│  • Simple addition to bank      │
│  • No deduplication             │
│  • Save to JSON storage         │
│  Output: entry_id               │
└─────────────────────────────────┘
```

### 2. MaTTS (Memory-Aware Test-Time Scaling) Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MaTTS Scaling Strategies                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MaTTS PARALLEL (Breadth)                   │    │
│  │                                                         │    │
│  │     Query ──┬──► Agent 1 ──► Trajectory 1             │    │
│  │             ├──► Agent 2 ──► Trajectory 2             │    │
│  │             └──► Agent 3 ──► Trajectory 3             │    │
│  │                      │                                 │    │
│  │                      ▼                                 │    │
│  │              Memory Aggregation                        │    │
│  │                      │                                 │    │
│  │                      ▼                                 │    │
│  │              Select Best Result                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             MaTTS SEQUENTIAL (Depth)                    │    │
│  │                                                         │    │
│  │     Query ──► Agent 1 ──► Memory 1                     │    │
│  │                  │            │                         │    │
│  │                  ▼            ▼                         │    │
│  │              Agent 2 ──► Memory 1+2                    │    │
│  │                  │            │                         │    │
│  │                  ▼            ▼                         │    │
│  │              Agent 3 ──► Memory 1+2+3                  │    │
│  │                  │                                      │    │
│  │                  ▼                                      │    │
│  │          Progressive Refinement                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Model Hierarchy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TrajectoryResult                                              │
│  ├── query: str                                                │
│  ├── full_trajectory: str                                      │
│  ├── success: bool                                             │
│  └── memory_items: List[MemoryItem]                            │
│                          │                                      │
│                          ▼                                      │
│  MemoryEntry (Stored in Memory Bank)                           │
│  ├── id: str (UUID)                                           │
│  ├── task_query: str                                          │
│  ├── trajectory: str                                          │
│  ├── success: bool                                            │
│  ├── timestamp: float                                         │
│  └── memory_items: List[MemoryItem]                           │
│                          │                                      │
│                          ▼                                      │
│  MemoryItem (Reusable Knowledge Unit)                          │
│  ├── title: str                                               │
│  ├── description: str                                         │
│  ├── content: str                                             │
│  ├── source_task_id: str                                      │
│  └── success_signal: bool                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interactions

### LLM Provider Integration

```
┌────────────────────────────────────────┐
│         LLM Provider Layer             │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────┐│
│  │Anthropic │  │  OpenAI  │  │Google││
│  │  Claude  │  │   GPT-4  │  │Gemini││
│  └────┬─────┘  └────┬─────┘  └──┬───┘│
│       │              │            │    │
│       └──────────┬───────────────┘    │
│                  ▼                     │
│         ┌────────────────┐             │
│         │  Agent Core    │             │
│         │  Judge         │             │
│         │  Extractor     │             │
│         └────────────────┘             │
└────────────────────────────────────────┘
```

### Embedding System Architecture

```
┌─────────────────────────────────────────────┐
│           Embedding Pipeline                │
├─────────────────────────────────────────────┤
│                                             │
│  Text Input                                 │
│      │                                      │
│      ▼                                      │
│  ┌─────────────────┐                       │
│  │ Embedding Cache │──► Cached? ──► Return │
│  └─────────────────┘                       │
│      │ Not Cached                          │
│      ▼                                      │
│  ┌─────────────────────────┐               │
│  │   Embedding Provider    │               │
│  │ • OpenAI (1536 dim)     │               │
│  │ • Google (768 dim)      │               │
│  └─────────────────────────┘               │
│      │                                      │
│      ▼                                      │
│  Store in Cache                             │
│      │                                      │
│      ▼                                      │
│  Return Embedding Vector                    │
│                                             │
└─────────────────────────────────────────────┘
```

## Memory Growth Strategy (Gap 22)

```
┌──────────────────────────────────────────────┐
│        Memory Bank Growth Pattern            │
├──────────────────────────────────────────────┤
│                                              │
│  Task 1 ──► Memory 1                         │
│  Task 2 ──► Memory 1, 2                      │
│  Task 3 ──► Memory 1, 2, 3                   │
│  ...                                         │
│  Task N ──► Memory 1, 2, 3, ..., N           │
│                                              │
│  Properties:                                 │
│  • Linear growth (no deduplication)          │
│  • Unbounded accumulation                    │
│  • Simple addition strategy                  │
│  • Performance target: <5s retrieval @ 1000+ │
│                                              │
└──────────────────────────────────────────────┘
```

## Testing Architecture

```
┌──────────────────────────────────────────────────┐
│              Testing Framework                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  Unit Tests                                      │
│  ├── test_config.py                             │
│  ├── test_models.py                             │
│  ├── test_judge.py                              │
│  ├── test_extractor.py                          │
│  ├── test_retriever.py                          │
│  └── test_consolidator.py                       │
│                                                  │
│  Integration Tests (E2E)                        │
│  ├── test_streaming_constraint.py (Gap 21)      │
│  ├── test_progressive_learning.py (Gap 1)       │
│  └── test_context_dependent.py (Gap 2)          │
│                                                  │
│  Ablation Tests                                 │
│  └── test_success_and_failure_extraction.py     │
│      (Gap 24 - Core innovation validation)      │
│                                                  │
│  Stress Tests                                   │
│  └── test_memory_growth_long_term.py (Gap 22)   │
│      • 10-task quick validation                 │
│      • 100-task growth test                     │
│      • 334-task performance test                │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Configuration System

```
┌─────────────────────────────────────────────────┐
│         Configuration Architecture              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ReasoningBankConfig                           │
│  ├── LLM Settings                              │
│  │   ├── provider: anthropic|openai|google     │
│  │   ├── model: claude-3|gpt-4|gemini         │
│  │   └── temperatures: agent|judge|extractor   │
│  │                                             │
│  ├── Memory Settings                           │
│  │   ├── memory_bank_path: str                │
│  │   ├── embedding_model: str                 │
│  │   ├── top_k_retrieval: int                 │
│  │   └── extract_from_failures: bool          │
│  │                                             │
│  └── Execution Settings                        │
│      ├── max_steps_per_task: int              │
│      └── enable_memory_injection: bool        │
│                                                 │
│  Preset Configurations                         │
│  ├── get_config_for_claude()                  │
│  ├── get_config_for_paper_replication()       │
│  ├── get_config_for_matts_parallel()          │
│  └── get_config_for_matts_sequential()        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Key Architectural Decisions

1. **Closed-Loop Learning**: Every task execution feeds back into the memory bank
2. **Simple Addition Strategy**: No deduplication or pruning (intentional design)
3. **Dual-Prompt Extraction**: Different prompts for success vs failure extraction
4. **Embedding-Based Retrieval**: Semantic similarity for memory selection
5. **JSON Persistence**: Human-readable, portable memory storage
6. **Provider Agnostic**: Supports multiple LLM providers transparently
7. **Test-Time Scaling**: Two strategies (parallel breadth, sequential depth)

## Performance Characteristics

- **Memory Retrieval**: O(n) complexity, <5s for 1000+ memories
- **Embedding Generation**: Cached to minimize API calls
- **Memory Growth**: Linear with task count
- **Storage Format**: JSON (human-readable, ~1-5KB per memory)
- **API Efficiency**: Batch operations where supported

## Future Architecture Considerations

1. **Vector Database Integration**: For scaling beyond 10,000 memories
2. **Memory Pruning Strategies**: Optional deduplication/compression
3. **Distributed Execution**: Multi-agent coordination
4. **Real-time Learning**: Stream processing for continuous learning
5. **Memory Versioning**: Track memory evolution over time