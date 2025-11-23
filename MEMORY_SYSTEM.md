# Hybrid Memory Management System

A comprehensive memory and context management system for AI agents that combines structured user profiles, conversation history, and menu tracking with intelligent context retrieval.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Hybrid Memory System                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  User Profiles  │  │  Conversation   │  │  Menu        │ │
│  │                 │  │  History        │  │  History     │ │
│  │ • Calories      │  │                 │  │              │ │
│  │ • Protein       │  │ • User msgs     │  │ • Dishes     │ │
│  │ • Allergies     │  │ • Agent msgs    │  │ • Proteins   │ │
│  │ • Equipment     │  │ • Timestamps    │  │ • Calories   │ │
│  │ • Preferences   │  │ • Sessions      │  │ • Dates      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           │                    │                    │        │
│           └────────────────────┴────────────────────┘        │
│                              │                               │
│                              ▼                               │
│                   ┌────────────────────┐                     │
│                   │  Context Builder   │                     │
│                   │                    │                     │
│                   │  Combines all      │                     │
│                   │  context sources   │                     │
│                   │  for LLM prompt    │                     │
│                   └────────────────────┘                     │
│                              │                               │
│                              ▼                               │
│                   ┌────────────────────┐                     │
│                   │   Agent + LLM      │                     │
│                   └────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Persistent User Profiles**
- Automatically extracts and stores user preferences from conversations
- Supports dietary restrictions, allergies, equipment, calorie targets, and more
- Handles ambiguous references like "my usual" by mapping to stored profile

### 2. **Conversation History**
- Tracks all user-agent interactions with timestamps
- Supports session-based organization
- Provides sliding window context to LLM (last N messages)
- Enables inter-session memory (remembering across multiple conversations)

### 3. **Menu History Tracking**
- Automatically extracts meal plans from agent responses
- Tracks dishes, proteins, ingredients, and macros
- Enforces variety rules (no repeats, protein rotation, ingredient diversity)
- Supports multi-day meal planning with date tracking

### 4. **Smart Context Building**
- Hierarchical memory: always-include profile + recent conversation + relevant history
- Optimized for token limits (only includes relevant context)
- Formatted prompts with clear sections for the LLM

### 5. **Database Persistence**
- SQLite database for reliable storage
- Survives application restarts
- Supports concurrent access
- Indexed for fast queries

## Components

### `memory_manager.py`
Core memory management system with:
- `UserProfile`: Structured user preferences dataclass
- `HybridMemoryManager`: Main memory manager class
- Database schema creation and management
- Context building for LLM prompts

### `menu_extractor.py`
Menu tracking utilities:
- `MenuExtractor`: Extracts structured meal data from responses
- Variety compliance checking
- Rotation policy enforcement
- Ingredient overlap detection

### `agent.py` (Updated)
Enhanced agent with memory integration:
- Accepts `user_context` parameter
- Formats context into prompts
- Automatically extracts profiles from conversations
- Tracks menu items from meal planning responses

### `discordBot.py` (Updated)
Discord integration with memory:
- Initializes shared `HybridMemoryManager`
- Passes memory manager to agent
- Persists all interactions automatically

## Usage

### Basic Setup

```python
from memory_manager import HybridMemoryManager
from agent import get_agent_message
import datetime

# Initialize memory manager (shared instance)
memory_manager = HybridMemoryManager(db_path="data/users.db")

# Get agent response with memory
response = get_agent_message(
    username="alice",
    inquiry="I'm cutting at 1800 calories, allergic to peanuts",
    timestamp=datetime.datetime.now(),
    memory_manager=memory_manager
)

# Later conversation - references "my usual"
response2 = get_agent_message(
    username="alice",
    inquiry="Give me a meal plan at my usual targets",
    timestamp=datetime.datetime.now(),
    memory_manager=memory_manager
)
# Agent will recall: 1800 calories, peanut allergy
```

### Working with User Profiles

```python
# Get user profile
profile = memory_manager.get_user_profile("alice")
print(profile.format_for_prompt())

# Manually update profile
memory_manager.update_user_profile("alice", {
    'calories_target': 1800,
    'protein_min': 140,
    'allergies': ['peanuts'],
    'equipment': ['stove', 'oven'],
    'dietary_restrictions': ['vegetarian']
})

# Profile merges intelligently (lists combine, values update)
memory_manager.update_user_profile("alice", {
    'fiber_min': 30,
    'allergies': ['shellfish']  # Adds to existing
})
# Result: allergies = ['peanuts', 'shellfish']
```

### Accessing Conversation History

```python
# Get recent messages
messages = memory_manager.get_recent_messages("alice", limit=10)
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")

# Get messages from specific session
session_msgs = memory_manager.get_session_messages("alice", "session_001")

# Format for display
formatted = memory_manager.format_conversation_history(messages)
```

### Menu History and Variety Tracking

```python
# Add menu item manually
memory_manager.add_menu_item(
    user_id="alice",
    dish_name="Grilled Chicken Salad",
    date_served=datetime.datetime.now(),
    meal_type="lunch",
    primary_protein="chicken",
    ingredients=["chicken breast", "lettuce", "tomatoes", "olive oil"],
    calories=450,
    protein=35,
    fiber=5
)

# Get recent menu history
menu_items = memory_manager.get_menu_history("alice", days=7)

# Check variety compliance
from menu_extractor import MenuExtractor
extractor = MenuExtractor(memory_manager)

proposed_meals = [
    {
        'dish_name': 'Grilled Chicken Salad',
        'primary_protein': 'chicken',
        'ingredients': ['chicken', 'lettuce']
    }
]

compliance = extractor.check_variety_compliance(
    user_id="alice",
    proposed_meals=proposed_meals,
    rotation_policy={
        'no_repeat_days': 7,
        'max_same_primary_protein_per_week': 2
    }
)

if not compliance['compliant']:
    print("Violations:", compliance['violations'])
```

### Building Context for LLM

```python
# Build complete context
context = memory_manager.build_context_for_llm(
    user_id="alice",
    current_query="Give me a dinner idea",
    include_menu_days=7,
    include_message_limit=10
)

# Access components
print("Profile:", context['profile_text'])
print("Recent conversation:", context['recent_conversation'])
print("Menu history:", context['menu_history'])
print("Current query:", context['current_query'])

# Format as full prompt
full_prompt = memory_manager.format_full_context_prompt(context)
```

## Database Schema

### `user_profiles`
```sql
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### `conversation_history`
```sql
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
)
```

### `menu_history`
```sql
CREATE TABLE menu_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT,
    meal_type TEXT,
    dish_name TEXT NOT NULL,
    primary_protein TEXT,
    ingredients TEXT,
    calories REAL,
    protein REAL,
    fiber REAL,
    date_served DATE NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
)
```

## How It Solves Benchmark Requirements

### ✅ Intra-session Memory
- Recent conversation history (last 10 turns) included in every prompt
- Conversation flows naturally with context awareness

### ✅ Inter-session Memory
- User profiles persist across sessions
- "My usual" maps to stored calories/protein/restrictions
- Menu history prevents repeats across sessions

### ✅ User Separation
- Each user has unique `user_id`
- Database ensures complete isolation
- No cross-user data leakage

### ✅ Variety Rules
- `no_repeat_days`: Checks menu history for exact dish matches
- `max_same_primary_protein_per_week`: Counts protein occurrences
- `ingredient_jaccard_max`: Computes ingredient overlap

### ✅ Profile Extraction
- LLM automatically extracts preferences from natural conversation
- Supports incremental updates (new info merges with existing)
- Handles ambiguous terms ("my usual", "like before")

### ✅ Tool Usage Tracking
- All tool calls logged in conversation metadata
- Can be queried for evaluation

## Testing

Run the test suite:

```bash
python test_memory_system.py
```

Tests cover:
1. Basic memory persistence
2. Multi-user separation
3. Menu history and variety tracking
4. Context building

## Integration with Discord Bot

The Discord bot automatically uses the memory system:

```python
# discordBot.py
memory_manager = HybridMemoryManager(db_path="data/users.db")

@bot.event
async def on_message(message: discord.Message):
    # ...
    reply_text = get_agent_message(
        sender.name,
        content,
        time_sent,
        memory_manager=memory_manager  # Automatic memory!
    )
```

Every Discord interaction:
1. Loads user profile + history
2. Generates contextualized response
3. Saves conversation
4. Extracts/updates profile
5. Tracks menu items

## Performance Considerations

### Context Window Management
- Default: last 10 messages (~2-3K tokens)
- Adjustable via `include_message_limit` parameter
- Profile is compact (~200 tokens)
- Menu history: ~50 tokens per week

### Database Optimization
- Indexes on `user_id` and `timestamp` columns
- Connection pooling for concurrent requests
- SQLite is fast enough for 1000s of users

### Memory Footprint
- One shared `HybridMemoryManager` instance
- Lazy database connections
- No in-memory caching (reads from DB each time)

## Future Enhancements

1. **Vector Store Integration** (commented out in design)
   - Semantic search over past conversations
   - Retrieve relevant context beyond recent window
   - Use ChromaDB, Pinecone, or FAISS

2. **Session Management**
   - Explicit session creation/closing
   - Session summaries for long conversations
   - Session-based context switching

3. **Advanced Profile Extraction**
   - Named entity recognition for ingredients
   - Confidence scores for extracted preferences
   - Conflict resolution (user changes mind)

4. **Analytics**
   - User engagement metrics
   - Profile completeness scores
   - Variety compliance trends

5. **Export/Import**
   - Export user data (GDPR compliance)
   - Import from other systems
   - Backup/restore utilities

## Troubleshooting

### Profile not updating
- Check logs for extraction errors
- Verify LLM response is valid JSON
- Test with explicit profile updates

### Menu items not tracked
- Ensure response contains meal details
- Check extraction prompt in `menu_extractor.py`
- Verify LLM can parse response format

### Context too large
- Reduce `include_message_limit`
- Reduce `include_menu_days`
- Consider summarization

### Database locked
- Ensure `close()` is called
- Check for long-running transactions
- Consider connection pooling

## Contributing

When adding features:
1. Update database schema in `_ensure_db_exists()`
2. Add migration logic for existing databases
3. Update tests in `test_memory_system.py`
4. Document new functionality here

## License

Same as parent project.
