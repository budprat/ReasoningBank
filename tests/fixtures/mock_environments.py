"""
ABOUTME: Mock environment for testing ReasoningBank without real web interactions
ABOUTME: Simulates agent actions, observations, and state transitions
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random


@dataclass
class MockState:
    """
    Represents the current state of the mock environment.

    Attributes:
        page: Current page name
        data: Dictionary of available data on current page
        history: List of actions taken
        step_count: Number of steps executed
    """
    page: str
    data: Dict[str, Any]
    history: List[str]
    step_count: int


class MockEnvironment:
    """
    Mock environment for testing ReasoningBank agents without real web navigation.

    Simulates web actions (click, search, navigate) and returns realistic observations.
    Configurable success/failure scenarios for testing different agent behaviors.

    Based on paper's WebArena-style environment but simplified for testing.
    """

    def __init__(self, scenario: str = "default", max_steps: int = 30):
        """
        Initialize mock environment with specific scenario.

        Args:
            scenario: Predefined scenario ("arithmetic", "search", "navigation", "default")
            max_steps: Maximum steps allowed (from paper: 30)
        """
        self.scenario = scenario
        self.max_steps = max_steps

        self.state = MockState(
            page="home",
            data={},
            history=[],
            step_count=0
        )

        self._load_scenario_data()

    def _load_scenario_data(self):
        """Load data based on selected scenario"""
        scenarios = {
            "arithmetic": {
                "home": {
                    "calculator": {"available": True, "type": "arithmetic"}
                }
            },
            "search": {
                "home": {
                    "search_box": {"available": True},
                    "results": []
                }
            },
            "navigation": {
                "home": {
                    "links": ["Products", "Cart", "Checkout", "About"]
                },
                "cart": {
                    "items": ["Item 1", "Item 2"],
                    "checkout_button": {"available": True}
                },
                "checkout": {
                    "form": {"billing": True, "shipping": True}
                }
            },
            "default": {
                "home": {
                    "search_box": {"available": True},
                    "calculator": {"available": True},
                    "links": ["Products", "About", "Contact"]
                }
            }
        }

        self.scenario_data = scenarios.get(self.scenario, scenarios["default"])
        self.state.data = self.scenario_data.get("home", {})

    def reset(self):
        """Reset environment to initial state"""
        self.state = MockState(
            page="home",
            data=self.scenario_data.get("home", {}),
            history=[],
            step_count=0
        )

    def execute_action(self, action: str) -> str:
        """
        Execute an action and return observation.

        Args:
            action: Action string in format "<action_type> <target>"

        Returns:
            str: Observation describing result of action

        Examples:
            >>> env.execute_action("calculate 25 * 4")
            "Result: 100"
            >>> env.execute_action("search 'To Kill a Mockingbird'")
            "Found: Harper Lee wrote 'To Kill a Mockingbird'"
        """
        self.state.step_count += 1
        self.state.history.append(action)

        # Check step limit
        if self.state.step_count > self.max_steps:
            return "Error: Maximum steps exceeded (30 steps limit)"

        # Parse action
        action_lower = action.lower().strip()

        # Handle different action types
        if action_lower.startswith("calculate"):
            return self._handle_calculate(action)
        elif action_lower.startswith("search"):
            return self._handle_search(action)
        elif action_lower.startswith("click"):
            return self._handle_click(action)
        elif action_lower.startswith("navigate"):
            return self._handle_navigate(action)
        elif action_lower.startswith("answer"):
            return self._handle_answer(action)
        else:
            return f"Unknown action type: {action}"

    def _handle_calculate(self, action: str) -> str:
        """Handle calculation actions"""
        # Extract expression
        expression = action.replace("calculate", "").strip()

        try:
            # Safe evaluation (only allow basic arithmetic)
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: Invalid calculation - {str(e)}"

    def _handle_search(self, action: str) -> str:
        """Handle search actions"""
        # Extract query
        query = action.replace("search", "").strip().strip("'\"")

        # Predefined search results
        search_database = {
            "to kill a mockingbird": "Found: Harper Lee wrote 'To Kill a Mockingbird', published in 1960",
            "to kill a mockingbird author": "Found: Harper Lee wrote 'To Kill a Mockingbird', published in 1960",
            "harper lee": "Found: Harper Lee (1926-2016) was an American author",
            "1984": "Error: Too many results - year 1984, books, movies, etc.",
            "1984 author": "Found: George Orwell wrote '1984', published in 1949",
            "george orwell": "Found: George Orwell (1903-1950) was an English author",
            "book 1984": "Still too many results, unable to determine author",
            "flights nyc lax": "Found 5 flights: $250, $320, $180, $410, $275"
        }

        query_lower = query.lower()
        if query_lower in search_database:
            return search_database[query_lower]
        elif len(query) < 3:
            return "Error: Search query too short"
        else:
            return "No results found"

    def _handle_click(self, action: str) -> str:
        """Handle click actions"""
        # Extract target
        target = action.replace("click", "").strip().strip("'\"")
        target_lower = target.lower()

        # Navigation based on current page
        navigation_map = {
            "home": {
                "products": ("products", "Page changed to products listing"),
                "cart": ("cart", "Page changed to shopping cart"),
                "view cart": ("cart", "Page changed to shopping cart with items listed"),
                "about": ("about", "Page changed to about page"),
                "about us": ("about", "Page changed to about page"),
            },
            "cart": {
                "checkout": ("checkout", "Page changed to checkout page with billing form"),
                "proceed to checkout": ("checkout", "Page changed to checkout page with billing information form"),
                "back": ("home", "Returned to home page"),
            },
            "products": {
                "cart": ("cart", "Page changed to shopping cart"),
                "home": ("home", "Returned to home page"),
            }
        }

        current_nav = navigation_map.get(self.state.page, {})

        for keyword, (new_page, observation) in current_nav.items():
            if keyword in target_lower:
                self.state.page = new_page
                self.state.data = self.scenario_data.get(new_page, {})
                return observation

        return f"Error: Could not find clickable element matching '{target}'"

    def _handle_navigate(self, action: str) -> str:
        """Handle navigation actions"""
        # Extract destination
        destination = action.replace("navigate", "").replace("to", "").strip()
        destination_lower = destination.lower()

        # Direct page navigation
        if destination_lower in self.scenario_data:
            self.state.page = destination_lower
            self.state.data = self.scenario_data[destination_lower]
            return f"Navigated to {destination_lower} page"
        else:
            return f"Error: Unknown page '{destination}'"

    def _handle_answer(self, action: str) -> str:
        """Handle answer actions (final responses)"""
        answer = action.replace("answer", "").strip().strip("'\"")
        return f"Agent answered: {answer}"

    def get_state(self) -> str:
        """
        Get current environment state description.

        Returns:
            str: Human-readable state description
        """
        state_desc = f"Current page: {self.state.page}\n"
        state_desc += f"Step count: {self.state.step_count}/{self.max_steps}\n"

        if self.state.data:
            state_desc += f"Available elements: {list(self.state.data.keys())}\n"

        return state_desc

    def is_task_complete(self, expected_outcome: str) -> bool:
        """
        Check if task is complete based on expected outcome.

        Args:
            expected_outcome: Expected final state or answer

        Returns:
            bool: True if task completed as expected
        """
        # Check final page
        if "page:" in expected_outcome.lower():
            target_page = expected_outcome.split("page:")[-1].strip()
            return self.state.page == target_page

        # Check action history
        if "answer" in expected_outcome.lower():
            last_action = self.state.history[-1] if self.state.history else ""
            return "answer" in last_action.lower()

        return False

    def get_trajectory(self) -> str:
        """
        Get complete trajectory as string.

        Returns:
            str: Full action history formatted as trajectory
        """
        trajectory = ""
        for i, action in enumerate(self.state.history, 1):
            trajectory += f"Step {i}: {action}\n"
        return trajectory


class ConfigurableMockEnvironment(MockEnvironment):
    """
    Mock environment with configurable responses for testing specific scenarios.

    Allows manual control over action outcomes for targeted testing.
    """

    def __init__(self, custom_responses: Optional[Dict[str, str]] = None, **kwargs):
        """
        Initialize with custom responses.

        Args:
            custom_responses: Dict mapping actions to custom observations
        """
        super().__init__(**kwargs)
        self.custom_responses = custom_responses or {}

    def execute_action(self, action: str) -> str:
        """Execute action with custom response if available"""
        # Check for custom response first
        if action in self.custom_responses:
            self.state.step_count += 1
            self.state.history.append(action)
            return self.custom_responses[action]

        # Fall back to standard behavior
        return super().execute_action(action)


def create_arithmetic_environment() -> MockEnvironment:
    """Create environment configured for arithmetic tasks"""
    return MockEnvironment(scenario="arithmetic")


def create_search_environment() -> MockEnvironment:
    """Create environment configured for search tasks"""
    return MockEnvironment(scenario="search")


def create_navigation_environment() -> MockEnvironment:
    """Create environment configured for navigation tasks"""
    return MockEnvironment(scenario="navigation")


def create_custom_environment(responses: Dict[str, str]) -> ConfigurableMockEnvironment:
    """
    Create environment with custom responses.

    Args:
        responses: Dict mapping actions to desired observations

    Returns:
        ConfigurableMockEnvironment instance
    """
    return ConfigurableMockEnvironment(custom_responses=responses)
