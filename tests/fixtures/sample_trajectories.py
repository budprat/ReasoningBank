"""
ABOUTME: Sample trajectory data for testing ReasoningBank components
ABOUTME: Provides realistic successful and failed agent trajectories
"""

from typing import Dict, Any, List


# ============= Successful Trajectories =============

ARITHMETIC_SUCCESS = {
    "query": "What is 25 * 4 + 15?",
    "trajectory": """<think>I need to solve this arithmetic problem. According to order of operations (PEMDAS), I should do multiplication before addition.</think>
<action>calculate 25 * 4</action>
<observation>Result: 100</observation>
<think>Good, 25 * 4 = 100. Now I need to add 15 to this result.</think>
<action>calculate 100 + 15</action>
<observation>Result: 115</observation>
<think>The final answer is 115. Let me verify: 25*4=100, 100+15=115. Correct.</think>
<action>answer "The answer is 115"</action>""",
    "final_state": "Calculation completed successfully. Result displayed: 115",
    "model_output": "The answer is 115",
    "success": True,
    "steps_taken": 3
}


INFORMATION_SEEKING_SUCCESS = {
    "query": "Who wrote the novel 'To Kill a Mockingbird'?",
    "trajectory": """<think>I need to find the author of 'To Kill a Mockingbird'. This is a famous novel, so I should search for it.</think>
<action>search "To Kill a Mockingbird author"</action>
<observation>Found: Harper Lee wrote "To Kill a Mockingbird", published in 1960</observation>
<think>Perfect! The information clearly states Harper Lee is the author.</think>
<action>answer "Harper Lee"</action>""",
    "final_state": "Search successful. Author information displayed.",
    "model_output": "Harper Lee wrote 'To Kill a Mockingbird'",
    "success": True,
    "steps_taken": 2
}


NAVIGATION_SUCCESS = {
    "query": "Navigate to the checkout page",
    "trajectory": """<think>I need to navigate to the checkout page. I should look for a cart or checkout button.</think>
<action>click button[aria-label="View Cart"]</action>
<observation>Page changed to shopping cart view with items listed</observation>
<think>Good, I'm now viewing the cart. I need to find the checkout button.</think>
<action>click button[text="Proceed to Checkout"]</action>
<observation>Page changed to checkout page with billing information form</observation>
<think>Successfully navigated to checkout page.</think>
<action>confirm "Reached checkout page"</action>""",
    "final_state": "Checkout page displayed with billing form visible",
    "model_output": "Successfully navigated to checkout page",
    "success": True,
    "steps_taken": 3
}


MULTI_STEP_REASONING_SUCCESS = {
    "query": "Find the cheapest flight from NYC to LAX on December 15th",
    "trajectory": """<think>I need to search for flights from NYC to LAX on December 15th and compare prices.</think>
<action>search flights from "NYC" to "LAX" on "2024-12-15"</action>
<observation>Found 5 flights: $250, $320, $180, $410, $275</observation>
<think>I have 5 flight options with different prices. I need to identify the cheapest one.</think>
<action>find minimum price in results</action>
<observation>Minimum price: $180</observation>
<think>The cheapest flight costs $180. Let me verify this is correct.</think>
<action>answer "The cheapest flight from NYC to LAX on December 15th costs $180"</action>""",
    "final_state": "Flight search completed with price comparison",
    "model_output": "The cheapest flight costs $180",
    "success": True,
    "steps_taken": 3
}


# ============= Failed Trajectories =============

ARITHMETIC_FAILURE = {
    "query": "What is 25 * 4 + 15?",
    "trajectory": """<think>I need to calculate this. Let me just add them all together.</think>
<action>calculate 25 + 4 + 15</action>
<observation>Result: 44</observation>
<think>That seems wrong. Let me try multiplying everything.</think>
<action>calculate 25 * 4 * 15</action>
<observation>Result: 1500</observation>
<think>I'm confused about the order of operations.</think>
<action>answer "I'm not sure, either 44 or 1500"</action>""",
    "final_state": "Incorrect calculation performed",
    "model_output": "I'm not sure, either 44 or 1500",
    "success": False,
    "steps_taken": 3
}


INFORMATION_SEEKING_FAILURE = {
    "query": "Find the author of the book '1984'",
    "trajectory": """<think>I'll search for the number 1984</think>
<action>search "1984"</action>
<observation>Error: Too many results - year 1984, books, movies, etc.</observation>
<think>That didn't work. Let me try a different search.</think>
<action>search "book 1984"</action>
<observation>Still too many results, unable to determine author</observation>
<think>I can't find the specific author information.</think>
<action>answer "Unable to find author information"</action>""",
    "final_state": "Search failed - query too broad",
    "model_output": "Unable to find author information",
    "success": False,
    "steps_taken": 3
}


NAVIGATION_FAILURE = {
    "query": "Navigate to the checkout page",
    "trajectory": """<think>I need to find the checkout. Let me click on the first link I see.</think>
<action>click link[text="Products"]</action>
<observation>Page changed to products listing page</observation>
<think>This is not the checkout. Let me click another link.</think>
<action>click link[text="About Us"]</action>
<observation>Page changed to about page</observation>
<think>Still not the checkout page. I'm lost.</think>
<action>answer "Unable to find checkout page"</action>""",
    "final_state": "User on wrong page (About Us), not checkout",
    "model_output": "Unable to find checkout page",
    "success": False,
    "steps_taken": 3
}


MISSING_ERROR_HANDLING_FAILURE = {
    "query": "Get the price of product ID 12345",
    "trajectory": """<think>I'll query the product database for ID 12345</think>
<action>query database for product_id=12345</action>
<observation>Error: Product ID 12345 not found in database</observation>
<think>Got an error. I'll just report the error.</think>
<action>answer "Error: Product not found"</action>""",
    "final_state": "Error encountered, no recovery attempted",
    "model_output": "Error: Product not found",
    "success": False,
    "steps_taken": 2
}


# ============= Ambiguous/Edge Case Trajectories =============

PARTIAL_SUCCESS = {
    "query": "Find the top 3 highest rated restaurants in NYC",
    "trajectory": """<think>I need to search for highly rated restaurants in NYC</think>
<action>search "top rated restaurants NYC"</action>
<observation>Found: 1. Le Bernardin (4.8 stars), 2. Eleven Madison Park (4.7 stars)</observation>
<think>I found 2 restaurants but need 3. Let me continue searching.</think>
<action>answer "Found 2 restaurants: Le Bernardin and Eleven Madison Park"</action>""",
    "final_state": "Only 2 of 3 requested restaurants found",
    "model_output": "Found 2 restaurants: Le Bernardin and Eleven Madison Park",
    "success": False,  # Incomplete result
    "steps_taken": 2
}


OVERCOMPLEX_SUCCESS = {
    "query": "What is 10 + 5?",
    "trajectory": """<think>This is an addition problem. Let me be very thorough.</think>
<action>verify both numbers are valid: 10, 5</action>
<observation>Both numbers are valid integers</observation>
<think>Good. Now let me check if addition is the correct operation.</think>
<action>confirm operation is addition (+)</action>
<observation>Operation confirmed: addition</observation>
<think>Excellent. Now let me perform the calculation step by step.</think>
<action>calculate 10 + 5</action>
<observation>Result: 15</observation>
<think>Let me verify this result by counting: 10...11,12,13,14,15. Correct!</think>
<action>double_check 10 + 5 = 15</action>
<observation>Verification: 15 is correct</observation>
<think>Perfect. The answer is definitely 15.</think>
<action>answer "15"</action>""",
    "final_state": "Calculation completed with excessive verification",
    "model_output": "15",
    "success": True,  # Correct but inefficient
    "steps_taken": 6
}


# ============= Collections for Testing =============

ALL_SUCCESSFUL_TRAJECTORIES: List[Dict[str, Any]] = [
    ARITHMETIC_SUCCESS,
    INFORMATION_SEEKING_SUCCESS,
    NAVIGATION_SUCCESS,
    MULTI_STEP_REASONING_SUCCESS,
    OVERCOMPLEX_SUCCESS
]


ALL_FAILED_TRAJECTORIES: List[Dict[str, Any]] = [
    ARITHMETIC_FAILURE,
    INFORMATION_SEEKING_FAILURE,
    NAVIGATION_FAILURE,
    MISSING_ERROR_HANDLING_FAILURE,
    PARTIAL_SUCCESS
]


ALL_TRAJECTORIES: List[Dict[str, Any]] = (
    ALL_SUCCESSFUL_TRAJECTORIES + ALL_FAILED_TRAJECTORIES
)


# ============= Trajectory Pairs for Comparison =============

TRAJECTORY_PAIRS_FOR_SELF_CONTRAST = [
    {
        "success": ARITHMETIC_SUCCESS,
        "failure": ARITHMETIC_FAILURE,
        "comparison_insight": "Successful trajectory followed order of operations (PEMDAS), failed trajectory did not"
    },
    {
        "success": INFORMATION_SEEKING_SUCCESS,
        "failure": INFORMATION_SEEKING_FAILURE,
        "comparison_insight": "Successful trajectory used specific search terms, failed trajectory used broad generic terms"
    },
    {
        "success": NAVIGATION_SUCCESS,
        "failure": NAVIGATION_FAILURE,
        "comparison_insight": "Successful trajectory had clear target goal and verified each step, failed trajectory clicked randomly"
    }
]


# ============= Helper Functions =============

def get_trajectory_by_type(task_type: str, success: bool) -> Dict[str, Any]:
    """
    Get a sample trajectory by task type and success status.

    Args:
        task_type: One of "arithmetic", "information_seeking", "navigation", "reasoning"
        success: True for successful trajectory, False for failed

    Returns:
        Dict containing trajectory data
    """
    type_map = {
        "arithmetic": (ARITHMETIC_SUCCESS, ARITHMETIC_FAILURE),
        "information_seeking": (INFORMATION_SEEKING_SUCCESS, INFORMATION_SEEKING_FAILURE),
        "navigation": (NAVIGATION_SUCCESS, NAVIGATION_FAILURE),
        "reasoning": (MULTI_STEP_REASONING_SUCCESS, MISSING_ERROR_HANDLING_FAILURE)
    }

    if task_type not in type_map:
        raise ValueError(f"Unknown task type: {task_type}")

    success_traj, failure_traj = type_map[task_type]
    return success_traj if success else failure_traj


def get_trajectory_pair(task_type: str) -> Dict[str, Any]:
    """
    Get a success/failure trajectory pair for comparison.

    Args:
        task_type: One of "arithmetic", "information_seeking", "navigation"

    Returns:
        Dict with "success", "failure", and "comparison_insight" keys
    """
    type_map = {
        "arithmetic": TRAJECTORY_PAIRS_FOR_SELF_CONTRAST[0],
        "information_seeking": TRAJECTORY_PAIRS_FOR_SELF_CONTRAST[1],
        "navigation": TRAJECTORY_PAIRS_FOR_SELF_CONTRAST[2]
    }

    if task_type not in type_map:
        raise ValueError(f"Unknown task type: {task_type}")

    return type_map[task_type]
