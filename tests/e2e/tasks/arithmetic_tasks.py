"""Real arithmetic tasks for self-evolution testing"""

from typing import List
import re


class ArithmeticTask:
    def __init__(self, query: str, expected_answer: int, difficulty: int):
        self.query = query
        self.expected_answer = expected_answer
        self.difficulty = difficulty  # 1-10 scale

    def validate(self, agent_answer: str) -> bool:
        """Validate agent's answer against expected result"""
        # Extract number from agent answer
        numbers = re.findall(r'-?\d+', agent_answer)
        if not numbers:
            return False
        return int(numbers[-1]) == self.expected_answer


# Progressive difficulty tasks
ARITHMETIC_TASKS = [
    # Level 1: Basic addition (5 tasks)
    ArithmeticTask("Calculate 10 + 5", 15, difficulty=1),
    ArithmeticTask("Calculate 23 + 17", 40, difficulty=1),
    ArithmeticTask("Calculate 45 + 38", 83, difficulty=1),
    ArithmeticTask("Calculate 67 + 89", 156, difficulty=1),
    ArithmeticTask("Calculate 102 + 98", 200, difficulty=1),

    # Level 2: Multiplication (5 tasks)
    ArithmeticTask("Calculate 12 * 8", 96, difficulty=2),
    ArithmeticTask("Calculate 15 * 7", 105, difficulty=2),
    ArithmeticTask("Calculate 23 * 4", 92, difficulty=2),
    ArithmeticTask("Calculate 31 * 6", 186, difficulty=2),
    ArithmeticTask("Calculate 42 * 9", 378, difficulty=2),

    # Level 3: Multi-step (10 tasks)
    ArithmeticTask("Calculate (10 + 5) * 3", 45, difficulty=3),
    ArithmeticTask("Calculate 100 - (20 * 3)", 40, difficulty=3),
    ArithmeticTask("Calculate (45 + 15) / 4", 15, difficulty=3),
    ArithmeticTask("Calculate 200 - (50 + 30) * 2", 40, difficulty=3),
    ArithmeticTask("Calculate (12 * 3) + (8 * 5)", 76, difficulty=3),
    ArithmeticTask("Calculate 1000 / (10 + 15 - 5)", 50, difficulty=3),
    ArithmeticTask("Calculate ((20 + 10) * 2) - 15", 45, difficulty=3),
    ArithmeticTask("Calculate (100 / 5) + (30 * 2)", 80, difficulty=3),
    ArithmeticTask("Calculate 500 - ((20 + 30) * 8)", 100, difficulty=3),
    ArithmeticTask("Calculate (15 * 4) - (10 * 3)", 30, difficulty=3),

    # Level 4: Complex expressions (10 tasks)
    ArithmeticTask("Calculate the 5th Fibonacci number", 5, difficulty=4),
    ArithmeticTask("Calculate the 7th Fibonacci number", 13, difficulty=4),
    ArithmeticTask("Calculate factorial of 5", 120, difficulty=4),
    ArithmeticTask("Calculate factorial of 6", 720, difficulty=4),
    ArithmeticTask("Calculate sum of numbers 1 to 10", 55, difficulty=4),
    ArithmeticTask("Calculate sum of even numbers 1 to 20", 110, difficulty=4),
    ArithmeticTask("Calculate 2^8", 256, difficulty=4),
    ArithmeticTask("Calculate 3^5", 243, difficulty=4),
    ArithmeticTask("Calculate greatest common divisor of 48 and 18", 6, difficulty=4),
    ArithmeticTask("Calculate least common multiple of 12 and 8", 24, difficulty=4),

    # Level 5: Word problems (20 tasks)
    ArithmeticTask("If 3 apples cost $6, how much do 7 apples cost?", 14, difficulty=5),
    ArithmeticTask("A car travels 60 miles in 1 hour. How far in 3.5 hours?", 210, difficulty=5),
    ArithmeticTask("If 5 workers complete a task in 8 hours, how long for 10 workers?", 4, difficulty=5),
    ArithmeticTask("A rectangle has width 12 and length 15. What's the area?", 180, difficulty=5),
    ArithmeticTask("Circle with radius 7. What's the area? (use π≈3.14)", 154, difficulty=5),
    ArithmeticTask("A train travels at 80 km/h for 2.5 hours. What distance?", 200, difficulty=5),
    ArithmeticTask("If a book costs $15 and there's a 20% discount, what's the final price?", 12, difficulty=5),
    ArithmeticTask("A box contains 144 eggs. How many dozens is that?", 12, difficulty=5),
    ArithmeticTask("If you save $50 per month, how much in 2 years?", 1200, difficulty=5),
    ArithmeticTask("A recipe needs 3 cups of flour for 12 cookies. How much for 36 cookies?", 9, difficulty=5),
    ArithmeticTask("A garden is 20m by 15m. What's the perimeter?", 70, difficulty=5),
    ArithmeticTask("If 8 pencils cost $4, what's the cost of 1 pencil in cents?", 50, difficulty=5),
    ArithmeticTask("A bus holds 45 passengers. How many buses for 180 passengers?", 4, difficulty=5),
    ArithmeticTask("If a pizza has 8 slices and you eat 3, what fraction remains? (as 8ths)", 5, difficulty=5),
    ArithmeticTask("A bottle holds 2 liters. How many 250ml cups can it fill?", 8, difficulty=5),
    ArithmeticTask("If you run 5km per day for 30 days, total distance?", 150, difficulty=5),
    ArithmeticTask("A clock shows 3:15. How many minutes until 4:00?", 45, difficulty=5),
    ArithmeticTask("If 4 friends split a $60 bill equally, how much each?", 15, difficulty=5),
    ArithmeticTask("A plant grows 3cm per week. How tall after 8 weeks starting at 10cm?", 34, difficulty=5),
    ArithmeticTask("If you double your money 4 times starting with $5, what's final amount?", 80, difficulty=5),
]


def get_tasks_by_difficulty(difficulty: int) -> List[ArithmeticTask]:
    """Get all tasks of specific difficulty level"""
    return [task for task in ARITHMETIC_TASKS if task.difficulty == difficulty]


def get_progressive_task_sequence() -> List[ArithmeticTask]:
    """Get tasks in progressive difficulty order for learning"""
    return sorted(ARITHMETIC_TASKS, key=lambda t: t.difficulty)
