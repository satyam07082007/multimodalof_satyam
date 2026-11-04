import os
import re
import logging
from typing import Dict, Any, Optional, List

import google.generativeai as genai

logger = logging.getLogger("haven.coding")


class CodingAssistantModule:
    """
    Dedicated Software Engineering, Debugging, and Algorithmic Assistant.
    Uses Gemini AI for live code generation, debugging, complexity analysis,
    and language explanations. Falls back to built-in templates offline.
    """

    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript", "java", "cpp", "c",
        "html", "css", "sql", "rust", "go", "bash", "kotlin", "swift"
    ]

    SYSTEM_PROMPT = (
        "You are Haven AI's expert Software Engineering and Coding Assistant.\n"
        "Guidelines:\n"
        "1. Write clean, well-commented, production-quality code.\n"
        "2. Always include time and space complexity analysis (Big-O notation).\n"
        "3. Format code in proper markdown fenced code blocks with the correct language tag.\n"
        "4. Provide a brief line-by-line explanation after the code.\n"
        "5. Include a working example / test case whenever appropriate.\n"
        "6. For debugging requests, clearly identify the root cause before proposing a fix.\n"
        "7. Support all major languages: Python, JavaScript, TypeScript, Java, C++, SQL, Rust, Go, Bash, etc.\n"
        "8. Never hallucinate library APIs — if unsure, say so and suggest checking the docs."
    )

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")

        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                logger.warning("Gemini config in CodingAssistant: %s", e)

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Calls Gemini with the coding system prompt."""
        if not self.gemini_key:
            return None
        try:
            model = genai.GenerativeModel(self.model_name)
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
            res = model.generate_content(full_prompt)
            if res and res.text:
                return res.text.strip()
        except Exception as e:
            logger.error("Gemini coding call failed: %s", e)
        return None

    def generate_code_solution(self, problem_description: str, language: str = "python") -> str:
        """
        Generates robust, clean code with time/space complexity and usage examples.
        Uses live Gemini AI when available; falls back to built-in templates.
        """
        clean_lang = language.lower().strip()

        # --- Live Gemini AI Path ---
        gemini_prompt = (
            f"Generate a complete, clean {clean_lang} solution for the following problem:\n\n"
            f"**Problem**: {problem_description}\n\n"
            f"Requirements:\n"
            f"- Use {clean_lang} as the programming language.\n"
            f"- Include full working code in a fenced code block.\n"
            f"- Add time and space complexity (Big-O) at the end.\n"
            f"- Include a test/example at the bottom.\n"
            f"- Add concise inline comments for key steps."
        )
        ai_result = self._call_gemini(gemini_prompt)
        if ai_result:
            return f"### 💻 Code Studio — Live AI Solution\n\n{ai_result}"

        # --- Offline Fallback Templates ---
        lower_problem = problem_description.lower()

        if "binary search" in lower_problem:
            return (
                "### 💻 Binary Search Implementation\n\n"
                "Binary search runs in **O(log N)** time complexity.\n\n"
                "```python\n"
                "def binary_search(arr: list[int], target: int) -> int:\n"
                "    left, right = 0, len(arr) - 1\n"
                "    while left <= right:\n"
                "        mid = left + (right - left) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            left = mid + 1\n"
                "        else:\n"
                "            right = mid - 1\n"
                "    return -1\n\n"
                "# Example:\n"
                "nums = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]\n"
                "print(binary_search(nums, 23))  # Output: 5\n"
                "```\n\n"
                "**Complexity**: Time **O(log n)** | Space **O(1)**"
            )

        if "fibonacci" in lower_problem:
            return (
                "### 💻 Fibonacci (Dynamic Programming)\n\n"
                "```python\n"
                "def fibonacci(n: int) -> int:\n"
                "    if n < 0: raise ValueError('n must be non-negative')\n"
                "    if n in (0, 1): return n\n"
                "    prev, curr = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        prev, curr = curr, prev + curr\n"
                "    return curr\n\n"
                "print([fibonacci(i) for i in range(10)])\n"
                "# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
                "```\n\n"
                "**Complexity**: Time **O(n)** | Space **O(1)**"
            )

        return (
            f"### 💻 {clean_lang.capitalize()} — Code Solution\n\n"
            f"*(Connect your Gemini API key in `.env` for full AI-powered code generation)*\n\n"
            f"```{clean_lang}\n"
            f"# Solution for: {problem_description}\n"
            f"def solution():\n"
            f"    \"\"\"Implements: {problem_description}\"\"\"\n"
            f"    # TODO: Add your implementation here\n"
            f"    pass\n\n"
            f"if __name__ == '__main__':\n"
            f"    solution()\n"
            f"```\n\n"
            f"**Tip**: Ask me to generate a full implementation, debug your code, or explain a concept!"
        )

    def debug_code(self, code_snippet: str, error_message: Optional[str] = None) -> str:
        """
        Diagnoses syntax, logic, and runtime errors in code using Gemini AI.
        """
        gemini_prompt = (
            f"Debug the following code and provide a detailed diagnosis:\n\n"
            f"**Code to Debug**:\n```\n{code_snippet}\n```\n\n"
            + (f"**Error Message**:\n```\n{error_message}\n```\n\n" if error_message else "")
            + "Please:\n"
            "1. Identify the root cause of the error clearly.\n"
            "2. Show the corrected code in a fenced code block.\n"
            "3. Explain what was wrong and why the fix works.\n"
            "4. Suggest any further improvements or best practices."
        )
        ai_result = self._call_gemini(gemini_prompt)
        if ai_result:
            return f"### 🛠️ AI Code Debugger — Live Analysis\n\n{ai_result}"

        # Offline fallback
        return (
            "### 🛠️ Code Diagnostic & Debugging Report\n\n"
            f"**Error Reviewed**: `{error_message or 'General Logic & Syntax Review'}`\n\n"
            "#### Common Root Causes\n"
            "1. **Index Out of Range**: Check array bounds and loop limits.\n"
            "2. **None / Null Reference**: Ensure objects are initialized before access.\n"
            "3. **Type Mismatch**: Verify function argument types match expectations.\n"
            "4. **Off-by-One**: Double-check loop ranges (`range(n)` vs `range(n+1)`).\n\n"
            "#### Suggested Fix Pattern\n"
            "```python\n"
            "try:\n"
            "    # Your corrected logic here\n"
            "    result = your_function()\n"
            "except (TypeError, IndexError, ValueError) as err:\n"
            "    print(f'Error: {err}')\n"
            "```\n\n"
            "*Add your Gemini API key for a live AI-powered line-by-line diagnosis.*"
        )

    def explain_concept(self, concept: str) -> str:
        """Explains a programming/CS concept clearly using Gemini AI."""
        gemini_prompt = (
            f"Explain the following computer science or programming concept clearly and thoroughly:\n\n"
            f"**Concept**: {concept}\n\n"
            f"Include:\n"
            f"- A clear plain-English definition.\n"
            f"- A practical code example.\n"
            f"- When and why to use it.\n"
            f"- Common pitfalls or mistakes."
        )
        ai_result = self._call_gemini(gemini_prompt)
        if ai_result:
            return f"### 📚 Concept Explanation — {concept}\n\n{ai_result}"

        return (
            f"### 📚 {concept}\n\n"
            f"**{concept}** is a core programming/CS concept.\n\n"
            f"*Connect your Gemini API key to get a full AI-powered explanation with examples.*"
        )

    def analyze_complexity(self, code_snippet: str) -> str:
        """Analyzes time and space complexity of a code snippet using Gemini."""
        gemini_prompt = (
            f"Analyze the time and space complexity of this code:\n\n"
            f"```\n{code_snippet}\n```\n\n"
            f"Provide:\n"
            f"1. Overall Time Complexity (Big-O) with justification.\n"
            f"2. Overall Space Complexity (Big-O) with justification.\n"
            f"3. Any optimization suggestions.\n"
            f"4. Best case / worst case / average case if applicable."
        )
        ai_result = self._call_gemini(gemini_prompt)
        if ai_result:
            return f"### ⏱️ Complexity Analysis\n\n{ai_result}"

        return (
            "### ⏱️ Complexity Analysis\n\n"
            "*Add your Gemini API key for live complexity analysis.*\n\n"
            "**General Guide**:\n"
            "- Single loop → O(n)\n"
            "- Nested loops → O(n²)\n"
            "- Divide & Conquer → O(log n) or O(n log n)\n"
            "- Hash Map lookup → O(1) average"
        )
