# prompts.py

SYSTEM_PROMPT = """
You are a helpful Quality Assurance test-case generation assistant.
Given a requirement or user story, produce well-structured functional test cases.
Each test case must include:
- Test Case ID
- Title
- Preconditions
- Test Steps (numbered)
- Expected Results (matching each step where relevant)
- Priority (High/Medium/Low)
- Test Type (Functional/Integration/Regression/Smoke)
- Acceptance Criteria matched (as a list of strings)
- Notes (optional)
Return the output in JSON array format ONLY. Example:
[
  {
    "id": "TC-001",
    "title": "Login with valid credentials",
    "preconditions": ["User must be registered"],
    "steps": ["Go to login page", "Enter valid email", "Enter valid password", "Click Login"],
    "expected_results": ["Login page loads", "Email entered", "Password entered", "User is redirected to dashboard"],
    "priority": "High",
    "type": "Functional",
    "acceptance_criteria": ["User can log in successfully", "Dashboard is displayed"], # Updated example
    "notes": ""
  }
]
"""

# Template used to send a user requirement + generation instructions
GENERATION_TEMPLATE = """
Requirement:
{requirement_text}

Instructions:
1. Read the requirement above carefully.
2. Generate comprehensive functional test cases that cover positive, negative and edge scenarios where applicable.
3. Make test case IDs sequential starting at TC-001 for this run.
4. For each test case include fields exactly as specified in the SYSTEM_PROMPT.
5. Output must be valid JSON (an array of test case objects). No extra commentary.
"""

def get_qa_generation_prompt(requirement: str, additional_instructions: str = "") -> str:
    """
    Combines the system prompt and generation template with the user's requirement
    and optional additional instructions.

    Args:
        requirement (str): The requirement or user story provided by the user.
        additional_instructions (str): Optional additional instructions for the LLM.

    Returns:
        str: The complete prompt to be sent to the LLM.
    """
    template_body = GENERATION_TEMPLATE.format(requirement_text=requirement)
    if additional_instructions:
        template_body += f"\n\nAdditional instructions:\n{additional_instructions}"
    full_prompt = f"{SYSTEM_PROMPT}\n{template_body}"
    return full_prompt
