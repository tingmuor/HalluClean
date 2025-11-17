# halluclean/prompts/da.py

# 1) Plan
DA_PLAN_PROMPT = """\
You are provided with a dialogue history and its corresponding response.
Your task is to determine whether the response contains hallucinated content. 
Let's understand the task and devise a plan to solve the task.
Dialogue History:{question}
Response:{answer}
"""

# 2) Reason
DA_REASON_PROMPT = """\
You are provided with a dialogue history and its corresponding response.
Your task is to determine whether the response contains hallucinated content. 
Dialogue History:{question}
Response:{answer}
Plan:{plan}
Let's carry out the plan and solve the task step by step. Show the reasoning process.
"""

# 3) Judge
DA_JUDGE_PROMPT = """\
You are provided with a dialogue history and its corresponding response.
Your task is to determine whether the response contains hallucinated content. 
Dialogue History:{question}
Response:{answer}
Analysis:{analysis}
Please conclude whether the response contains hallucinated content with Yes or No.
"""

# Revise
DA_REVISE_PROMPT = """\
Given a dialogue history, its corresponding hallucinated response, and an analysis explaining why the response contains hallucinated content.
Your task is to regenerate the response without introducing any hallucinations.
Dialogue History:{question}
Hallucinated Response:{answer}
Analysis:{analysis}
Just output the response, without including any additional explanation in the output.
"""
