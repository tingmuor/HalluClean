# halluclean/prompts/tsc.py

# 1) Plan
TSC_PLAN_PROMPT = """\
You are given two texts. 
Your task is to determine whether the information in the two texts is contradictory. 
Let's understand the task and devise a plan to solve the task.
Text 1:
{text1}
Text 2:
{text2}
"""

# 2) Reason
TSC_REASON_PROMPT = """\
You are given two texts. 
Your task is to determine whether the information in the two texts is contradictory. 
Text 1:
{text1}
Text 2:
{text2}
Plan:
{plan}
Let's carry out the plan and solve the task step by step.
Show the reasoning process.
"""

# 3) Judge
TSC_JUDGE_PROMPT = """\
You are given two texts. 
Your task is to determine whether the information in the two texts is contradictory. 
Text 1:
{text1}
Text 2:
{text2}
Analysis:
{analysis}
Please conclude whether the two texts are contradictory with Yes or No.
"""

# Revise
TSC_REVISE_PROMPT = """\
Given Text 1, Text 2, and the analysis of the contradiction between them. 
Your task is to revise Text 2 to remove the contradiction, making it consistent with Text 1.
Text 1:{text1}
Text 2:{text2}
Analysis:{analysis}
Just output the revised Text 2, without including any additional explanation in the output.
"""
