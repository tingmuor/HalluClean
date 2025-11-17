# halluclean/prompts/mwp.py

# 1) Plan
MWP_PLAN_PROMPT = """\
You are provided with a math word problem.
Your task is to determine whether the problem is unanswerable. 
Let's understand the task and devise a plan to solve the task.
Problem:{question}
"""

# 2) Reason
MWP_REASON_PROMPT = """\
You are provided with a math word problem.
Your task is to determine whether the problem is unanswerable.
Problem:{question}
Plan:{plan}
Let's carry out the plan and solve the task step by step. Show the reasoning process.
"""

# 3) Judge
MWP_JUDGE_PROMPT = """\
You are provided with a math word problem.
Your task is to determine whether the problem is unanswerable.
Problem:{question}
Analysis:{analysis}
Please conclude whether the problem is unanswerable with Yes or No.
"""

# Revise
MWP_REVISE_PROMPT = """\
Given a unanswerable math word problem and an analysis explaining why it is unanswerable.
Your task is to revise the problem to make it answerable.
Problem:{question}
Analysis:{analysis}
Just output the revised problem, without including any additional explanation in the output.
"""
