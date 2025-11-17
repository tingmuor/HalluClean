# halluclean/prompts/sum.py

# 1) Plan
SUM_PLAN_PROMPT = """\
You are provided with a document and its corresponding summary.
Your task is to determine whether the summary contains hallucinated content. 
Let's understand the task and devise a plan to solve the task.
Document:{question}
Summary:{answer}
"""

# 2) Reason
SUM_REASON_PROMPT = """\
You are provided with a document and its corresponding summary.
Your task is to determine whether the summary contains hallucinated content. 
Document:{question}
Summary:{answer}
Plan:{plan}
Let's carry out the plan and solve the task step by step. Show the reasoning process.
"""

# 3) Judge
SUM_JUDGE_PROMPT = """\
Given a document,its corresponding hallucinated summary , and an analysis explaining why the summary contains hallucinated content.
Your task is to regenerate the summary without introducing any hallucinations.
Document:{question}
Hallucinated Summary:{answer}
Analysis:{analysis}
Just output the summary, without including any additional explanation in the output.
"""
