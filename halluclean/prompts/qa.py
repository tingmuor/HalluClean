# halluclean/prompts/qa.py

# 1) 规划阶段：Plan
QA_PLAN_PROMPT = """\
You are provided with a question and its corresponding answer.
Your task is to determine whether the answer contains hallucinated content.
Let's understand the task and devise a plan to solve the task.
Question:{question}
Answer:{answer}
"""

# 2) 推理阶段：Reason
QA_REASON_PROMPT = """\
You are provided with a question and its corresponding answer.
Your task is to determine whether the answer contains hallucinated content.
Question:{question}
Answer:{answer}
Plan:{plan}
Let's carry out the plan and solve the task step by step. Show the reasoning process.
"""

# 3) 判决阶段：Judge
QA_JUDGE_PROMPT = """\
You are provided with a question and its corresponding answer.
Your task is to determine whether the answer contains hallucinated content.
Question:{question}
Answer:{answer}
Analysis:{analysis}
Please conclude whether the answer contains hallucinated content with Yes or No.
"""

# 4) 修订阶段：Revise
QA_REVISE_PROMPT = """\
Given a question, its corresponding hallucinated answer, and an analysis explaining why the answer contains hallucinated content.
Your task is to answer the question without introducing any hallucinations.
Question:{question}
Hallucinated Answer:{answer}
Analysis:{analysis}
Just output the answer, without including any additional explanation in the output.
"""

