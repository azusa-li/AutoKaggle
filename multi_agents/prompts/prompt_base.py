AGENT_ROLE_TEMPLATE = '''You are an excellent {agent_role}.\n'''

PHASES_IN_CONTEXT_PREFIX = '''
I am working on a data science competition called "# {competition_name}". 
I plan to divide the task into the following phases and complete them in order:
'''

PROMPT_EACH_EXPERIENCE_WITH_SUGGESTION = '''
## EXPERIENCE {index}##
<EXPERIENCE>
{experience}
</EXPERIENCE>
<SUGGESTION>
{suggestion}
</SUGGESTION>
<SCORE>
{score}
</SCORE>
'''

PROMPT_FEATURE_INFO = '''# FEATURE INFO
## TARGET VARIABLE
{target_variable}
## FEATURES BEFORE THIS PHASE
{features_before}
## FEATURES AFTER THIS PHASE
{features_after}
'''

PROMPT_EXTRACT_TOOLS = '''
Please extract the all tools involved in the following document.
{document}
All available tools are as follows:
{all_tool_names}

Each tool name MUST be in the available tool names.
Your response should be in the following format:
```json
{{
    "tool_names": list=[
        "tool_name 1",
        "tool_name 2",
        "tool_name 3",
        ...
    ]
}}
```
'''

REORGANIZE_REPLY_TYPE1 = '''
# TASK #
Please extract essential information and reorganize into a JSON format. You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. You must ensure that the essential information is complete and accurate.
#############
# INFORMATION #
{information}
#############
# RESPONSE: JSON FORMAT #
```json
{{
    "final_thought": str="Summarize your understanding and confirm that you now have the final answer.",
    "final_answer": str="Provide the final answer to the original task."
}}
```
#############
# START REORGANIZING #
'''

REORGANIZE_REPLY_TYPE2 = '''
# TASK #
Please extract essential information and reorganize into a JSON format. You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. You must ensure that the essential information is complete and accurate.
#############
# INFORMATION #
{information}
#############
# RESPONSE: JSON FORMAT #
```json
{{
    "final_thought": str="Summarize your understanding and confirm that you now have the final answer.",
    "final_answer": {{
	    "final_suggestion": {{
            str="agent name": str="Specific suggestions for improving the agent's performance"
        }},
        "final_score": {{
            str="agent name": int="The final score you assign to the evaluated agent, only one score in range 1-5"
        }}
    }}
}}
```
#############
# START REORGANIZING #
'''


REORGANIZE_REPLY_TYPE3 = '''
# TASK #
Please extract essential information and reorganize into a JSON format. You need to organize the information in a clear and concise manner, ensuring that the content is logically structured and easy to understand. You must ensure that the essential information is complete and accurate.
#############
# INFORMATION #
{information}
#############
# RESPONSE: JSON FORMAT #
```json
{{
    "final_thought": str="Summarize your understanding and confirm that you now have the final answer.",
    "final_answer": list=[
        {{
            "task": str="The specific task to be performed",
            "method": list=["Methods to be used"],
        }}
    ]
}}
```
#############
# START REORGANIZING #
'''

# ## Thought Process ##
# (This Thought/Action/Observation sequence may repeat as needed.)
# - Thought: str="Reflect on the current situation and consider how to proceed in fulfilling the user's requirements."
# - Action: str="Describe the action you plan to take to meet the user's needs."
# - Observation: str="Note the expected or actual results of the action."
# ## Final Thought ##
# str="Summarize your understanding and confirm that you now have the final answer."
# ## Final Answer ##
# str="Provide the final answer to the original task."