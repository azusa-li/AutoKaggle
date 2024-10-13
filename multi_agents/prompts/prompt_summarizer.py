PROMPT_SUMMARIZER_IMAGE_CHOOSE = '''
# CONTEXT #
{phases_in_context}

#############
# TASK #
Currently, you are in phase: {phase_name}.
According to the current phase, you need to choose {num} images from the following images, they should be most relevant to the current phase, and useful for the next phase.

#############
# IMAGES #
{images}

#############
# RESPONSE #
Please give me the image names that you choose. You should follow the format:
```json
{{
    "images": list=[
        "[image name 1]",
        "[image name 2]",
        "[image name 3]",
        ...
    ]
}}
```

#############
# START CHOOSING IMAGES #
Let's work this out in a step by step way.
'''


PROMPT_SUMMARIZER_DESIGN_QUESITONS = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase: {phase_name}.

#############
# TASK #
Your task is to design a series of questions that will be used to summarize the findings at the current phase and provide guidance for the next phase. 
The current competition phase is: {phase_name}  
The next phase is: {next_phase_name}

I will provide the competition information (COMPETITION INFO), the plan given by the planner for this phase (PLAN).

Based on this information, design 6 key questions that are most worth focusing on and will be most helpful for the next phase. These questions should:
1. Be targeted, specifically designed for the transition from {phase_name} to {next_phase_name}.
2. Summarize the key findings of the current phase ({phase_name}).
3. Provide guidance for the planner in formulating the execution plan in the next phase ({next_phase_name}).

<sample questions>  
(Assuming the phase is Preliminary EDA and the next phase is Data Cleaning)  
1. What files did you process? Which files were generated? Answer with detailed file path. (This is a FIXED question for each phase.)
2. Which features were involved in this phase? What changes did they undergo? If any feature types were modified, answer which features are modified and how they are modified. If any features were deleted or created, answer which features are deleted or created and provide detailed explanations. (This is a FIXED question for each phase.)
3. Which features have missing values, and what is the proportion of missing values?  
4. Are there any obvious outliers in the data? If so, which features do they mainly affect?  
5. What interesting findings did you have in this phase? What do you think should be done in the next phase to better complete this competition?  
...
</sample questions>

Please ensure that your questions have both breadth and depth, capable of comprehensively summarizing the work of the current phase and providing valuable insights for the upcoming {next_phase_name}. List the questions you design and briefly explain the purpose or importance of each question.

#############
# RESPONSE: MARKDOWN FORMAT #
NOTE that you only need design questions, do not answer the questions yourself.
Let's work this out in a step by step way.

#############
# START DESIGN QUESTIONS #
If you are ready, please request from me the COMPETITION INFO, PLAN.
'''

PROMPT_SUMMARIZER_REORGAINZE_QUESTIONS = '''
# TASK #
Please reorganize the questions that you have designed in the previous reply.

#############
# RESPONSE: MARKDOWN FORMAT #
```markdown
## QUESTIONS
### Question 1
What files did you process? Which files were generated? Answer with detailed file path.

### Question 2
Which features were involved in this phase? What changes did they undergo? If any feature types were modified, answer which features are modified and how they are modified. If any features were deleted or created, answer which features are deleted or created and provide detailed explanations. (This is a FIXED question for each phase.)

### Question 3
[content of question 3]

### Question 4
[content of question 4]

### Question 5
[content of question 5]

### Question 6
[content of question 6]
```

#############
# START REORGANIZE QUESTIONS #
'''

PROMPT_SUMMARIZER_ANSWER_QUESTIONS = '''
# CONTEXT #
{phases_in_context}
Currently, I am at phase: {phase_name}.

#############
# TASK #
Please answer a series of questions that will help summarize the current phase.
Your answer should be concise and detailed, for example, if the question is about how to clean data, your answer should be specific to each feature.
I will provide the competition information (COMPETITION INFO), the plan given by the planner for this phase (PLAN), the code written by the developer in this phase and the output of the code execution (CODE AND OUTPUT), insight from images you generated (INSIGHT FROM VISUALIZATION), as well as the reviewer's evaluation of the planner's and developer's task completion for this phase (REVIEW).
When answering each question, you can first consider which information you need to use, and then answer the question based on this information.

#############
# QUESTIONS #
{questions}

#############
# RESPONSE: MARKDOWN FORMAT #
Let's work this out in a step by step way.

#############
# START ANSWER QUESTIONS #
If you are ready, please request from me the COMPETITION INFO, PLAN, CODE AND OUTPUT, INSIGHT FROM VISUALIZATION, REVIEW.
'''

PROMPT_INFORMATION_FOR_ANSWER = '''
# COMPETITION INFO #
{competition_info}

#############
# PLAN #
{plan}

#############
# CODE AND OUTPUT #
## CODE ##
{code}

## OUTPUT ##
{output}

#############
# INSIGHT FROM VISUALIZATION #
{insight_from_visualization}

#############
# REVIEW #
{review}
'''

PROMPT_SUMMARIZER_REORGANIZE_ANSWERS = '''
# TASK #
Please reorganize the answers that you have given in the previous step, and synthesize them into a report.

#############
# RESPONSE: MARKDOWN FORMAT #
```markdown
# REPORT
## QUESTIONS AND ANSWERS  
### Question 1
What files did you process? Which files were generated? Answer with detailed file path.
### Answer 1
[answer to question 1]

### Question 2
Which features were involved in this phase? What changes did they undergo? If any feature types were modified, answer which features are modified and how they are modified. If any features were deleted or created, answer which features are deleted or created and provide detailed explanations. (This is a FIXED question for each phase.)
### Answer 2
[answer to question 2]

### Question 3
[repeat question 3]
### Answer 3
[answer to question 3]

### Question 4
[repeat question 4]
### Answer 4
[answer to question 4]

### Question 5
[repeat question 5]
### Answer 5
[answer to question 5]

### Question 6
[repeat question 6]
### Answer 6
[answer to question 6]
</Markdown>

#############
# START REORGANIZE QUESTIONS #
'''