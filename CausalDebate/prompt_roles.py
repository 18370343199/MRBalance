roles ={
    "knowledge graphs expert":"You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
    "causal analysis expert":"You are a causal analysis expert, specializing in identifying and reasoning about causal relationships between events. You need to accurately find causally related event pairs from the provided text and explain their logical connections.",
    "NLPexpert":"You are an expert in natural language processing, focusing on extracting causal cues from language. You will analyze whether there is a causal connection between two events based on the explanation of causal triggers, temporal relationships, and character actions, and make judgments based on the logical structure in the language.",
    "none":""
}
system_prompt = """Can you answer the following question as accurately as possible?"""

prompt_roles_intra={
    "none":"",
    "positive":"""
When analyzing a sentence, consider the following questions:

Are there causal language indicators?
Does the sequence of events support the causal direction?
Is there a plausible mechanism linking A and B?
Are there alternative explanations?
Is the relationship consistent and supported by evidence?
---
""",
    "normal":"""
1.Event Clarity
Events A and B must be clearly distinct.
If they are vague or overlapping → causality cannot be judged.
2.Temporal Requirement
A potential cause must precede its effect.
If B precedes A → consider B causes A.
If order is unclear → causality not established.
3.Mechanism Requirement
There must be a plausible, reasonable pathway linking one event to the other.
Without a mechanism, a causal claim is not valid.
4.Counterfactual Consideration
Ask: If the supposed cause did not happen, would the effect still occur or change?
If yes → supports causality.
If no → weakens causality.
5.Confounder Check
If a third factor explains both A and B, the direct causal link between them is not established.
If evidence favors such an explanation, prefer None.
6.Language and Expression
Strong causal verbs (cause, lead to, result in, trigger, because) strengthen the claim.
Associative words (related, linked, correlated) alone are insufficient.
Hedging (may, might, possibly, likely) = uncertain evidence; can support causality only if combined with temporal and mechanism evidence.
7.Decision Principle
To accept A→B: need temporal priority + plausible mechanism + no strong confounder.
To accept B→A: same rules applied in reverse.
If neither direction meets the requirements → None.
---
""",
    "normal1":"""
Refined Rules for Identifying Causal Relationships:

1.Look for Causal Indicators:
Keyword Identification: Pay attention to words in the sentence that indicate causality, such as "because," "leads to," "causes," "results in," "so," "therefore," "thus," "due to," etc.
Voice Analysis: Active voice often indicates causality more directly than passive voice.
2.Determine the Temporal Order of Events:
Cause Before Effect: Ensure that Event A (cause) occurs before or simultaneously with Event B (effect).
Avoid Temporal Reversal: If B occurs before A, it might be that B causes A; assess carefully.
3.Evaluate Logic and Mechanism:
Rationality: Judge whether it is logically reasonable for A to cause B, consistent with common sense or scientific principles.
Mechanism Explanation: Is there a known mechanism explaining how A leads to B?
4.Rule Out Other Possibilities:
Control Confounding Factors: Consider whether there is a third factor C that affects both A and B.
Independence Verification: Confirm that the relationship between A and B is not entirely explained by other factors.
5.Assess the Strength of Association:
Direct Association: Is the association between A and B direct rather than indirect?
Frequency of Association: When A occurs, does B often or always occur?
6.Determine the Direction of Causality:
Unidirectionality: Clarify whether A causes B or B causes A to avoid confusion.
Possibility of Bidirectionality: If bidirectional causality exists, evaluate each direction separately.
7.Consider Specificity:
Exclusivity: Does A cause only B and not other unrelated outcomes?
Exceptions: If A causes multiple outcomes, focus on its relationship with B.
8.Observe Consistency:
Repeated Verification: Is the relationship where A causes B consistent across different situations or multiple observations?
Applicability Across Contexts: Does the causal relationship hold in different environments and populations?
9.Dose-Response Relationship (if applicable):
Quantitative Changes: Does an increase in the intensity or amount of A lead to an increase in the intensity or occurrence of B?
10.Empirical Support:
Empirical Evidence: Is there data, experiments, or research supporting the conclusion that A causes B?
Expert Consensus: Is the causal relationship widely recognized in the professional field?
---
    """,
    "negative":"""
there are some examples for ECI:
Input: The teacher praised the student, and the student completed the assignment.
Question: What is the causal relationship between "the teacher praised the student" and "the student completed the assignment"?
Answer: (C)
Rationale: There is no direct evidence in the sentence to suggest that the praise caused the assignment completion, so there is no causal relationship.
Input: The heavy rain caused flooding in the city, leading to severe traffic jams.
Question: What is the causal relationship between "heavy rain" and "flooding in the city"?
Answer: (A)
Rationale: Heavy rain directly led to flooding, indicating that "heavy rain" was the cause of "flooding in the city".
Input: The car accident was caused by the slippery road conditions.
Question: What is the causal relationship between "the car accident" and "the slippery road conditions"?
Answer: (B)
Rationale: The slippery road conditions led to the car accident, meaning the car accident was caused by the slippery road conditions.
Input: The teacher praised the student, and the student completed the assignment.
Question: What is the causal relationship between "the teacher praised the student" and "the student completed the assignment"?
Answer: (C)
Rationale: There is no direct evidence in the sentence to suggest that the praise caused the assignment completion, so there is no causal relationship.
---
    """
}


def construct_ranking_message(responses, question):

    prefix_string = "Here is the question:\n" + question + "\n\nThese are the solutions to the problem from other agents: "

    for aid, aresponse in enumerate(responses, 1):
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\nPlease choose the best 2 solutions and think step by step. Put your answer in the form like [1,2] or [3,4] at the end of your response."

    return {"role": "user", "content": prefix_string}

def construct_message(responses, question,prompt_type="normal"):

    if len(responses) == 0:
        prefix_string =prompt_roles_intra[prompt_type] + "\nHere is the question:\n" +  question + "\nPut your answer in the form (X) at the end of your response. (X) represents choice (A), (B), or (C). Your answer should be concise and your reasons should be condensed into one sentence."
        return {"role": "user", "content": prefix_string}
    prefix_string = "\nHere is the question:\n"  + question + "\n\nThese are the solutions to the problem from other agents: "

    for aid, aresponse in enumerate(responses, 1):
        response = "\n\nAgent solution " + str(aid) + ": ```{}```".format(aresponse)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\nBased off the opinion of other agents, can you give an updated response. Examine your solution and that other agents. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), or (C). Along with the answer, give a score ranged from 1 to 5 to the solutions of other agents. Put all {} scores in the form like [[1, 5, 2, ...]].".format(len(responses))
    
    return {"role": "user", "content": prefix_string}
