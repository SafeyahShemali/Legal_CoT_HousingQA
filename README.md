# Simulated Respondents for a Legal AI Reliance Study

Code and data for the simulated pre-test reported in Chapter 4 of the thesis.
Three LLMs each simulated 60 respondents answering 18 US housing-law questions
under three explanation conditions (no explanation, concise, structured),
giving 3,240 responses.

## What this is

A test-bed built to rehearse a human study: constructed AI advice and
explanations spanning a 2x2 of AI-answer correctness and explanation
correctness, answered by LLM-simulated respondents.

## Repository layout

    Annotation/  the 18 items, ground truth, naturalAI advice code 
    Analysis/	 analysis outputs (accuracy tables, Fisher screen, correlations)
    Persona/     building persona and matrix assignment
    Study Item Construction/    Item construction and validation (construction gates, blind-judge results) codes and Results 
    Simulator/ 	 Simulator code and raw responses per model
    GLMM on Respondent/ GLMM on R trial on GPT responses
