This folder contains the original code and data for one-step evaluation using GPT.

First, you should run gpt_evaluator to get the original evaluation data.
Do not forget to choose different prompt templates.

Second, you should run turn_level_judge.py to parse the original data to a certain form.

Finally, you can run calc_new_turn_level.py to get the turn state accuracy based on GPT and the coherence of this evaluation between GPT and MWZ.