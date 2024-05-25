In this folder, it is the code of the two-dimensional evaluate

You should first run gpt_evaluator.py and gpt_evaluator_completeness.py to generate the gpt result for evaluating the correctness and completeness separately.

Second, you can run parser_acc_result.py to parse the correctness result to a certain form

Third, please turn into the "analyze" folder, you should first run parser.py to gain the formatted results that combines the correctness and completeness evaluation together.
Do not forget to change the arguments in this parser.py. After obtaining the parsed results, you can run analyze.py to get the turn state accuracy and the coherence between gpt evaluation and evaluation based on MWZ.
Finally, you can run gpt_jga.py to get the JGA score evaluated based on GPT.