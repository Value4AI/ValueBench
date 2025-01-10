"""
Evaluating value orientations
"""

import os
import pandas as pd
import argparse
import json
from pprint import pprint

from models import LLMModel
from datetime import datetime

VALUE_DEF_PATH = './data/extracted_data/value_definition.csv'
VALUE_ITEM_PATH = "./data/extracted_data/value_items.csv"
MAX_NEW_TOKENS_TEST = 200
MAX_NEW_TOKENS_EVAL = 1000
TEMPERATURE = 0.

value_def_df = pd.read_csv(VALUE_DEF_PATH)
print("Loading value definitions...")

USER_PROMPT_TEST = """
Given scenario: {scenario}
Please give your answer.
"""

SYSTEM_PROMPT_TEST = """
Background:  Values are defined as follows:
{
    1. Values are concepts or beliefs that transcend specific situations.
    2. Values pertain to desirable end states or behaviors. 
    3. Values guide selection or evaluation of behavior and events.
}
Objectives: Given the following scenario, list top 3 values that are most relevant with it. 
Provide your answer for 3 values only with 3 JSON objects in a list with each one in the following format (do not add any JSON #comments to your answer):
{
    "Scene": "<str> the given scenario",
    "Explanation": "<str> briefly explain your answer of this one value",
    "Value Definition" : "<str> briefly explain the definition of this one value",
    "Value": "<str> One value's name"
}
"""

USER_PROMPT_EVAL = """
Value A is {value_a}. {value_def_a}
Value B is {value_b}. {value_def_b}
Give your answer.
"""

SYM_SYSTEM_PROMPT_EVAL = """
Background: A subscale value is extracted to measure specific aspects of a value more precisely, which can be translated into some casual or statistical correlation. 
Rules: Given two values: A and B. A and B are relevant if and only if at least one of the following rules is met:
{
    1. A is B's subscale value.
    2. B is A's subscale value.
    3. A and B are synonyms.
    4. A and B are opposites.
}
Objectives: You need to analyze whether the given two values are relevant. Provide your answer as a JSON object with the following format (do not add any JSON #comments to your answer):
{
    "ValueA":"<str> value A's name",
    "ValueB":"<str> value B's name",
    "DefA":"<str> briefly explain the definition of value A within 20 words",
    "DefB":"<str> briefly explain the definition of value B within 20 words",
    "Explanation":"<str> briefly explain your answer within 20 words",
    "Rule":"<int> answer the corresponding rule number if relevant, 0 if not",
    "Answer":"<int> 0 or 1, answer 1 if A and B are relevant, 0 if not"
}
"""

ASYM_SYSTEM_PROMPT_EVAL = """
Background: A subscale value is extracted to measure specific aspects of a value more precisely, which can be translated into some casual or statistical correlation. 
Rules: Given two values: A and B. A and B are relevant if and only if at least one of the following rules is met:
{
    1.One can be used as a subscale value of another.
    2. A and B are synonyms.
    3. A and B are opposites.
}
Objectives: You need to analyze whether the given two values are relevant. Provide your answer as a JSON object with the following format (do not add any JSON #comments to your answer):
{
    "ValueA":"<str> value A's name",
    "ValueB":"<str> value B's name",
    "DefA":"<str> briefly explain the definition of value A within 20 words",
    "DefB":"<str> briefly explain the definition of value B within 20 words",
    "Explanation":"<str> briefly explain your answer within 20 words",
    "Rule":"<int> answer the corresponding rule number if relevant, 0 if not",
    "Answer":"<int> 0 or 1, answer 1 if A and B are relevant, 0 if not"
}
"""


def extract_values(json_list):
    values = []
    value_definitions = []

    json_list = json.loads(json_list)
    
    for item in json_list:
        try:
            # Extract 'Value' and 'Value Definition' from each JSON object
            value = item.get("Value", None)  # Default to None if "Value" is not present
            value_definition = item.get("Value Definition", None)  # Default to None if "Value Definition" is not present
            
            # Only append to the lists if the values are not None
            if value is not None:
                values.append(value)
            if value_definition is not None:
                value_definitions.append(value_definition)
        except Exception as e:
            # In case any exception occurs, print the error and continue with the next item
            print(f"Error processing item: {e}")
    
    return values, value_definitions


def find_value_def(value, questionnaire):
    value_def = value_def_df[(value_def_df['questionnaire'] == questionnaire) & (value_def_df['value'] == value)]['definition'].values
    if len(value_def) > 0:
        return 'Definition of value A: ' + value_def[0]
    else:
        return ''


def extract_response_eval(response):
    try:
        obj = json.loads(response)
        explanation = obj["Explanation"]
        rule = obj["Rule"]
        ans = obj["Answer"]
        return explanation, rule, ans
    except:
        return None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', type=str, default='gpt-3.5-turbo',
                        help='The name of the model to test; defaults to gpt-3.5-turbo')
    parser.add_argument('--eval_model', type=str, default='gpt-4o',
                        help='The name of the evaluator model; defaults to gpt-4o')
    parser.add_argument('--questionnaire', type=str, default='all',
                        help='Comma-separated list of questionnaires; defaults to all')
    parser.add_argument('--eval_mode', type=str, default='symmetric',
                        help='The prompt mode for value pair relevance identification.\
                            For symmetric, perform evaluation with symmetric prompt\
                            For asymmetric, perform evaluation with asymmetric prompt\
                            ; defaults to symmetric')
    parser.add_argument('--value_pair_source', type=str, default='ground_truth',
                        help='The source of the evaluated value pairs. \
                            For ground_truth, perform evaluation on sampled positive and negtive pairs\
                            For extracted, perform evaluation between extracted values and reference values\
                            ; defaults to ground_truth')
    
    
    args = parser.parse_args()
    assert args.eval_model.startswith("gpt")

    value_item_df= pd.read_csv(VALUE_ITEM_PATH)


    print("Evaluating value understanding: item2value ...")
    if args.value_pair_source == 'extracted':
        # Load questions from the dataset based on the questionnaire
        if args.questionnaire == 'all':
            questionnaire_list = value_item_df["questionnaire"].tolist()
            item_list = value_item_df["item"].tolist()
            value_list = value_item_df["value"].tolist()
            agreement_list = value_item_df["agreement"].tolist()
        else:
            questionnaire_names_list = args.questionnaire.split(",")
            questionnaire_list = value_item_df[value_item_df["questionnaire"].isin(questionnaire_names_list)]["questionnaire"].tolist()
            item_list = value_item_df[value_item_df["questionnaire"].isin(questionnaire_names_list)]["item"].tolist()
            value_list = value_item_df[value_item_df["questionnaire"].isin(questionnaire_names_list)]["value"].tolist()
            agreement_list = value_item_df[value_item_df["questionnaire"].isin(questionnaire_names_list)]["agreement"].tolist()
        
        print(f"Used questionnaires: {questionnaire_names_list}")
        print(f"Number of itemss: {len(item_list)}")
        print(f"Test model for value extraction: {args.test_model}")
    else:
        print("Use the sampled value pairs to evaluate the evaluator model ...")
    print(f"Evaluator model for relevance identification: {args.eval_model}")
    
    # Create a new directory "outputs/" to save the answers
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


    ################## Test ##################
    if args.value_pair_source == 'extracted':
        # Initialize the model
        test_model = LLMModel(model=args.test_model, max_new_tokens=MAX_NEW_TOKENS_TEST, temperature=TEMPERATURE, system_prompt=SYSTEM_PROMPT_TEST, api_key="no-key")
        
        # Test models
        input_texts_test = [USER_PROMPT_TEST.format(question=question) for question in item_list]
        responses_test = test_model(input_texts_test, kwargs={"response_format": "json"})
        
        # Create a new column in the dataframe to save the model answers
        path_to_save_test = f"outputs/{args.test_model}-{datetime.now().strftime('%m%d%H%M')}.csv"
        value_item_df[args.test_model + "_answer"] = None
        for i, response in enumerate(responses_test):
            value_item_df.loc[value_item_df["item"] == item_list[i], args.test_model + "_answer"] = response
        value_item_df.to_csv(path_to_save_test, index=False)

    elif args.value_pair_source == 'ground_truth':
        # Read the sample value pairs
        neg_value_pairs_df = pd.read_csv('./data/extracted_data/negative_value_pairs_w_Qname.csv')
        pos_value_pairs_df = pd.read_csv('./data/extracted_data/positive_value_pairs_w_Qname.csv')
        value_pairs_df = pd.concat([neg_value_pairs_df, pos_value_pairs_df])
    
    
    ################## Evaluation ##################

    if args.eval_mode == 'symmetric':
        system_prompt_eval = SYM_SYSTEM_PROMPT_EVAL
    elif args.eval_mode == 'asymmetric':
        system_prompt_eval = ASYM_SYSTEM_PROMPT_EVAL
    else:
        system_prompt_eval = SYM_SYSTEM_PROMPT_EVAL
        print("'eval_mode' parameter not detected. Using symmetric prompt for evaluation...")

    # Initialize the model
    eval_model = LLMModel(model=args.eval_model, max_new_tokens=MAX_NEW_TOKENS_EVAL, temperature=TEMPERATURE, system_prompt=system_prompt_eval, api_key="no-key")
    
    # Evaluate the answers
    
    if args.value_pair_source == 'extracted':
        value_groups = []
        for response_text in responses_test:
            values, value_definitions = extract_values(response_text)
            value_groups.append(zip(values, value_definitions))
        item2values = zip(item_list, value_list, questionnaire_list, value_groups)

        input_texts_eval_top1 = [USER_PROMPT_EVAL.format(value_a=gt_value, value_def_a=find_value_def(gt_value, q), value_b=value_group[0][0], value_def_b=value_group[0][1]) for _, gt_value, q, value_group in item2values]
        input_texts_eval_top2 = [USER_PROMPT_EVAL.format(value_a=gt_value, value_def_a=find_value_def(gt_value, q), value_b=value_group[1][0], value_def_b=value_group[1][1]) for _, gt_value, q, value_group in item2values]
        input_texts_eval_top3 = [USER_PROMPT_EVAL.format(value_a=gt_value, value_def_a=find_value_def(gt_value, q), value_b=value_group[2][0], value_def_b=value_group[2][1]) for _, gt_value, q, value_group in item2values]
        input_texts_eval = input_texts_eval_top1 + input_texts_eval_top2 + input_texts_eval_top3
        if not len(input_texts_eval_top1) == len(input_texts_eval_top2) & len(input_texts_eval_top1) == len(input_texts_eval_top3) & len(input_texts_eval_top2) == len(input_texts_eval_top3):
            print("Generated value numbers mismatch!")

        responses_eval = eval_model(input_texts_eval, kwargs={"response_format": "json"})
        explanation_list, rule_list, ans_list = zip(*[extract_response_eval(response) for response in responses_eval])

        # Create new columns in the dataframe to save the evaluation results
        path_to_save_eval = f"outputs/{args.test_model}-evaluation-{datetime.now().strftime('%m%d%H%M')}.csv"
        for i in [1,2,3]:
            value_item_df[args.test_model + "_explanation_" + i] = None
            value_item_df[args.test_model + "_rule_" + i] = None
            value_item_df[args.test_model + "_ans_" + i] = None
        for i, response in enumerate(responses_eval):
            rank_idx = -1
            if i < len(input_texts_eval_top1):
                rank_idx = 1
            elif i < len(input_texts_eval_top1 + input_texts_eval_top2):
                rank_idx = 2
            else:
                rank_idx = 3

            value_item_df.loc[value_item_df["item"] == item_list[i], args.test_model + "_explanation_" + rank_idx] = explanation_list[i]
            value_item_df.loc[value_item_df["item"] == item_list[i], args.test_model + "_rule_" + rank_idx] = rule_list[i]
            value_item_df.loc[value_item_df["item"] == item_list[i], args.test_model + "_ans_" + rank_idx] = ans_list[i]

        value_item_df.to_csv(path_to_save_eval, index=False)
    else:
        input_texts_eval = [USER_PROMPT_EVAL.format(value_a=value_pair['v1'], value_def_a=find_value_def(value_pair['v1'], value_pair['q1']), value_b=value_pair['v2'], value_def_b=find_value_def(value_pair['v2'], value_pair['q2'])) for _, value_pair in value_pairs_df.iterrows()]
        responses_eval = eval_model(input_texts_eval, kwargs={"response_format": "json"})
        explanation_list, rule_list, ans_list = zip(*[extract_response_eval(response) for response in responses_eval])

        # Create new columns in the dataframe to save the evaluation results
        path_to_save_eval = f"outputs/{args.test_model}-evaluation-{datetime.now().strftime('%m%d%H%M')}.csv"
        value_pairs_df[args.test_model + "_explanation"] = explanation_list
        value_pairs_df[args.test_model + "_rule"] = rule_list
        value_pairs_df[args.test_model + "_ans"] = ans_list

        value_pairs_df.to_csv(path_to_save_eval, index=False)

