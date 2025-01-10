# ValueBench Dataset

We provide data files extracted from the ValueBench dataset in order to test value understanding. 

- `value_definition.csv`: The collected value dimensions with their definitions.
- `value_items.csv`: The collected psychometric items with their corresponding values.
- `positive_value_pairs_w_Qname.csv`: The collected relevant value pairs.
- `negative_value_pairs_w_Qname.csv`: The sampled irrelevant value pairs that have been mannually checked.

We have labeled the data with their original inventories (questionnaires). Specially, the relation label can be interpreted as follows:
- irrelevant: 0
- sub_of: 1
- super_of: 2
- synonym_of: 3
- opposite_of: 4
  