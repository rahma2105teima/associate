import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

transacts =([{'l1':1,'l2':1,'l5':1}
            ,{'l2':1,'l4':1}
            ,{'l2':1,'l3':1}
            ,{'l1':1,'l2':1,'l4':1}
            ,{'l1':1,'l3':1}
            ,{'l2':1,'l3':1}
            ,{'l1':1,'l3':1}
            ,{'l1':1,'l2':1,'l3':1,'l5':1}
            ,{'l1':1,'l2':1,'l3':1}])


index = ['T'+str(i+1)  for i in range(9)]
df = pd.DataFrame(transacts,index=index)
print(df)

# replace NaN with 0 in the whole DataFrame
df.replace(np.nan, 0,inplace=True)
print(df)
# Building the models and analyzing the results
# Building the model
frq_items = apriori(df, min_support = 0.2, max_len= 3, use_colnames = True )
print(frq_items)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, min_threshold = 1)
rules = rules.sort_values(['confidence'], ascending =[False])
print(rules)
