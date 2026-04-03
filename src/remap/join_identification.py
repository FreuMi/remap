import pandas as pd

### Function to compute overlap ###
def compute_overlap_ratio(dfA, colA, dfB, colB):
    distinctA = set(dfA[colA].dropna().unique())
    distinctB = set(dfB[colB].dropna().unique())
    
    # Intersection of distinct values
    common_values = distinctA.intersection(distinctB)
    intersection_size = len(common_values)
    
    # Overlap ratio = intersection / min(sizeA, sizeB)
    sizeA = len(distinctA)
    sizeB = len(distinctB)
    if min(sizeA, sizeB) == 0:
        overlap_ratio = 0
    else:
        overlap_ratio = intersection_size / min(sizeA, sizeB)
    
    return overlap_ratio

def identify_join(dfA_path, dfB_path):
    ### Load data ###
    dfA = pd.read_csv(dfA_path, dtype=str)
    dfB = pd.read_csv(dfB_path, dtype=str)
    
    ### Generate candidate column pairs and score ###
    results = []
    for colA in dfA.columns:
        for colB in dfB.columns:
            score = compute_overlap_ratio(dfA, colA, dfB, colB)
            
            results.append({
                'colA': colA,
                'colB': colB,
                'score': score
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    ### Identify top candidate for joining ###
    best_score = results_df['score'].max()
    best_pairs = results_df[results_df['score'] == best_score]

    # Print all best pairs
    #print("\nBest candidate pairs")
    #print(best_pairs[['colA', 'colB', 'score']].to_string(index=False))

    max_val = 0
    max_child = ""
    max_parent = ""
    for row in best_pairs.itertuples(index=False):
        row = row._asdict()
        child = row["colA"]
        parent = row["colB"]

        ### Perform join on the best pairs and find max ###
        df_joined = dfA.merge(dfB, left_on=child, right_on=parent, how='inner')

        if len(df_joined) >= max_val:
            max_child = child
            max_parent = parent
            max_val = len(df_joined)

    return max_child, max_parent