import pandas as pd

def identify_join(dfA_path, dfB_path):
    ### 1. Load data ###

    dfA = pd.read_csv(dfA_path, dtype=str)

    dfB = pd.read_csv(dfB_path, dtype=str)

    #print("DataFrame A:")
    #print(dfA)
    #print("\nDataFrame B:")
    #print(dfB)

    ### Function to compute overlap & coverage metrics ###

    def compute_overlap_coverage(dfA, colA, dfB, colB):
        """
        Return a dictionary with overlap_ratio, coverageA, coverageB
        for the given columns in dfA and dfB.
        """
        # Convert columns to sets of distinct values
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
        
        # Coverage A: fraction of rows in dfA that match at least one row in dfB
        # We treat coverage as: # rows in A whose colA is in B[colB] / total rows in A
        matchedA = dfA[colA].isin(common_values).sum()
        coverageA = matchedA / len(dfA) if len(dfA) > 0 else 0
        
        # Coverage B: fraction of rows in dfB that match at least one row in dfA
        matchedB = dfB[colB].isin(common_values).sum()
        coverageB = matchedB / len(dfB) if len(dfB) > 0 else 0
        
        return {
            'overlap_ratio': overlap_ratio,
            'coverageA': coverageA,
            'coverageB': coverageB
        }
    
    ### Generate candidate column pairs and score ###

    results = []
    for colA in dfA.columns:
        for colB in dfB.columns:
            # Basic dtype check
            if pd.api.types.is_numeric_dtype(dfA[colA]) and pd.api.types.is_numeric_dtype(dfB[colB]):
                metrics = compute_overlap_coverage(dfA, colA, dfB, colB)
                
                # Scoring function: average of (overlap, coverageA, coverageB)
                score = (metrics['overlap_ratio'] + metrics['coverageA'] + metrics['coverageB']) / 3
                
                results.append({
                    'colA': colA,
                    'colB': colB,
                    'overlap_ratio': metrics['overlap_ratio'],
                    'coverageA': metrics['coverageA'],
                    'coverageB': metrics['coverageB'],
                    'score': score
                })
            elif pd.api.types.is_object_dtype(dfA[colA]) and pd.api.types.is_object_dtype(dfB[colB]):
                metrics = compute_overlap_coverage(dfA, colA, dfB, colB)
                score = (metrics['overlap_ratio'] + metrics['coverageA'] + metrics['coverageB']) / 3
                
                results.append({
                    'colA': colA,
                    'colB': colB,
                    'overlap_ratio': metrics['overlap_ratio'],
                    'coverageA': metrics['coverageA'],
                    'coverageB': metrics['coverageB'],
                    'score': score
                })

    # Convert results to df
    results_df = pd.DataFrame(results)

    # Sort by score
    results_df = results_df.sort_values('score', ascending=False)


    ### Identify top candidate for joining ###

    # Identify the best scores
    best_score = results_df['score'].max()

    # Get all pairs
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

        #print("\nJoined df:")
        #print(df_joined)

        if len(df_joined) >= max_val:
            max_child = child
            max_parent = parent
            max_val = len(df_joined)

    #print("Best match:", max_child, max_parent)

    return max_child, max_parent