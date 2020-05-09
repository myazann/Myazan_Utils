#!/usr/bin/env python
# coding: utf-8

# In[61]:


def Feature_Class_Distribution(df, target, depth = 3, min_leaf_node_size = 100):
    
    class_report = {}
    dot_file = ""
    png_file = ""

    threshold = {}
    thres = [] 
    i = 1
    result = pd.DataFrame(columns = ['Feature', 'Min Value', 'Max Value', 'Count of Class 1' , 'Count of Class 0', 'Total Count', 'Distribution of Class 1'])
    
    if not os.path.isdir("png_files"):        
        os.mkdir("png_files")
        
    if not os.path.isdir("dot_files"):    
        os.mkdir("dot_files")
    
    for column in df.columns:
        if df[column].dtype == 'O':
            df = df.drop(column, axis = 1)
    
    col_names = list(df.drop(target, axis = 1).columns)
    
    for column in col_names:
        temp_df = df[[column,target]]
        temp_df = temp_df.dropna()

        X = temp_df.drop(target,axis=1)
        y = temp_df[target]

        X_train, X_test, y_train, y_test = train_test_split( 
              X, y, test_size = 0.3, random_state = 100)

        temp_X_train = X_train[column]
        temp_X_test = X_test[column]
        temp_X_train = temp_X_train.dropna()
        temp_X_test = temp_X_test.dropna()


        dt = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_leaf = min_node_size)

        dt.fit(temp_X_train.values.reshape(-1,1), y_train)
        y_pred = dt.predict(temp_X_test.values.reshape(-1,1))
        class_report[column] = classification_report(y_test, y_pred,output_dict = True)


        dot_file = "dot_files/" + column + ".dot"
        export_graphviz(dt, dot_file , class_names=True, filled = True, proportion = True)

        png_file = "png_files/" + column + ".png"
        check_call(['dot','-Tpng', dot_file, '-o',png_file])

        dot_file = ""
        png_file = ""

        thres.append(temp_X_train.min())

        for i in sorted([a for a in dt.tree_.threshold if a != -2]): 
            thres.append(i)
        thres.append(temp_X_train.max())

        threshold[column] = thres
        thres = []
        
           
    for key in threshold:
        i = 1
        while i < len(threshold[key]):        

            result = result.append({
                "Feature":key,
                "Max Value":threshold[key][i],
                "Min Value":threshold[key][i-1],
                "Total Count":df.loc[(df[key] >= threshold[key][i-1]) & 
                             (df[key] <= threshold[key][i]),][target].count(),
                "Count of Class 1":df.loc[(df[key] >= threshold[key][i-1]) & 
                             (df[key] <= threshold[key][i]),][target].sum(),
                "Count of Class 0":df.loc[(df[key] >= threshold[key][i-1]) & 
                             (df[key] <= threshold[key][i]),][target].count()
                -df.loc[(df[key] >= threshold[key][i-1]) & 
                             (df[key] <= threshold[key][i]),][target].sum()

            },ignore_index=True)
            i += 1

        if len(df.loc[df[key].isnull()]) != 0:
            result = result.append({
                "Feature":key,
                "Max Value":"Null",
                "Min Value":"Null",
                "Total Count":df.loc[df[key].isnull()][target].count(),
                "Count of Class 1":df.loc[df[key].isnull()][target].sum(),
                "Count of Class 0":df.loc[df[key].isnull()][target].count()-df.loc[df[key].isnull()][target].sum()

            },ignore_index=True)              
    
    result["Distribution of Class 1"] = (result["Count of Class 1"]/result["Total Count"])*100
    result["Distribution of Class 1"] = result["Distribution of Class 1"].astype("float")
    result["Distribution of Class 1"] = round(result["Distribution of Class 1"],2)


    result.columns = result.columns.str.upper()
        
    return result

