

def standardizer_data(args, pandas_input):
    for col in pandas_input.columns:
        if not col in args.columns_not_to_process:
            mask = pandas_input[col] == pandas_input[col]
            filledValues = pandas_input[col][mask]
            filledValues -= min(filledValues)
            filledValues /= max(filledValues)
            pandas_input.loc[mask, col] = filledValues

    return pandas_input