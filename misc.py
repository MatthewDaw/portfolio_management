import numpy as np

def getHoleBlocks(input, col):
    '''
    This function finds the nan holes in a datset so we can know what we need to fill
    :param input (pandas dataframe): the inputted data to search holes for
    :param col (string): col of dataset we wish to examine
    :return holeBlocks (NX2 int Array),: returns holes in dataset. Each entry holeBlocks[i] corresponds to
      tuple [a,b] where a is the index of where a hole starts and b is the index of where that hole ends
    '''
    values = input[col]
    holes = values[values != values]
    holeBlocks = []
    end = -1
    for holeIndex in holes.index:
        start = holeIndex
        if end >= holeIndex:
            continue
        end = holeIndex
        i = 1
        if holeIndex + i < len(values):
            while holeIndex + i < len(values) and values.iloc[holeIndex + i] != values.iloc[holeIndex + i]:
                i += 1
                if len(values) < holeIndex + i:
                    break
        end = start + i - 1
        holeBlocks.append([start, start + i])
    return holeBlocks


def getContinuousBlocks(values, acceptableGapsSize=0):
    '''
    This function gets all blocks that have values unbroken by nans. It ignores nan blocks if they are larger than a
    certain size
    :param input (pandas datafrmae): data to get blocks from
    :param col (string): column to get blocks from
    :param acceptableGapsSize (int): max size of holes allowed in a "continous" block
    :return acceptedBlocks (NX2 int Array),: returns continous blocks in dataset. Each entry acceptedBlocks[i] corresponds to
        tuple [a,b] where a is the index of where an accepted block starts and b is the index of where that block ends
    '''
    # get input where values are non-nan
    # values = input[col]

    # code trick to get all nan values from a list
    filledParts = values[values == values]
    filledPartBlocks = []
    end = -1

    # loop through each valid point
    for filledPartIndex in filledParts.index:
        start = filledPartIndex
        if end >= filledPartIndex:
            continue
        i = 1
        if filledPartIndex + i < len(values):
            while filledPartIndex + i < len(values) and values.iloc[filledPartIndex + i] == values.iloc[
                filledPartIndex + i]:
                i += 1
                if len(values) < filledPartIndex + i:
                    break
        end = start + i - 1
        filledPartBlocks.append([start, start + i])

    # accepted blocks to return
    acceptedBlocks = []

    if len(filledPartBlocks) > 0:
        if len(filledPartBlocks[0]) > 0:

            # filling accepted blocks up
            start = filledPartBlocks[0][0]
            for i in range(1, len(filledPartBlocks)):
                if filledPartBlocks[i][0] - filledPartBlocks[i - 1][1] > acceptableGapsSize:
                    acceptedBlocks.append([start, filledPartBlocks[i - 1][1]])
                    start = filledPartBlocks[i][0]

            acceptedBlocks.append([start, filledPartBlocks[-1][1]-1])

    if len(acceptedBlocks) > 0:
        acceptedBlocks[-1][-1] += 1

    return acceptedBlocks


def getContinuousBlocksFullDataset(args, pandasInput, acceptableGapsSize=0):
    all_blocks = {}
    for col in pandasInput.columns:
        blocks_for_column = []
        if not col in args.columns_not_to_process:

            blocks_for_col = getContinuousBlocks(pandasInput[col], acceptableGapsSize=0)

            blocks_for_column.append(blocks_for_col)

            # for block in blocks_for_col:
            #     continuousDataBlock = pandasInput[col][block[0]:block[1]].to_numpy()
            #     assert not (np.isnan(continuousDataBlock)).any()

        all_blocks[col] = blocks_for_column

    return all_blocks