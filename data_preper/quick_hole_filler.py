import numpy as np

from misc import getHoleBlocks


def fillQuickHoles(args, pandasInput):
    '''
    For this function, we will fill all nan holes in data that are smaller than args.largest_holes_to_fill points
    :param dataSetInput: data set to fill the holes of
    :return: version of dataset with holes filled
    '''

    def explorePossibleValues(data, hole, args):
        '''
        In order to fill holes, we need points to interpolate with. To get these points, we wearch for canddiates
        to left and right of point. If there is no hole that is larger than args.patience size then we accept the candidate
        point for interpolation.
        :param data: data to search for candidates
        :param hole: hole to search around
        :param args.patience: number of gaps to allow before rejecting point
        :param args.max_branch: maximum of accepted points to one side that is needed
        :return: points that can be used for interpolation
        '''

        leftBranch = data[data.index < hole[0]].copy()
        acceptedLeftValues = []
        # searching left size of data point
        if len(leftBranch) > 0:
            validValuesLeft = leftBranch[leftBranch == leftBranch]
            startindIndex = hole[0] - 1

            for i in range(len(validValuesLeft)):
                if i >= len(validValuesLeft):
                    break
                if abs(validValuesLeft.index[-1 * (i + 1)] - startindIndex) > args.patience or len(
                        acceptedLeftValues) >= args.max_branch:
                    break
                acceptedLeftValues.append(validValuesLeft.index[-1 * (i + 1)])
                startindIndex = validValuesLeft.index[-1 * (i + 1)]

        # searching right side of data point
        rightBranch = data[data.index > hole[1]]
        acceptedRightValues = []

        # comining both branches of accepted points
        if 0 < len(rightBranch):
            validValuesRight = rightBranch[rightBranch == rightBranch]
            startindIndex = hole[1]

            for i in range(len(validValuesRight)):
                if i >= len(validValuesRight):
                    break
                if abs(validValuesRight.index[i] - startindIndex) > args.patience or len(
                        acceptedRightValues) >= args.max_branch:
                    break
                acceptedRightValues.append(validValuesRight.index[i])
                startindIndex = validValuesRight.index[i]
        return acceptedLeftValues, acceptedRightValues
        # end of search

    if pandasInput is None:
        return None

    columnNames = pandasInput.columns
    updatedIndexes = {}
    pointsFilled = 0

    for col in columnNames:
        updatedColumnIndexes = []
        if not col in args.columns_not_to_process:
            holeBlocks = getHoleBlocks(pandasInput, col)

            # fill in specific hole one at a time
            for hole in holeBlocks:
                if hole[1] - hole[0] < args.largest_holes_to_fill:
                    # get branches to interpolate with
                    leftBranch, rightBranch = explorePossibleValues(pandasInput[col], hole, args)

                    # we need at least 5 known points near gap, or we can't fill it in
                    if len(leftBranch) + len(rightBranch) < args.min_points_needed_to_interpolate:
                        continue

                    # process branches to right size and format
                    if len(leftBranch) + len(rightBranch) > args.min_points_needed_to_interpolate:
                        if len(leftBranch) > args.min_points_needed_to_interpolate // 2 and len(
                                rightBranch) > args.min_points_needed_to_interpolate // 2:
                            leftBranch = leftBranch[:args.min_points_needed_to_interpolate // 2]
                            rightBranch = rightBranch[:args.min_points_needed_to_interpolate // 2]
                        elif len(leftBranch) < len(rightBranch):
                            rightBranch = rightBranch[:(args.min_points_needed_to_interpolate - len(leftBranch))]
                        else:
                            leftBranch = leftBranch[:(args.min_points_needed_to_interpolate - len(rightBranch))]

                    # perform interpolation and save point value
                    combined = np.concatenate([leftBranch, rightBranch])
                    combined.sort()
                    if len(combined) > 0:
                        X = pandasInput[col].iloc[combined].index.values
                        Y = pandasInput[col].iloc[combined].values
                        Xhat = np.arange(hole[0], hole[1], 1)
                        try:
                            Yhat = np.interp(np.array(Xhat, dtype=int), np.array(X, dtype=int),
                                             np.array(Y, dtype=float))
                        except Exception:
                            print(X)
                            print(Y)
                            print(Xhat)
                            print(f"error processing interpolation of following points in {dataSetInput.filePath}")
                            continue

                        # optionally plot (for debugging)
                        # plt.cla()
                        # plt.scatter( X, Y )
                        # plt.scatter(Xhat, Yhat, c="red")
                        # plt.show()

                        # save interpolated points
                        for i in range(len(Yhat)):
                            pandasInput.loc[Xhat[i], col] = Yhat[i]
                            updatedColumnIndexes.append(Xhat[i])
                        pointsFilled += len(Yhat)
        updatedIndexes[col] = updatedColumnIndexes

    return pandasInput, updatedIndexes, pointsFilled
