""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    min_arr = min(arr)
    max_arr = max(arr)
    diff = max_arr - min_arr
    scalled_arr = (((x - min_arr)/diff) for x in arr)

    return scalled_arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)

