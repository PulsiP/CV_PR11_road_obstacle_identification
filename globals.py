from collections import namedtuple


# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

CS_LABEL = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled",              0,      0, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle",            1,      0, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border",   2,      0, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi",             3,      0, "void", 0, False, True, (0, 0, 0)),
    Label("static",                 4,      0, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic",                5,      0, "void", 0, False, True, (111, 74, 0)),
    Label("ground",                 6,      0, "void", 0, False, True, (81, 0, 81)),
    Label("road",                   7,      1, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk",               8,      0, "flat", 1, False, False, (244, 35, 232)),
    Label("parking",                9,      0, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track",             10,     0, "flat", 1, False, True, (230, 150, 140)),
    Label("building",               11,     0, "construction", 2, False, False, (70, 70, 70)),
    Label("wall",                   12,     0, "construction", 2, False, False, (102, 102, 156)),
    Label("fence",                  13,     0, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail",             14,     2, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge",                 15,     0, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel",                 16,     0, "construction", 2, False, True, (150, 120, 90)),
    Label("pole",                   17,     3, "object", 3, False, False, (153, 153, 153)), # pole - polegroup
    Label("polegroup",              18,     3, "object", 3, False, True, (153, 153, 153)),  # pole - polegroup
    Label("traffic light",          19,     4, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign",           20,     5, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation",             21,     0, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain",                22,     0, "nature", 4, False, False, (152, 251, 152)),
    Label("sky",                    23,     0, "sky", 5, False, False, (70, 130, 180)),
    Label("person",                 24,     6, "human", 6, True, False, (220, 20, 60)),
    Label("rider",                  25,     7, "human", 6, True, False, (255, 0, 0)),
    Label("car",                    26,     8, "vehicle", 7, True, False, (0, 0, 142)), # car - license_plate
    Label("truck",                  27,     9, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus",                    28,     10, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan",                29,     11, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer",                30,     12, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train",                  31,     13, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle",             32,     14, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle",                33,     15, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate",          -1,     8, "vehicle", 7, False, True, (0, 0, 142)), # car - license_plate
]

CS_OBJECTS = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled",              0,      0, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle",            1,      0, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border",   2,      0, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi",             3,      0, "void", 0, False, True, (0, 0, 0)),
    Label("static",                 4,      0, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic",                5,      0, "void", 0, False, True, (111, 74, 0)),
    Label("ground",                 6,      0, "void", 0, False, True, (81, 0, 81)),
    Label("road",                   7,      0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk",               8,      0, "flat", 1, False, False, (244, 35, 232)),
    Label("parking",                9,      0, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track",             10,     0, "flat", 1, False, True, (230, 150, 140)),
    Label("building",               11,     0, "construction", 2, False, False, (70, 70, 70)),
    Label("wall",                   12,     0, "construction", 2, False, False, (102, 102, 156)),
    Label("fence",                  13,     0, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail",             14,     0, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge",                 15,     0, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel",                 16,     0, "construction", 2, False, True, (150, 120, 90)),
    Label("pole",                   17,     1, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup",              18,     1, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light",          19,     1, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign",           20,     1, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation",             21,     0, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain",                22,     0, "nature", 4, False, False, (152, 251, 152)),
    Label("sky",                    23,     0, "sky", 5, False, False, (70, 130, 180)),
    Label("person",                 24,     1, "human", 6, True, False, (220, 20, 60)),
    Label("rider",                  25,     1, "human", 6, True, False, (255, 0, 0)),
    Label("car",                    26,     1, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck",                  27,     1, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus",                    28,     1, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan",                29,     1, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer",                30,     1, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train",                  31,     1, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle",             32,     1, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle",                33,     1, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate",          -1,     1, "vehicle", 7, False, True, (0, 0, 142)), # remap -1 -> 19
]


CS_BLABEL = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled",              0,      1, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle",            1,      0, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border",   2,      0, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi",             3,      0, "void", 0, False, True, (0, 0, 0)),
    Label("static",                 4,      0, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic",                5,      1, "void", 0, False, True, (111, 74, 0)),
    Label("ground",                 6,      0, "void", 0, False, True, (81, 0, 81)),
    Label("road",                   7,      0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk",               8,      0, "flat", 1, False, False, (244, 35, 232)),
    Label("parking",                9,      0, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track",             10,     0, "flat", 1, False, True, (230, 150, 140)),
    Label("building",               11,     0, "construction", 2, False, False, (70, 70, 70)),
    Label("wall",                   12,     0, "construction", 2, False, False, (102, 102, 156)),
    Label("fence",                  13,     0, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail",             14,     0, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge",                 15,     0, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel",                 16,     0, "construction", 2, False, True, (150, 120, 90)),
    Label("pole",                   17,     1, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup",              18,     1, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light",          19,     1, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign",           20,     1, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation",             21,     0, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain",                22,     0, "nature", 4, False, False, (152, 251, 152)),
    Label("sky",                    23,     0, "sky", 5, False, False, (70, 130, 180)),
    Label("person",                 24,     1, "human", 6, True, False, (220, 20, 60)),
    Label("rider",                  25,     1, "human", 6, True, False, (255, 0, 0)),
    Label("car",                    26,     1, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck",                  27,     1, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus",                    28,     1, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan",                29,     1, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer",                30,     1, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train",                  31,     1, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle",             32,     1, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle",                33,     1, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate",          -1,     1, "vehicle", 7, False, True, (0, 0, 142)), # remap -1 -> 19
]

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

OBS_LABEL = [
    #       name                     id      trainId    hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,       0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  0 ,       0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  0 ,       0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  0 ,       0       , False        , True         , (  0,  0,  0) ),
    Label(  'background'           ,  0 ,       0       , False        , False        , (  0,  0,  0) ),
    Label(  'free'                 ,  1 ,       1       , False        , False        , (128, 64,128) ),
    Label(  '01'                   ,  2 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '02'                   ,  3 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '03'                   ,  4 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '04'                   ,  5 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '05'                   ,  6 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '06'                   ,  7 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '07'                   ,  8 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '08'                   ,  9 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '09'                   , 10 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '10'                   , 11 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '11'                   , 12 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '12'                   , 13 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '13'                   , 14 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '14'                   , 15 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '15'                   , 16 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '16'                   , 17 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '17'                   , 18 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '18'                   , 19 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '19'                   , 20 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '20'                   , 21 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '21'                   , 22 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '22'                   , 23 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '23'                   , 24 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '24'                   , 25 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '25'                   , 26 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '26'                   , 27 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '27'                   , 28 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '28'                   , 29 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '29'                   , 30 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '30'                   , 31 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '31'                   , 32 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '32'                   , 33 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '33'                   , 34 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '34'                   , 35 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '35'                   , 36 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '36'                   , 37 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '37'                   , 38 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '38'                   , 39 ,       0       , True         , False        , (  0,  0,  0) ),
    Label(  '39'                   , 40 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '40'                   , 41 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '41'                   , 42 ,       2       , True         , False        , (  0,  0,142) ),
    Label(  '42'                   , 43 ,       2       , True         , False        , (  0,  0,142) ),

]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# trainId → color
CS_LABEL2COLOR = {label.trainId: label.color for label in CS_LABEL if label.trainId != 0}
CS_LABEL2COLOR[0] = (0,0,0)
OBS_LABEL2COLOR = {label.trainId: label.color for label in OBS_LABEL if label.trainId != 0}
OBS_LABEL2COLOR[0] = (0,0,0)


CSB_LABEL2COLOR = {
    1 : (255,255,255) 
}

# color → trainId
CS_COLOR2LABEL = {label.color: label.trainId for label in CS_LABEL if label.trainId != 0}
CS_COLOR2LABEL[(0,0,0)] = 0
CSB_COLOR2LABEL = {label.color: label.trainId for label in CS_BLABEL if label.trainId != 0}
CSB_COLOR2LABEL[(0,0,0)] = 0
OBS_COLOR2LABEL = {label.color: label.trainId for label in OBS_LABEL if label.trainId != 0}
OBS_COLOR2LABEL[(0,0,0)] = 0

# Plus mapping

CS_PLUS = {label.color : label.trainId for label in CS_OBJECTS if label.trainId != 0}

if __name__ == "__main__":
    print(CS_LABEL2COLOR)
    print(CS_COLOR2LABEL)