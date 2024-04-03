category_map={
    "bathtub":0,
    "bed":1,
    "cabinet":2,
    "chair":3,
    "dishwasher":4,
    "fireplace":5,
    "oven":6,
    "refrigerator":7,
    "shelf":8,
    "sink":9,
    "sofa":10,
    "stool":11,
    "stove":12,
    "table":13,
    "toilet":14,
    "washer":15
}

category_map_from_synthetic={
    "03001627":0,
    "future_chair":0,
    "ABO_chair":0,
    "arkit_chair":0,
    "future_stool":0,
    "arkit_stool":0,

    "04256520":1,
    "future_sofa":1,
    "ABO_sofa":1,
    "arkit_sofa":1,

    "04379243":2,
    "ABO_table":2,
    "future_table":2,
    "arkit_table":2,

    "02933112":3,
    "future_cabinet":3,
    "ABO_cabinet":3,
    "arkit_cabinet":3,
    "arkit_oven":3,
    "arkit_refrigerator":3,
    "arkit_dishwasher":3,
    "03207941":3,

    "02818832":4,
    "future_bed":4,
    "ABO_bed":4,
    "arkit_bed":4,

    "02871439":5,
    "future_shelf":5,
    "ABO_shelf":5,
    "arkit_shelf":5,

}

synthetic_category_combined={
    "sofa":["future_sofa","ABO_sofa","04256520"],
    "chair":["03001627","future_chair","ABO_chair",
            "future_stool"],
    "table":[
        "04379243",
        "future_table",
        "ABO_table",
    ],
    "cabinet":["02933112","03207941","future_cabinet","ABO_cabinet"],
    "bed":["02818832","future_bed","ABO_bed"],
    "shelf":["02871439","future_shelf","ABO_shelf"],
    "all":["future_sofa","ABO_sofa","04256520",
           "03001627", "future_chair", "ABO_chair",
           "future_stool","04379243","future_table",
           "ABO_table","02933112","03207941","future_cabinet","ABO_cabinet",
           "02818832","future_bed","ABO_bed",
           "02871439","future_shelf","ABO_shelf"
           ]
}

synthetic_arkit_category_combined={
    "sofa":["future_sofa","ABO_sofa","04256520","arkit_sofa"],
    "chair":["03001627","future_chair","ABO_chair",
            "future_stool","arkit_chair","arkit_stool"],
    "table":["04379243","ABO_table","future_table","arkit_table"],
    "cabinet":["02933112","03207941","future_cabinet","ABO_cabinet","arkit_cabinet","arkit_stove","arkit_washer","arkit_dishwasher","arkit_refrigerator","arkit_oven"],
    "bed":["02818832","future_bed","ABO_bed","arkit_bed"],
    "shelf":["02871439","future_shelf","ABO_shelf","arkit_shelf"],
    "all":[
        "future_sofa","ABO_sofa","04256520","arkit_sofa",
        "03001627","future_chair","ABO_chair",
        "future_stool","arkit_chair","arkit_stool",
        "04379243","ABO_table","future_table","arkit_table",
        "02933112","03207941","future_cabinet","ABO_cabinet","arkit_cabinet","arkit_dishwasher","arkit_refrigerator","arkit_oven",
        "02818832","future_bed","ABO_bed","arkit_bed",
        "02871439","future_shelf","ABO_shelf","arkit_shelf"
    ]
}

arkit_category={
    "chair":["arkit_chair","arkit_stool"],
    "sofa":["arkit_sofa"],
    "table":["arkit_table"],
    "cabinet":["arkit_cabinet","arkit_dishwasher","arkit_refrigerator","arkit_oven"],
    "bed":["arkit_bed"],
    "shelf":["arkit_shelf"],
    "all":["arkit_chair","arkit_stool",
           "arkit_sofa","arkit_table",
           "arkit_cabinet","arkit_dishwasher","arkit_refrigerator","arkit_oven",
           "arkit_bed",
           "arkit_shelf"],
}