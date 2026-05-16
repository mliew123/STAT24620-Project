import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def preproccesing(df):

    #Splicing the variables into numerical and categorical and ordinal for each section of data
    ames = df.copy()

    #Location and Land Varaibles
    location_land_categorical = ["MS_Zoning", "Neighborhood", "Lot_Config"]
    location_land_numerical = ["Lot_Area", "Lot_Frontage"]
    location_land_ordinal = ["Lot_Shape", "Land_Contour", "Land_Slope"]

    maps_location_land = {
        "Lot_Shape": {"IR3": 1, "IR2": 2, "IR1": 3,"Reg": 4},
        "Land_Contour": {"Lvl": 3, "HLS": 2, "Bnk": 2, "Low": 1},
        "Land_Slope": {"Sev": 0, "Mod": 1, "Gtl": 2}
    }

    for col, mapping in maps_location_land.items():
        ames[col] = ames[col].map(mapping)

    #Filling in relavant missing values. A property must have a Lot frontage so we fill the few missing vals with the median
    ames["Lot_Frontage"] = df["Lot_Frontage"].fillna(df["Lot_Frontage"].median())

    #Aggregating all the location_land vars
    location_land_all = location_land_categorical + location_land_numerical + location_land_ordinal

    #Building Type and Age
    building_type_numerical = ["Year_Built", "Year_Remod_Add"]
    building_type_categorical = ["Bldg_Type", "House_Style", "MS_SubClass"]

    building_type_total = building_type_numerical + building_type_categorical

    #Quality and Condition
    quality_and_condition_numerical = ["Overall_Qual", "Overall_Cond"]
    quality_and_condition_ordinals =  ["Exter_Qual", "Exter_Cond", "Kitchen_Qual", "Heating_QC"]
    quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    for col in quality_and_condition_ordinals:
        ames[col] = ames[col].map(quality_map)

    quality_and_condition_total = quality_and_condition_numerical + quality_and_condition_ordinals


    #Size and Rooms
    size_and_rooms_numeric = ["Gr_Liv_Area", "X1st_Flr_SF", "X2nd_Flr_SF", "Total_Bsmt_SF", "Full_Bath", "Half_Bath", "Bedroom_AbvGr", "Kitchen_AbvGr", "TotRms_AbvGrd"]
    ames["Total_Bsmt_SF"] = ames["Total_Bsmt_SF"].fillna(0)

    size_and_rooms_total = size_and_rooms_numeric

    #Basement and Garage
    basement_garage_numerical = ["Garage_Cars", "Garage_Area"]
    basement_garage_categorical = ["Garage_Type"]
    basement_garage_ordinal = ["Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure", "BsmtFin_Type_1", "Garage_Finish"]

    #We fill the numerical null values with 0 and the categorical and ordinal with 0 assuming that null values correspond to the lack of a basement
    ames["Garage_Cars"] = ames["Garage_Cars"].fillna(0)
    ames["Garage_Area"] = ames["Garage_Area"].fillna(0)
    ames["Garage_Type"] = ames["Garage_Type"].fillna("No_Garage")

    for col in basement_garage_ordinal:
        ames[col] = ames[col].fillna("None")

    basement_garage_maps = {
        "Bsmt_Qual": {"None": 0, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Bsmt_Cond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Bsmt_Exposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
        "BsmtFin_Type_1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
        "Garage_Finish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
    }

    for col, mapping in basement_garage_maps.items():
        ames[col] = ames[col].map(mapping)

    basement_garage_total = basement_garage_numerical + basement_garage_categorical + basement_garage_ordinal

    #Sale Variables
    sale_variables_numeric = ["Yr_Sold"]
    sale_cyclical = ["Mo_Sold"]
    sale_categorical = ["Sale_Type", "Sale_Condition"]

    sale_total = sale_variables_numeric + sale_cyclical + sale_categorical

    ames["Mo_Sold"] = np.sin(2 * np.pi * ames["Mo_Sold"] / 12)

    #Aggregting all categorical variables
    categorical_vars = sale_categorical + basement_garage_categorical + building_type_categorical + location_land_categorical 
    numerical_vars = sale_variables_numeric + basement_garage_numerical + building_type_numerical + location_land_numerical + quality_and_condition_numerical + size_and_rooms_numeric
    ordinal_vars = basement_garage_ordinal + location_land_ordinal+ quality_and_condition_ordinals
    total_columns = categorical_vars + numerical_vars + ordinal_vars
    ames = ames[total_columns]
    ames = pd.get_dummies(ames, columns = categorical_vars, drop_first=True)

    column_breakdown = {
    "location_land": {
        "categorical": ["MS_Zoning", "Neighborhood", "Lot_Config"],
        "numerical": ["Lot_Area", "Lot_Frontage"],
        "ordinal": ["Lot_Shape", "Land_Contour", "Land_Slope"]
    },

    "building_type_age": {
        "categorical": ["Bldg_Type", "House_Style", "MS_SubClass"],
        "numerical": ["Year_Built", "Year_Remod_Add"],
        "ordinal": []
    },

    "quality_condition": {
        "categorical": [],
        "numerical": [],
        "ordinal": [
            "Overall_Qual", "Overall_Cond",
            "Exter_Qual", "Exter_Cond",
            "Kitchen_Qual", "Heating_QC"
        ]
    },

    "size_rooms": {
        "categorical": [],
        "numerical": [
            "Gr_Liv_Area", "X1st_Flr_SF", "X2nd_Flr_SF",
            "Total_Bsmt_SF", "Full_Bath", "Half_Bath",
            "Bedroom_AbvGr", "Kitchen_AbvGr", "TotRms_AbvGrd"
        ],
        "ordinal": []
    },

    "basement_garage": {
        "categorical": ["Garage_Type"],
        "numerical": ["Garage_Cars", "Garage_Area"],
        "ordinal": [
            "Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure",
            "BsmtFin_Type_1", "Garage_Finish"
        ]
    },

    "sale": {
        "categorical": ["Sale_Type", "Sale_Condition"],
        "numerical": ["Yr_Sold"],
        "ordinal": [],
        "cyclical": ["Mo_Sold"]
    }
    }

    return ames, column_breakdown








