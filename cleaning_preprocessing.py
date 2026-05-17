import pandas as pd
import numpy as np

def preproccesing(df):

    #Splicing the variables into numerical and categorical and ordinal for each section of data
    ames = df.copy()
    y_cat = "HighSalePrice"
    y = "SalePrice"

    #Location and Land Varaibles
    location_land_categorical = ["MS_Zoning", "Neighborhood", "Lot_Config", "Street", "Alley", "Condition_1", "Condition_2"]
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
    ames["Alley"] = ames["Alley"].fillna("None")

    #Aggregating all the location_land vars
    location_land_all = location_land_categorical + location_land_numerical + location_land_ordinal

    #Building Type and Age
    building_type_numerical = ["Year_Built", "Year_Remod_Add","Mas_Vnr_Area"]
    building_type_categorical = ["Bldg_Type", "House_Style", "MS_SubClass", "Roof_Style", "Roof_Matl", "Exterior_1st", "Exterior_2nd", "Mas_Vnr_Type", "Foundation", "Heating", "Central_Air", "Electrical"]
    building_type_ordinal = ["Functional"]
    building_type_total = building_type_numerical + building_type_categorical + building_type_ordinal

    ames["Mas_Vnr_Type"] = ames["Mas_Vnr_Type"].fillna("None")
    ames["Mas_Vnr_Area"] = ames["Mas_Vnr_Area"].fillna(0)

    functional_map = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}
    ames["Functional"] = ames["Functional"].map(functional_map)

    #Quality and Condition
    quality_and_condition_numerical = ["Overall_Qual", "Overall_Cond"]
    quality_and_condition_ordinals =  ["Exter_Qual", "Exter_Cond", "Kitchen_Qual", "Heating_QC"]
    quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    for col in quality_and_condition_ordinals:
        ames[col] = ames[col].map(quality_map)

    quality_and_condition_total = quality_and_condition_numerical + quality_and_condition_ordinals

    #Size and Rooms
    size_and_rooms_numeric = [
        "Gr_Liv_Area", "X1st_Flr_SF", "X2nd_Flr_SF", "Total_Bsmt_SF", "Full_Bath", "Half_Bath", "Bedroom_AbvGr", "Kitchen_AbvGr", "TotRms_AbvGrd", "Low_Qual_Fin_SF", "Wood_Deck_SF",
            "Open_Porch_SF",
            "Enclosed_Porch",
            "X3Ssn_Porch",
            "Screen_Porch",
            "Pool_Area",
            "Fireplaces",
        ]
    
    size_and_rooms_ordinal = ["Fireplace_Qu", "Pool_QC", "Fence", "Paved_Drive"]

    ames["Fireplace_Qu"] = ames["Fireplace_Qu"].map(quality_map)
    ames["Fireplace_Qu"] = ames["Fireplace_Qu"].fillna(0)

    ames["Pool_QC"] = ames["Pool_QC"].map(quality_map)
    ames["Pool_QC"] = ames["Pool_QC"].fillna(0)

    ames["Total_Bsmt_SF"] = ames["Total_Bsmt_SF"].fillna(0)

    fence_map = {np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    ames["Fence"] = ames["Fence"].map(fence_map)
    paved_drive_map = {"Y": 2, "P": 1, "N": 0}
    ames["Paved_Drive"] = ames["Paved_Drive"].map(paved_drive_map)

    size_and_rooms_total = size_and_rooms_numeric + size_and_rooms_ordinal

    #Basement and Garage
    basement_garage_numerical = ["Garage_Cars", "Garage_Area", "Garage_Yr_Blt", "BsmtFin_SF_1", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Bsmt_Full_Bath", "Bsmt_Half_Bath"]


    basement_garage_categorical = ["Garage_Type"]
    basement_garage_ordinal = ["Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure", "BsmtFin_Type_1", "Garage_Finish", "BsmtFin_Type_2", "Garage_Qual", "Garage_Cond"]

    #We fill the numerical null values with 0 and the categorical and ordinal with 0 assuming that null values correspond to the lack of a basement
    ames["Garage_Cars"] = ames["Garage_Cars"].fillna(0)
    ames["BsmtFin_SF_1"] = ames["BsmtFin_SF_1"].fillna(0)
    ames["BsmtFin_SF_2"] = ames["BsmtFin_SF_2"].fillna(0)
    ames[["Bsmt_Unf_SF", "Bsmt_Full_Bath", "Bsmt_Half_Bath"]] = ames[["Bsmt_Unf_SF", "Bsmt_Full_Bath", "Bsmt_Half_Bath"]].fillna(0)
    ames["Garage_Yr_Blt"] = ames["Garage_Yr_Blt"].fillna(0)
    ames["Garage_Area"] = ames["Garage_Area"].fillna(0)
    ames["Garage_Type"] = ames["Garage_Type"].fillna("No_Garage")

    for col in basement_garage_ordinal:
        ames[col] = ames[col].fillna("None")

    basement_garage_maps = {
        "Bsmt_Qual": {"None": 0, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Bsmt_Cond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Garage_Qual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Garage_Cond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Bsmt_Exposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
        "BsmtFin_Type_1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
        "Garage_Finish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
    }

    for col, mapping in basement_garage_maps.items():
        ames[col] = ames[col].map(mapping)

    basement_garage_total = basement_garage_numerical + basement_garage_categorical + basement_garage_ordinal

    #Sale Variables

    ames["Mo_Sold_Sin"] = np.sin(2 * np.pi * ames["Mo_Sold"] / 12)
    ames["Mo_Sold_Cos"] = np.cos(2 * np.pi * ames["Mo_Sold"] / 12)
    ames = ames.drop(columns=["Mo_Sold"])

    sale_variables_numeric = ["Yr_Sold", "Mo_Sold_Sin", "Mo_Sold_Cos", "Misc_Val"]
    sale_categorical = ["Sale_Type", "Sale_Condition", "Misc_Feature"]

    ames["Misc_Feature"] = ames["Misc_Feature"].fillna("No Feature")

    sale_total = sale_variables_numeric + sale_categorical

    #Aggregting all categorical variables
    categorical_vars = sale_categorical + basement_garage_categorical + building_type_categorical + location_land_categorical 
    numerical_vars = sale_variables_numeric + basement_garage_numerical + building_type_numerical + location_land_numerical + quality_and_condition_numerical + size_and_rooms_numeric
    ordinal_vars = basement_garage_ordinal + location_land_ordinal+ quality_and_condition_ordinals + size_and_rooms_ordinal + building_type_ordinal
    total_columns = categorical_vars + numerical_vars + ordinal_vars
    total_columns += [y_cat, y]
    ames = ames[total_columns]

    column_breakdown = {
    "location_land": {
        "categorical": location_land_categorical,
        "numerical": location_land_numerical,
        "ordinal": location_land_ordinal
    },

    "building_type_age": {
        "categorical": building_type_categorical,
        "numerical": building_type_numerical,
        "ordinal": building_type_ordinal
    },

    "quality_condition": {
        "categorical": [],
        "numerical": quality_and_condition_numerical,
        "ordinal": quality_and_condition_ordinals
    },

    "size_rooms": {
        "categorical": [],
        "numerical": size_and_rooms_numeric,
        "ordinal": size_and_rooms_ordinal
    },

    "basement_garage": {
        "categorical": basement_garage_categorical,
        "numerical": basement_garage_numerical,
        "ordinal": basement_garage_ordinal
    },

    "sale": {
        "categorical": sale_categorical,
        "numerical": sale_variables_numeric,
        "ordinal": [],
    }
    }

    return ames, column_breakdown








