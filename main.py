from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import UUID, uuid4

import pandas as pd
import pulp as pl
import os

app = FastAPI()

# Get the path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
CSV_PATH = os.path.join(BASE_DIR, 'recommendation_system', 'df_p2.csv')

class NutrientConfig(BaseModel):
    cal_lo: int = 2000  # Default value
    cal_up: int = 2500  # Default value
    pro_lo: int = 50    # Default value
    pro_up: int = 150   # Default value
    fat_lo: int = 20    # Default value
    fat_up: int = 70    # Default value
    sod_lo: int = 1000  # Default value
    sod_up: int = 2300  # Default value
class RecipeRequest(BaseModel):
    number_of_dishes: int = 5
    number_of_candidates: int =10
    nut_conf: NutrientConfig = NutrientConfig()    


class PersonalInformation(BaseModel):
    number_of_dishes: int = 5
    number_of_candidates: int = 10
    current_weight: int = 60
    desired_weight: int = 65
    height: int = 175
    age: int = 22
    gender: str = 'Male'
    activity_level: str = 'moderate'


def calculate_bmr(weight, height, age, gender):
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)
#ước lượng rằng 1 kg cân nặng tương đương với khoảng 7700 kcal (calories).
def calculate_time_to_goal(current_weight, desired_weight, calorie_change_per_day):
    weight_change = desired_weight - current_weight  # Dương nếu tăng cân, âm nếu giảm cân
    total_calories_needed = weight_change * 7700  # Tổng calo cần để đạt mục tiêu
    time_in_days = total_calories_needed / calorie_change_per_day
    return abs(time_in_days)  # Trả về giá trị tuyệt đối để có số ngày dương

def calculate_calorie_range(tdee, calorie_change_per_day):
    # Tính lượng calo tối thiểu và tối đa cần tiêu thụ mỗi ngày
    min_calories = tdee - calorie_change_per_day  # Thâm hụt calo (giảm cân)
    max_calories = tdee + calorie_change_per_day  # Thặng dư calo (tăng cân)
    return min_calories, max_calories


@app.get("/")
def home():
    return "Hello"

def recipe_recommend(df, number_of_dishes, number_of_candidates, nut_conf):
    tmp = df.copy()
    candidates_list = []

    for i in range(0, number_of_candidates):
        m = pl.LpProblem(sense=pl.LpMaximize)
        tmp['v'] = [pl.LpVariable(f'x{i}', cat=pl.LpBinary) for i in range(len(tmp))]

        m += pl.lpDot(tmp["rating"], tmp["v"])
        m += pl.lpSum(tmp["v"]) <= number_of_dishes
        m += pl.lpDot(tmp["calories"], tmp["v"]) >= nut_conf["cal_lo"]
        m += pl.lpDot(tmp["calories"], tmp["v"]) <= nut_conf["cal_up"]
        m += pl.lpDot(tmp["protein"], tmp["v"]) >= nut_conf["pro_lo"]
        m += pl.lpDot(tmp["protein"], tmp["v"]) <= nut_conf["pro_up"]
        m += pl.lpDot(tmp["fat"], tmp["v"]) >= nut_conf["fat_lo"]
        m += pl.lpDot(tmp["fat"], tmp["v"]) <= nut_conf["fat_up"]
        m += pl.lpDot(tmp["sodium"], tmp["v"]) >= nut_conf["sod_lo"]
        m += pl.lpDot(tmp["sodium"], tmp["v"]) <= nut_conf["sod_up"]

        m.solve(pl.PULP_CBC_CMD(msg=0, options=['maxsol 1']))

        if m.status == 1:
            tmp['val'] = tmp["v"].apply(lambda x: pl.value(x))
            ret = tmp.query('val==1')["title"].values
            candidates_list.append(ret)
            tmp = tmp.query('val==0')

    return candidates_list

# Load the DataFrame (df_p2)
df_p2 = pd.read_csv(CSV_PATH)  # Adjust the path to your actual CSV file

@app.post('/recommend')
def recommend(
    request: PersonalInformation = PersonalInformation(),
    season: str = Query("summer", description="Filter by season (summer, winter)"),
    meal_type: str = Query("breakfast", description="Filter by meal type (breakfast, low_cal)"),
    quick_recipe: bool = Query(False, description="Filter by quick recipe (few ingredients and directions)")
):
    try:
        df_filtered = df_p2.copy()

        # Apply filters based on query parameters
        if season:
            if season.lower() == "summer":
                df_filtered = df_filtered.query("summer==1")
            elif season.lower() == "winter":
                df_filtered = df_filtered.query("winter==1")

        if meal_type:
            if meal_type.lower() == "breakfast":
                df_filtered = df_filtered.query("breakfast==1")
            elif meal_type.lower() == "low_cal":
                df_filtered = df_filtered[df_filtered["low cal"] == 1]

        if quick_recipe:
            df_filtered = df_filtered.query("len_ingredients <= 9 and len_directions <= 3")

        # Tính BMR và TDEE dựa trên cân nặng hiện tại
        bmr = calculate_bmr(request.current_weight, request.height, request.age, request.gender)
        tdee = calculate_tdee(bmr, request.activity_level)

        # Lượng calo thặng dư hoặc thiếu hụt mỗi ngày
        calorie_change_per_day = 1500  # Nếu muốn giảm cân, đặt giá trị dương cho thiếu hụt calo; nếu tăng cân thì đặt giá trị thặng dư calo

        # Tính thời gian để đạt mục tiêu
        days_to_goal = calculate_time_to_goal(request.current_weight, request.desired_weight, calorie_change_per_day)

        # Tính lượng calo tối thiểu và tối đa cần tiêu thụ mỗi ngày
        min_calories, max_calories = calculate_calorie_range(tdee, calorie_change_per_day)

        nutrient  =  NutrientConfig()
        nutrient.cal_lo = min_calories                     
        nutrient.cal_up = max_calories
        # Call the recipe_recommend function with the filtered DataFrame
        recommendations = recipe_recommend(df_filtered, request.number_of_dishes, request.number_of_candidates, nutrient.dict())
        return {"recommendations": [list(rec) for rec in recommendations]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)