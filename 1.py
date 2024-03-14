import requests

# Replace with your actual API key
API_KEY = "your_api_key_here"
API_ID = "your_api_id_here"

def get_food_nutrition(food_name):
    """
    Retrieves detailed nutritional information for a given food item using the Edamam Nutrition Analysis API.

    Args:
        food_name (str): The name of the food item.

    Returns:
        dict: A dictionary containing the nutritional information for the food item.
    """
    url = f"https://api.edamam.com/api/nutrition-data"
    params = {
        "app_id": API_ID,
        "app_key": API_KEY,
        "ingr": food_name
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        nutrition_data = data.get("totalNutrients", {})
        return nutrition_data
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return {}

# Example usage
food_name = "bulgogi"
nutrition_data = get_food_nutrition(food_name)

if nutrition_data:
    print(f"Nutritional information for {food_name}:")
    for nutrient, data in nutrition_data.items():
        print(f"{nutrient}: {data['label']} - {data['quantity']} {data['unit']}")
else:
    print("Failed to retrieve nutritional information.")