import openai
import re
import time
from tools import remove_brackets


def strip_func(data):

    # 정규식을 이용해 탄수화물, 지방, 단백질 수치 추출
    carb_match = re.search(r'Carbonhydrate: (\d+)g', data)
    fat_match = re.search(r'Fat: (\d+)g', data)
    protein_match = re.search(r'Protein: (\d+)g', data)
    cuisine_match = re.search(r'Cuisine: (\w+)', data)

    # 추출된 값을 숫자로 변환
    carb = int(carb_match.group(1)) if carb_match else 0
    fat = int(fat_match.group(1)) if fat_match else 0
    protein = int(protein_match.group(1)) if protein_match else 0
    cuisine = cuisine_match.group(1) if cuisine_match else "Unknown"

    # 음식 종류를 One-Hot Encoding으로 변환하는 함수
    def cuisine_to_onehot(cuisine):
        cuisines = ['Korean', 'Western', 'Chinese', 'Japanese', 'Other']
        one_hot = [1 if cuisine == c else 0 for c in cuisines]
        return one_hot

    # 음식 종류 One-Hot Encoding 변환
    cuisine_onehot = cuisine_to_onehot(cuisine)

    # 최종 결과
    result = [carb, fat, protein, cuisine_onehot]

    return result
def analyze_food_with_gpt(menu):
    
    
    prompt = f"Give me the carbohydrate, fat, and protein content of {menu} for one person by the number, and also specify the cuisine type (e.g., Korean, Japanese, American, etc.) Don't answer the other infomation."
    
    # response = openai.Completion.create(
    #     engine="gpt-4o-mini",  
    #     prompt=prompt,
    #     max_tokens=100
    # )
    
    # data = response.choices[0].text.strip()
    dummy = "Carbonhydrate: 12g\nFat: 7g\nProtein: 22g\nCuisine: Korean"
    data = dummy
    result = strip_func(data)
    return result


def analyze_food_with_crawling(menu):
    menu = remove_brackets(menu)
    import requests
    from bs4 import BeautifulSoup
    if '밥' in menu or (menu.endswith('김치') and len(menu) == 2 + len('김치')):
        print(menu)
        return
    # 메뉴 검색 URL
    base_url = "https://www.fatsecret.kr/%EC%B9%BC%EB%A1%9C%EB%A6%AC-%EC%98%81%EC%96%91%EC%86%8C/search?q={}"

    # URL에 메뉴를 삽입하여 완성
    search_url = base_url.format(menu)
    
    # HTTP 요청
    response = requests.get(search_url)

    # 페이지를 파싱
    soup = BeautifulSoup(response.text, 'html.parser')
   
    first_result = soup.find('div', class_='smallText greyText greyLink')
    time.sleep(1)
    # 영양 성분 정보를 추출
    if first_result:
        nutrition_info = first_result.text.strip()
        
        # 영양 정보 정규 표현식으로 추출
        calorie_match = re.search(r'칼로리: (\d+)kcal', nutrition_info)
        fat_match = re.search(r'지방: ([\d.]+)g', nutrition_info)
        carb_match = re.search(r'탄수화물: ([\d.]+)g', nutrition_info)
        protein_match = re.search(r'단백질: ([\d.]+)g', nutrition_info)
        
        # 값이 존재하면 추출
        calorie = int(calorie_match.group(1)) if calorie_match else 0
        fat = float(fat_match.group(1)) if fat_match else 0.0
        carbohydrate = float(carb_match.group(1)) if carb_match else 0.0
        protein = float(protein_match.group(1)) if protein_match else 0.0
        
        print([carbohydrate, fat, protein, [1,0,0,0,0]])
        return [carbohydrate, fat, protein, [1,0,0,0,0]]
    else:
        print(f"영양 정보를 찾을 수 없습니다.{menu}")
      
        return [0,0,0,[0,0,0,0,1]]
    

def main():
    # # 테스트: 'hamburger' 메뉴에 대한 영양 정보와 음식 종류 분석
    # menu = '오징어볶음'
    # result = analyze_food_with_gpt(menu)
    # print(result)
    analyze_food_with_crawling('김치')
if __name__ == "__main__":
    main()