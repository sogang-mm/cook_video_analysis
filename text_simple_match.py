# -*- coding: utf-8 -*-
def text_simple_match(data, db, topk):
    
    """
    1. data: list, 질의할 preprocessed scenetext 분석 결과, e.g) egg_text (= preprocess(data=result_30, score=0.9, video_name='egg', module_name='scenetxt'))
    2. db: pd.Dataframe, 매칭할 데이터 베이스 e.g) 식재료 데이터 베이스, df (= pd.read_csv('./Dataset/dataset/refined/ingredients.csv', encoding = 'cp949' ))
    3. topk: int, 몇 개의 빈도수 상위 태그를 뽑을 것인지 e.g) 5
    """

    result_dict={}
    
    # count the number of fail to match
    no_result = 0

    for query in data:

        # scenetxt_result <-> ingredient DB 
        search_result = list(filter(lambda x: x == query, db['ingr_ko']))

        if search_result: #exist

                if search_result[0] in result_dict:
                    result_dict[search_result[0]] += 1
                else:
                    result_dict[search_result[0]] = 1

        else: #empty
            no_result += 1
    
    #show the ratio of no_result
    #print(f'no_result_ratio : {round(no_result/len(data), 2)}')
    
    topk_tags = [label[0] for label in sorted(result_dict.items(), key = lambda x: -x[1])[:topk]]
    
    return topk_tags
