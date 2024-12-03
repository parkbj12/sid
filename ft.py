import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import openai
import chardet

# OpenAI API 키 설정
api_key = st.secrets["google_api_key"]
# CSV 파일 경로 설정
FILE_PATH = '서울시 생활체육포털(3만).csv'

@st.cache_data
def load_data(file_path):
    try:
        # 파일 인코딩 자동 감지
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # 파일 내용 분석
            encoding = result['encoding']
        
        # 감지된 인코딩으로 데이터 읽기
        data = pd.read_csv(file_path, encoding=encoding)
        return data
    
    except FileNotFoundError:
        st.error(f"'{file_path}' 경로에 파일이 없습니다.")
        return pd.DataFrame()
    except UnicodeDecodeError:
        st.error("파일 인코딩을 확인하세요. 자동 인코딩 감지에 실패했습니다.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"알 수 없는 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 데이터 로드
data = load_data(FILE_PATH)

# OpenAI 응답 생성 함수
def generate_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": user_input}],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()

# Streamlit 애플리케이션 UI 구성
st.title("서울시 피트니스 네트워크")
st.sidebar.header("메뉴")

menu = st.sidebar.selectbox(
    "페이지 선택",
    options=["홈", "구 별 데이터", "추천 시스템", "챗봇"]
)

if menu == "홈":
    st.subheader("서울에서 본인에게 맞는 체육생활을 찾아보세요!")
    st.markdown("**서울특별시 지역**을 선택하거나 검색하세요!")
# CSV 파일에서 데이터를 읽는 함수
def read_csv(file_path):
    # CSV 파일을 DataFrame으로 읽기
    df = pd.read_csv(file_path)
    return df

# 특정 구에 해당하는 데이터를 추출하는 함수
def get_district_data_from_csv(dataframe, district):
    # 구 이름으로 데이터를 필터링
    district_data = dataframe[dataframe['구명'].str.contains(district, na=False)]  # '구명'은 CSV 파일에 구 이름이 있는 열 이름이라고 가정
    if district_data.empty:
        return None  # 해당 구에 대한 데이터가 없으면 None 반환

    return district_data

# 이미지 맵을 포함한 HTML 코드
map_html = """
    <img src="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMzEwMDNfMjYy%2FMDAxNjk2MzI5NDU4Nzc2.aOCTdfY18de7L41hBjp9pYCZHF1wDKHaFaoaAZiVM8sg.3YLCSSm8RdNwOPhtFkSsgcQWy0Xsh-lwyfa6mSDTH40g.PNG.sunstory77%2Fimage.png&type=sc960_832" usemap="#seoul-map" style="max-width: 100%;">
    <map name="seoul-map">
        <area target="" alt="강서구" title="강서구" href="/강서구" coords="129,256,34" shape="circle">
        <area target="" alt="양천구" title="양천구" href="/양천구" coords="180,332,28" shape="circle">
        <area target="" alt="은평구" title="은평구" href="/은평구" coords="289,148,34" shape="circle">
        <area target="" alt="도봉구" title="도봉구" href="/도봉구" coords="438,95,23" shape="circle">
        <area target="" alt="노원구" title="노원구" href="/노원구" coords="501,136,24" shape="circle">
        <area target="" alt="강북구" title="강북구" href="/강북구" coords="416,140,21" shape="circle">
        <area target="" alt="중랑구" title="중랑구" href="/중랑구" coords="526,195,26" shape="circle">
        <area target="" alt="강동구" title="강동구" href="/강동구" coords="599,270,22" shape="circle">
        <area target="" alt="송파구" title="송파구" href="/송파구" coords="548,346,24" shape="circle">
        <area target="" alt="성북구" title="성북구" href="/성북구" coords="416,201,25" shape="circle">
        <area target="" alt="동대문구" title="동대문구" href="/동대문구" coords="466,226,24" shape="circle">
        <area target="" alt="광진구" title="광진구" href="/광진구" coords="517,283,24" shape="circle">
        <area target="" alt="종로구" title="종로구" href="/종로구" coords="356,222,24" shape="circle">
        <area target="" alt="서대문구" title="서대문구" href="/서대문구" coords="287,231,30" shape="circle">
        <area target="" alt="중구" title="중구" href="/중구" coords="382,267,24" shape="circle">
        <area target="" alt="성동구" title="성동구" href="/성동구" coords="446,279,25" shape="circle">
        <area target="" alt="마포구" title="마포구" href="/마포구" coords="277,285,24" shape="circle">
        <area target="" alt="용산구" title="용산구" href="/용산구" coords="357,312,26" shape="circle">
        <area target="" alt="강남구" title="강남구" href="/강남구" coords="469,369,30" shape="circle">
        <area target="" alt="서초구" title="서초구" href="/서초구" coords="403,389,28" shape="circle">
        <area target="" alt="동작구" title="동작구" href="/동작구" coords="323,361,26" shape="circle">
        <area target="" alt="영등포구" title="영등포구" href="/영등포구" coords="261,334,25" shape="circle">
        <area target="" alt="관악구" title="관악구" href="/관악구" coords="309,423,27" shape="circle">
        <area target="" alt="금천구" title="금천구" href="/금천구" coords="240,425,24" shape="circle">
        <area target="" alt="구로구" title="구로구" href="/구로구" coords="151,378,26" shape="circle">
    </map>
    """

# 각 구에 대한 세부 페이지 처리
if menu != "홈":
    st.subheader(f"{menu}에 대한 정보")
    st.write(f"{menu}에 대한 구체적인 정보를 여기에 표시합니다.")
# Streamlit 앱 설정
st.markdown(map_html, unsafe_allow_html=True)

# "구 별 데이터" 메뉴에 대한 처리
if menu == "구 별 데이터":
    st.subheader("서울시 구 별 프로그램 데이터")
    district = st.selectbox("지역을 선택하세요", options=data["지역구"].unique())
    search_query = st.text_input("검색어를 입력하세요")
    
    district_data = data[data["지역구"] == district]
    if search_query:
        district_data = district_data[
            district_data.apply(lambda row: search_query.lower() in row.astype(str).str.lower().to_string(), axis=1)
        ]
    
    if not district_data.empty:
        st.dataframe(district_data.iloc[:, 1:].reset_index(drop=True))
    else:
        st.warning("검색 결과가 없습니다.")

elif menu == "추천 시스템":
    st.subheader("추천 시스템")
    
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased').to(device)
        return tokenizer, model, device

    tokenizer, model, device = load_model()
    
    @st.cache_data
    def load_embedding_data():
        rec_data = pd.read_csv('recommendation_data_with_embeddings.csv', encoding='cp949')
        rec_data['embedding'] = rec_data['embedding'].apply(lambda x: np.array(list(map(float, x.split(',')))))
        return rec_data
    
    rec_data = load_embedding_data()

    def get_distilbert_embedding(text):
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def recommend_program(user_input, rec_data, top_n=5):
        user_embedding = get_distilbert_embedding(user_input)
        rec_data['similarity'] = rec_data['embedding'].apply(lambda x: cosine_similarity([x], user_embedding)[0][0])
        recommended = rec_data.sort_values(by='similarity', ascending=False).head(top_n)
        return recommended[['대상', '내용', '지역구', '장소', '전화번호', '기관홈페이지']]

    target = st.text_input("찾고 싶은 대상을 입력하세요 (ex. 어르신)")
    region = st.text_input("찾고 싶은 지역구를 입력하세요 (ex. 강서구)")

    if st.button("추천"):
        user_input = f"{region} {target}"
        recommendations = recommend_program(user_input, rec_data)
        st.dataframe(recommendations)
    

elif menu == "챗봇":
    st.subheader("🏋️GYM 챗봇🏋️")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_message = st.text_input("챗봇에게 질문하세요!", key="user_input")

    if st.button("전송"):
        if user_message:
            st.session_state["chat_history"].append({"role": "user", "content": user_message})
            response = generate_response(user_message)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.markdown(f"**사용자:** {message['content']}")
        else:
            st.markdown(f"**챗봇:** {message['content']}")
