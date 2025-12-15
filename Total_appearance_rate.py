# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager, rc
import platform

st.set_page_config(page_title="출현율 계산 프로그램", layout="wide")

# 한글 폰트 설정
def set_korean_font():
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
    elif platform.system() == 'Darwin':  # macOS
        font_name = 'AppleGothic'
    else:  # Linux (Streamlit Cloud)
        font_name = 'NanumGothic'
        # 폰트 캐시 재생성
        import matplotlib.font_manager as fm
        fm._load_fontmanager(try_read_cache=False)

    try:
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        # 폰트 로드 실패 시 기본 폰트 사용
        print(f"Font loading error: {e}")
        pass

def get_direction_16(angle):
    """각도를 16방위로 변환"""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # 0~360 범위로 정규화
    angle = angle % 360

    # 16방위로 변환
    # N: 348.75~11.25, NNE: 11.25~33.75, NE: 33.75~56.25, ...
    idx = int((angle + 11.25) / 22.5) % 16
    return directions[idx]

def create_wind_rose(df_work, speed_bins, labels, data_type):
    """장미도 그래프 생성"""
    set_korean_font()

    # 16방위
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # 각 방향의 각도 (라디안)
    angles = np.arange(0, 360, 22.5) * np.pi / 180

    # 방향별, 속도구간별 빈도 계산
    direction_speed_counts = df_work.groupby(['direction', 'speed_bin']).size().unstack(fill_value=0)
    direction_speed_counts = direction_speed_counts.reindex(directions, fill_value=0)

    # 퍼센트로 변환
    total = len(df_work)
    direction_speed_pct = (direction_speed_counts / total * 100)

    # 그래프 생성
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')

    # 색상 설정 (속도 구간별)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(labels)))

    # 막대 너비
    width = 2 * np.pi / 16

    # 각 속도 구간별로 누적 막대 그래프
    bottom = np.zeros(16)

    for i, speed_label in enumerate(labels):
        if speed_label in direction_speed_pct.columns:
            values = direction_speed_pct[speed_label].values
            bars = ax.bar(angles, values, width=width, bottom=bottom,
                         color=colors[i], label=speed_label, alpha=0.8, edgecolor='white')
            bottom += values

    # 방향 레이블 설정
    ax.set_xticks(angles)
    ax.set_xticklabels(directions, fontsize=12)

    # 0도를 북쪽(위)으로 설정
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # 시계방향

    # 그리드 설정
    ax.set_ylim(0, bottom.max() * 1.1)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 범례
    ax.legend(title=f'{data_type} 속도 구간', loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # 제목
    plt.title(f'{data_type} 장미도 (Wind Rose)\n출현율 (%)', fontsize=16, pad=20)

    return fig

def create_sample_data():
    """예시 데이터 생성"""
    np.random.seed(42)

    # 2023년 1월 데이터 (1000개)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')

    # Wind 데이터 생성 (특정 방향에 편중되도록)
    angles = []
    speeds = []

    for _ in range(1000):
        # 주요 풍향: N, NE, E (0, 45, 90도 근처)
        main_direction = np.random.choice([0, 45, 90, 180, 270])
        angle = main_direction + np.random.normal(0, 15)  # 주변에 분산
        angle = angle % 360

        # 속도는 0~20 사이, 평균 6 정도
        speed = abs(np.random.normal(6, 3))

        angles.append(angle)
        speeds.append(speed)

    df = pd.DataFrame({
        'Year': dates.year,
        'Month': dates.month,
        'Day': dates.day,
        'Hour': dates.hour,
        'Speed': speeds,
        'Direction': angles
    })

    return df

def main():
    st.title("🌊 출현율 계산 프로그램")

    # 사이드바 - 설정
    st.sidebar.header("⚙️ 설정")

    # 데이터 타입 선택
    data_type = st.sidebar.radio(
        "데이터 타입",
        ["Wind", "Wave", "Current"],
        horizontal=True
    )

    # 예시 데이터 사용 옵션
    use_sample = st.sidebar.checkbox("예시 데이터 사용 (UI 테스트용)", value=True)

    # 파일 업로드
    st.header("1️⃣ 데이터 파일 업로드")

    if use_sample:
        st.info("🔍 예시 데이터를 사용 중입니다. UI 테스트용으로 1000개의 샘플 데이터가 로드되었습니다.")
        df = create_sample_data()
        uploaded_file = "sample"
    else:
        uploaded_file = st.file_uploader(
            "Excel 또는 CSV 파일을 선택하세요",
            type=['xlsx', 'xls', 'csv'],
            help="Wind, Wave, Current 데이터가 포함된 파일을 업로드하세요"
        )

    if uploaded_file is not None:
        try:
            # 파일 로드 (예시 데이터가 아닌 경우만)
            if uploaded_file != "sample":
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                else:
                    df = pd.read_excel(uploaded_file)

            st.success(f"✅ 파일 로드 완료! (총 {len(df):,}개 행)")

            # 데이터 미리보기
            with st.expander("📊 데이터 미리보기 (처음 5행)"):
                st.dataframe(df.head(), use_container_width=True)

            columns = list(df.columns)

            # 열 선택 섹션
            st.header("2️⃣ 데이터 열 선택")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📅 날짜 열")
                date_option = st.radio(
                    "날짜 형식 선택",
                    ["단일 날짜 열", "년/월/일 분리"],
                    index=1 if use_sample else 0,
                    help="데이터의 날짜 형식에 맞게 선택하세요"
                )

                if date_option == "단일 날짜 열":
                    date_col = st.selectbox("날짜 열", [''] + columns, key='date_col')
                    year_col = month_col = day_col = None
                else:
                    # 예시 데이터인 경우 자동 선택
                    year_default = columns.index('Year') if use_sample and 'Year' in columns else 0
                    month_default = columns.index('Month') if use_sample and 'Month' in columns else 0
                    day_default = columns.index('Day') if use_sample and 'Day' in columns else 0

                    year_col = st.selectbox("년 열", [''] + columns, index=year_default + 1 if use_sample else 0, key='year_col')
                    month_col = st.selectbox("월 열", [''] + columns, index=month_default + 1 if use_sample else 0, key='month_col')
                    day_col = st.selectbox("일 열", [''] + columns, index=day_default + 1 if use_sample else 0, key='day_col')
                    date_col = None

            with col2:
                st.subheader("📏 속도 & 각도 열")
                # 예시 데이터인 경우 자동 선택
                speed_default = columns.index('Speed') if use_sample and 'Speed' in columns else 0
                angle_default = columns.index('Direction') if use_sample and 'Direction' in columns else 0

                speed_col = st.selectbox("속도 열", columns, index=speed_default, key='speed_col')
                angle_col = st.selectbox("각도 열", columns, index=angle_default, key='angle_col')

            # 출력 테이블 구조
            st.header("3️⃣ 출력 테이블 구조")

            col1, col2 = st.columns(2)

            with col1:
                row_choice = st.radio(
                    "행 (Row)",
                    ["날짜", "속도", "각도(방향)"],
                    index=2,
                    help="테이블의 행에 표시할 항목을 선택하세요"
                )

            with col2:
                col_choice = st.radio(
                    "열 (Column)",
                    ["날짜", "속도", "각도(방향)"],
                    index=1,
                    help="테이블의 열에 표시할 항목을 선택하세요"
                )

            # 속도 빈도 설정
            st.header("4️⃣ 속도 빈도 설정")
            speed_bins_input = st.text_input(
                "속도 구간 (쉼표로 구분)",
                value="0, 2, 4, 6, 8, 10, 15, 20",
                help="예: 0, 2, 4, 6, 8, 10, 15, 20"
            )

            # 16방위 정보 표시
            with st.expander("ℹ️ 16방위 정보"):
                st.info("""
                **16방위 범위:**
                - N: 348.75~11.25°
                - NNE: 11.25~33.75°
                - NE: 33.75~56.25°
                - ENE: 56.25~78.75°
                - E: 78.75~101.25°
                - ESE: 101.25~123.75°
                - SE: 123.75~146.25°
                - SSE: 146.25~168.75°
                - S: 168.75~191.25°
                - SSW: 191.25~213.75°
                - SW: 213.75~236.25°
                - WSW: 236.25~258.75°
                - W: 258.75~281.25°
                - WNW: 281.25~303.75°
                - NW: 303.75~326.25°
                - NNW: 326.25~348.75°
                """)

            # 계산 버튼
            st.header("5️⃣ 출현율 계산")

            if st.button("🚀 출현율 계산 실행", type="primary", use_container_width=True):

                # 검증
                row_map = {"날짜": "date", "속도": "speed", "각도(방향)": "direction"}
                col_map = {"날짜": "date", "속도": "speed", "각도(방향)": "direction"}

                if row_map[row_choice] == col_map[col_choice]:
                    st.error("❌ 행과 열은 서로 다른 항목을 선택해야 합니다!")
                    return

                # 날짜 검증
                if date_option == "단일 날짜 열" and not date_col:
                    st.error("❌ 날짜 열을 선택하세요!")
                    return
                elif date_option == "년/월/일 분리" and (not year_col or not month_col or not day_col):
                    st.error("❌ 년, 월, 일 열을 모두 선택하세요!")
                    return

                if not speed_col or not angle_col:
                    st.error("❌ 속도와 각도 열을 모두 선택하세요!")
                    return

                try:
                    with st.spinner("계산 중..."):
                        # 작업용 데이터프레임 복사
                        df_work = df.copy()

                        # 날짜 컬럼 처리
                        if date_col:
                            df_work['date'] = pd.to_datetime(df_work[date_col])
                        else:
                            df_work['date'] = pd.to_datetime(
                                df_work[[year_col, month_col, day_col]].rename(
                                    columns={year_col: 'year',
                                            month_col: 'month',
                                            day_col: 'day'}
                                )
                            )

                        # 속도와 각도 처리
                        df_work['speed'] = pd.to_numeric(df_work[speed_col], errors='coerce')
                        df_work['angle'] = pd.to_numeric(df_work[angle_col], errors='coerce')

                        # 결측치 제거
                        df_work = df_work.dropna(subset=['speed', 'angle'])

                        # 속도 구간 생성
                        bins = [float(x.strip()) for x in speed_bins_input.split(',')]
                        bins.append(np.inf)

                        # 속도 구간 라벨 생성
                        labels = []
                        for i in range(len(bins)-1):
                            if bins[i+1] == np.inf:
                                labels.append(f"{bins[i]}+")
                            else:
                                labels.append(f"{bins[i]}-{bins[i+1]}")

                        df_work['speed_bin'] = pd.cut(df_work['speed'], bins=bins, labels=labels, right=False)

                        # 각도를 16방위로 변환
                        df_work['direction'] = df_work['angle'].apply(get_direction_16)

                        # 날짜를 년-월 형식으로 변환
                        df_work['date_str'] = df_work['date'].dt.strftime('%Y-%m')

                        # 출현율 계산
                        total_count = len(df_work)

                        # 그룹화 컬럼 매핑
                        group_map = {
                            'date': 'date_str',
                            'speed': 'speed_bin',
                            'direction': 'direction'
                        }

                        row_col = group_map[row_map[row_choice]]
                        col_col = group_map[col_map[col_choice]]

                        # 출현율 계산
                        appearance_rate = df_work.groupby([row_col, col_col]).size().unstack(fill_value=0)
                        appearance_rate_pct = (appearance_rate / total_count * 100).round(2)

                        # 방향이 행이나 열인 경우 16방위 순서대로 정렬
                        direction_order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

                        if row_map[row_choice] == 'direction':
                            appearance_rate_pct = appearance_rate_pct.reindex(direction_order, fill_value=0)

                        if col_map[col_choice] == 'direction':
                            appearance_rate_pct = appearance_rate_pct[
                                [col for col in direction_order if col in appearance_rate_pct.columns]
                            ]

                        # 합계 행 추가
                        appearance_rate_pct.loc['Total'] = appearance_rate_pct.sum()

                        # 합계 열 추가
                        appearance_rate_pct['Total'] = appearance_rate_pct.sum(axis=1)

                        # 결과 표시
                        st.success("✅ 출현율 계산 완료!")

                        st.subheader(f"📊 출현율 결과 ({data_type})")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("전체 데이터 수", f"{total_count:,}")
                        with col2:
                            st.metric("행", row_choice)
                        with col3:
                            st.metric("열", col_choice)

                        st.write("**출현율 (%)**")
                        st.dataframe(
                            appearance_rate_pct.style.format("{:.2f}").background_gradient(cmap='YlOrRd', axis=None),
                            use_container_width=True,
                            height=600
                        )

                        # 장미도 그래프
                        st.subheader("🌹 장미도 (Wind Rose)")

                        try:
                            fig = create_wind_rose(df_work, bins, labels, data_type)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.warning(f"장미도 생성 중 오류: {str(e)}")

                        # 다운로드 버튼
                        st.subheader("💾 결과 다운로드")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Excel 다운로드
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                appearance_rate_pct.to_excel(writer, sheet_name='출현율')
                            buffer.seek(0)

                            st.download_button(
                                label="📥 Excel 파일 다운로드",
                                data=buffer,
                                file_name=f"출현율_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                        with col2:
                            # CSV 다운로드
                            csv = appearance_rate_pct.to_csv(encoding='utf-8-sig')

                            st.download_button(
                                label="📥 CSV 파일 다운로드",
                                data=csv,
                                file_name=f"출현율_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"❌ 계산 중 오류 발생: {str(e)}")
                    st.exception(e)

        except Exception as e:
            st.error(f"❌ 파일 로드 실패: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 먼저 데이터 파일을 업로드하세요.")

        # 사용 안내
        with st.expander("📖 사용 방법"):
            st.markdown("""
            ### 사용 방법

            1. **데이터 파일 업로드**: Excel(.xlsx, .xls) 또는 CSV 파일을 업로드합니다.
            2. **데이터 타입 선택**: 사이드바에서 Wind, Wave, Current 중 선택합니다.
            3. **열 선택**: 날짜, 속도, 각도에 해당하는 열을 선택합니다.
            4. **출력 테이블 구조**: 행과 열에 표시할 항목을 선택합니다.
            5. **속도 빈도 설정**: 속도 구간을 쉼표로 구분하여 입력합니다.
            6. **계산 실행**: '출현율 계산 실행' 버튼을 클릭합니다.
            7. **결과 다운로드**: Excel 또는 CSV 형식으로 결과를 다운로드합니다.

            ### 데이터 형식 예시

            | Year | Month | Day | Speed | Direction |
            |------|-------|-----|-------|-----------|
            | 2023 | 1     | 1   | 5.2   | 45        |
            | 2023 | 1     | 1   | 3.8   | 120       |
            | 2023 | 1     | 2   | 7.1   | 270       |
            """)

if __name__ == "__main__":
    main()
