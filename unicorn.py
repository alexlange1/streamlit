import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import json
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(
    page_title="Unicorn Companies Analysis",
    page_icon="🦄",
    layout="wide"
)

# Add title and subtitle
st.title("🦄 Unicorn Startup Analysis")
st.markdown("### Discover the World's Most Valuable Private Companies")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
    .metric-value {
        font-size: 48px;  /* Reduced from 72px */
        font-weight: bold;
        color: #9370DB;
    }
    .metric-label {
        font-size: 18px;  /* Increased from 16px */
        color: #666;
        margin-top: 5px;
    }
    .investor-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #333; /* Added darker text color for better readability */
    }
    .news-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 25px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateY(-5px);
    }
    .news-title {
        color: #9370DB;
        font-size: 24px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    .news-desc {
        color: #444;
        font-size: 16px;
        line-height: 1.6;
    }
    .news-meta {
        color: #888;
        font-size: 14px;
        margin-top: 15px;
        border-top: 1px solid #eee;
        padding-top: 10px;
    }
    .prediction-result {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv('/Users/alexanderlange/Desktop/Unicorn_Companies.csv')

# Convert Valuation to numeric, removing "$" and "B"
df['Valuation'] = df['Valuation'].str.replace('$', '').str.replace('B', '').astype(float)

# Merge AI categories
df['Industry'] = df['Industry'].replace({
    'Artificial Intelligence': 'Machine Learning',
    'AI & Machine Learning': 'Machine Learning'
})

# Add filters in sidebar with improved organization
st.sidebar.header('Filter Data')

# Create expanders for each filter category
with st.sidebar.expander("Industry Filter", expanded=False):
    industries = sorted(df['Industry'].unique().tolist())
    if 'industry_select' not in st.session_state:
        st.session_state.industry_select = industries
    selected_industries = st.multiselect(
        'Select Industries',
        industries,
        default=st.session_state.industry_select
    )
    st.session_state.industry_select = selected_industries

with st.sidebar.expander("Country Filter", expanded=False):
    countries = sorted(df['Country'].unique().tolist())
    selected_countries = st.multiselect(
        'Select Countries',
        countries,
        default=countries,
        key='country_select'
    )

with st.sidebar.expander("Valuation Filter", expanded=False):
    min_val = float(df['Valuation'].min())
    max_val = float(df['Valuation'].max())
    valuation_range = st.slider(
        'Valuation Range (Billion $)',
        min_val, max_val, (min_val, max_val)
    )

# Reset button for filters at the bottom of sidebar
if st.sidebar.button('🔄 Reset All Filters', type='primary'):
    st.session_state.industry_select = df['Industry'].unique().tolist()
    st.session_state.country_select = df['Country'].unique().tolist()
    st.rerun()

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
filtered_df = filtered_df[
    (filtered_df['Valuation'] >= valuation_range[0]) & 
    (filtered_df['Valuation'] <= valuation_range[1])
]

# Add latitude and longitude data for mapping
country_coords = {
    'United States': (37.0902, -95.7129),
    'China': (35.8617, 104.1954),
    'India': (20.5937, 78.9629),
    'United Kingdom': (55.3781, -3.4360),
    'Germany': (51.1657, 10.4515),
    'France': (46.2276, 2.2137),
    'Israel': (31.0461, 34.8516),
    'Singapore': (1.3521, 103.8198),
    'Brazil': (-14.2350, -51.9253),
    'Canada': (56.1304, -106.3468),
    'Sweden': (60.1282, 18.6435),
    'South Korea': (35.9078, 127.7669),
    'Australia': (-25.2744, 133.7751),
    'Netherlands': (52.1326, 5.2913),
    'Indonesia': (-0.7893, 113.9213),
    'Japan': (36.2048, 138.2529),
    'Switzerland': (46.8182, 8.2275)
}

# Add coordinates to dataframe
filtered_df['Latitude'] = filtered_df['Country'].map(lambda x: country_coords.get(x, (0,0))[0])
filtered_df['Longitude'] = filtered_df['Country'].map(lambda x: country_coords.get(x, (0,0))[1])

# Create tabs with enhanced styling
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Data Explorer",
    "📈 Insights", 
    "🗺️ Geographic View",
    "📰 News",
    "💼 Investors",
    "🎮 Trivia",
    "✨ 🔮 **PREDICTOR** 🔮 ✨" # Special formatting for predictor
])

# Add enhanced styling for Predictor tab
st.markdown("""
    <style>
    /* Make predictor tab larger and more prominent */
    .stTabs [data-baseweb="tab-list"] button:last-child [data-testid="stMarkdownContainer"] p {
        font-size: 22px !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #9370DB, #4B0082);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s infinite;
        padding: 5px 15px;
        border-radius: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }

    /* Pulsing animation */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }

    /* Rainbow effect */
    .stTabs [data-baseweb="tab-list"] button:last-child [data-testid="stMarkdownContainer"] p::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8f00ff);
        animation: rainbow 2s linear infinite;
        mix-blend-mode: overlay;
        opacity: 0.5;
    }

    @keyframes rainbow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Sparkle effect */
    .stTabs [data-baseweb="tab-list"] button:last-child [data-testid="stMarkdownContainer"] p::after {
        content: '✨';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: sparkle 1s ease-in-out infinite;
        opacity: 0;
    }

    @keyframes sparkle {
        0%, 100% { opacity: 0; transform: translate(-50%, -50%) scale(0); }
        50% { opacity: 1; transform: translate(-50%, -50%) scale(1.5); }
    }

    /* Enhanced hover effect */
    .stTabs [data-baseweb="tab-list"] button:last-child:hover [data-testid="stMarkdownContainer"] p {
        transform: scale(1.1);
        transition: transform 0.3s ease;
        box-shadow: 0 0 20px rgba(147, 112, 219, 0.8);
        text-shadow: 2px 2px 4px rgba(75, 0, 130, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

with tab1:
    st.header("Explore the Unicorn Companies Dataset")
    st.dataframe(filtered_df)

with tab2:
    st.header("Key Insights")
    
    # Top industries by number of unicorns
    st.subheader("Top Industries by Number of Unicorns")
    fig1 = px.bar(
        filtered_df['Industry'].value_counts().head(10).reset_index(),
        x='Industry',
        y='count',
        color='Industry',
        title="Number of Unicorns by Industry"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Valuation distribution
    st.subheader("Valuation Distribution")
    fig2 = px.box(
        filtered_df,
        x='Industry',
        y='Valuation',
        title="Valuation Distribution by Industry"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary metrics with enhanced styling
    st.subheader("Summary Metrics")
    st.markdown("""
        <div style='display: flex; justify-content: space-around; margin: 30px 0;'>
            <div class='metric-card' style='flex: 1; margin: 0 15px; background: linear-gradient(135deg, #9370DB, #8A2BE2); color: white;'>
                <div class='metric-value' style='font-size: 64px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{}</div>
                <div class='metric-label' style='color: rgba(255,255,255,0.9); font-size: 20px;'>Total Unicorns</div>
            </div>
            <div class='metric-card' style='flex: 1; margin: 0 15px; background: linear-gradient(135deg, #BA55D3, #9370DB); color: white;'>
                <div class='metric-value' style='font-size: 64px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>${:.1f}B</div>
                <div class='metric-label' style='color: rgba(255,255,255,0.9); font-size: 20px;'>Average Valuation</div>
            </div>
            <div class='metric-card' style='flex: 1; margin: 0 15px; background: linear-gradient(135deg, #DDA0DD, #BA55D3); color: white;'>
                <div class='metric-value' style='font-size: 64px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{}</div>
                <div class='metric-label' style='color: rgba(255,255,255,0.9); font-size: 20px;'>Total Countries</div>
            </div>
        </div>
    """.format(
        len(filtered_df),
        filtered_df['Valuation'].mean(),
        filtered_df['Country'].nunique()
    ), unsafe_allow_html=True)

    # Quick Facts with enhanced styling
    st.subheader("Quick Facts")
    top_city = filtered_df['City'].value_counts().index[0]
    top_city_count = filtered_df['City'].value_counts().iloc[0]
    avg_val_top_city = filtered_df[filtered_df['City'] == top_city]['Valuation'].mean()
    total_cities = filtered_df['City'].nunique()

    # Create 2x2 grid of metrics using columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Leading Innovation Hub
        st.markdown("""
            <div style='background: linear-gradient(120deg, #FF6B6B, #FF8E8E); 
                      padding: 20px;
                      border-radius: 15px;
                      text-align: center;
                      margin: 10px;
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>{}</h1>
                <p style='color: white; font-size: 18px; margin-top: 10px;'>Leading Innovation Hub</p>
            </div>
        """.format(top_city), unsafe_allow_html=True)
        
        # Average Valuation
        st.markdown("""
            <div style='background: linear-gradient(120deg, #4ECDC4, #45B7AF);
                      padding: 20px;
                      border-radius: 15px;
                      text-align: center;
                      margin: 10px;
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>${:.1f}B</h1>
                <p style='color: white; font-size: 18px; margin-top: 10px;'>Average Valuation</p>
            </div>
        """.format(avg_val_top_city), unsafe_allow_html=True)

    with col2:
        # Unicorns in Hub
        st.markdown("""
            <div style='background: linear-gradient(120deg, #FFD93D, #F4C430);
                      padding: 20px;
                      border-radius: 15px;
                      text-align: center;
                      margin: 10px;
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>{}</h1>
                <p style='color: white; font-size: 18px; margin-top: 10px;'>Unicorns in Hub</p>
            </div>
        """.format(top_city_count), unsafe_allow_html=True)
        
        # Global Cities
        st.markdown("""
            <div style='background: linear-gradient(120deg, #6C63FF, #5A52E5);
                      padding: 20px;
                      border-radius: 15px;
                      text-align: center;
                      margin: 10px;
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>{}</h1>
                <p style='color: white; font-size: 18px; margin-top: 10px;'>Global Cities</p>
            </div>
        """.format(total_cities), unsafe_allow_html=True)
    
    # Add a time series chart of unicorn formation
    st.subheader("Unicorn Growth Over Time")
    filtered_df['Year'] = pd.to_datetime(filtered_df['Date Joined']).dt.year
    yearly_counts = filtered_df.groupby('Year').size().reset_index(name='New Unicorns')
    yearly_counts['Cumulative Unicorns'] = yearly_counts['New Unicorns'].cumsum()
    
    fig_timeline = px.line(
        yearly_counts,
        x='Year',
        y=['New Unicorns', 'Cumulative Unicorns'],
        title="Growth of Unicorn Startups Over Time",
        labels={'value': 'Number of Unicorns', 'Year': 'Year'},
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)

with tab3:
    st.header("Geographic Distribution of Unicorns")
    
    # Group by country and count unicorns
    location_counts = filtered_df.groupby('Country').agg({
        'Valuation': ['count', 'mean']
    }).reset_index()
    location_counts.columns = ['Country', 'Count', 'Avg_Valuation']
    
    location_counts = location_counts.merge(
        filtered_df.groupby('Country')[['Latitude', 'Longitude']].mean(),
        on='Country'
    )
    
    # Create a map using plotly with darker purple shades
    fig_map = go.Figure(go.Scattermapbox(
        lat=location_counts['Latitude'],
        lon=location_counts['Longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=location_counts['Count']*4,
            color=location_counts['Count'],
            colorscale=[
                [0, '#D8BFD8'],    # Light purple but darker than before
                [0.2, '#DDA0DD'],  # Medium light purple
                [0.4, '#BA55D3'],  # Medium purple
                [0.6, '#9370DB'],  # Medium dark purple
                [0.8, '#8A2BE2'],  # Dark purple
                [1, '#4B0082']     # Indigo
            ],
            showscale=True,
            colorbar=dict(title='Number of Unicorns'),
            sizemin=8,
            sizemode='area'
        ),
        text=location_counts['Country'] + 
             '<br>Unicorns: ' + location_counts['Count'].astype(str) +
             '<br>Avg Valuation: $' + location_counts['Avg_Valuation'].round(2).astype(str) + 'B',
        hoverinfo='text'
    ))

    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=20, lon=0),
            zoom=1
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.header("📰 Latest Startup News")
    
    # Function to fetch news using NewsAPI
    def fetch_startup_news():
        api_key = "2ab2c1afd8b243648ba810263eb5d4c4"  # Your API key
        url = f"https://newsapi.org/v2/everything?q=startup+funding+unicorn&apiKey={api_key}&sortBy=publishedAt&language=en"
        try:
            response = requests.get(url)
            news = response.json()
            return news.get('articles', [])[:5]  # Get latest 5 articles
        except:
            return []

    news_articles = fetch_startup_news()
    
    if news_articles:
        for article in news_articles:
            st.markdown("""
                <div class='news-card'>
                    <div class='news-title'>{}</div>
                    <div class='news-desc'>{}</div>
                    <div class='news-meta'>
                        Published: {} | Source: {} | 
                        <a href="{}" target="_blank">Read More →</a>
                    </div>
                </div>
            """.format(
                article['title'],
                article['description'],
                article['publishedAt'].split('T')[0],
                article['source']['name'],
                article['url']
            ), unsafe_allow_html=True)
    else:
        st.warning("Unable to fetch news at the moment. Please try again later.")

with tab5:
    st.header("💼 Top Investors in Unicorn Startups")
    
    # Expanded investor data
    investors = {
        'Sequoia Capital': {
            'website': 'https://www.sequoiacap.com',
            'investments': 45,
            'total_value': '120B',
            'focus': 'Technology, Healthcare, Consumer'
        },
        'SoftBank Vision Fund': {
            'website': 'https://www.softbank.jp/en/',
            'investments': 40,
            'total_value': '100B',
            'focus': 'AI, Technology, Platform Businesses'
        },
        'Tiger Global Management': {
            'website': 'https://www.tigerglobal.com',
            'investments': 35,
            'total_value': '80B',
            'focus': 'Internet, Software, Consumer Tech'
        },
        'Accel': {
            'website': 'https://www.accel.com',
            'investments': 30,
            'total_value': '60B',
            'focus': 'Enterprise Software, Consumer Internet'
        },
        'Andreessen Horowitz': {
            'website': 'https://a16z.com',
            'investments': 28,
            'total_value': '55B',
            'focus': 'Software, Crypto, Fintech'
        },
        'Kleiner Perkins': {
            'website': 'https://www.kleinerperkins.com',
            'investments': 25,
            'total_value': '50B',
            'focus': 'Digital Health, Fintech, Enterprise'
        }
    }
    
    # Display investor statistics in a grid
    col1, col2 = st.columns(2)
    
    investors_list = list(investors.items())
    half = len(investors_list) // 2
    
    for i, (name, data) in enumerate(investors_list):
        with col1 if i < half else col2:
            st.markdown(f"""
                <div class='investor-card'>
                    <h3>{name}</h3>
                    <p><strong>Investments:</strong> {data['investments']}</p>
                    <p><strong>Portfolio Value:</strong> ${data['total_value']}</p>
                    <p><strong>Focus Areas:</strong> {data['focus']}</p>
                    <a href='{data['website']}' target='_blank'>Visit Website →</a>
                </div>
            """, unsafe_allow_html=True)

with tab6:
    st.header("Unicorn Trivia Game")
    
    st.markdown("""
    ### Test Your Knowledge About Unicorn Companies!
    Play this interactive trivia game to learn more about the world's most valuable startups.
    """)
    
    # Initialize session state variables
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'question_number' not in st.session_state:
        st.session_state.question_number = 0
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = set()
    if 'show_answer' not in st.session_state:
        st.session_state.show_answer = False
    if 'answer_time' not in st.session_state:
        st.session_state.answer_time = None
    
    # Define trivia questions
    trivia_questions = [
        {
            'question': 'What is a "unicorn" company?',
            'options': [
                'A company with a mythical business model',
                'A private company valued at $1 billion or more',
                'A company founded by college dropouts',
                'A company that has never made a profit'
            ],
            'correct': 1
        },
        {
            'question': 'Which industry has the most unicorn companies?',
            'options': ['Fintech', 'E-commerce', 'AI/ML', 'Social Media'],
            'correct': 0
        },
        {
            'question': 'Which country, after the US, has the most unicorn companies?',
            'options': ['India', 'China', 'UK', 'Germany'],
            'correct': 1
        },
        {
            'question': 'What is the typical age of a company when it reaches unicorn status?',
            'options': ['1-2 years', '3-4 years', '7-8 years', '10+ years'],
            'correct': 2
        }
    ]
    
    # Display current score
    st.markdown(f"**Current Score: {st.session_state.score} / {len(trivia_questions)}**")
    
    # Display question
    if st.session_state.question_number < len(trivia_questions):
        current_q = trivia_questions[st.session_state.question_number]
        
        st.markdown(f"### Question {st.session_state.question_number + 1}:")
        st.markdown(f"**{current_q['question']}**")
        
        # Create radio buttons for answers
        answer = st.radio(
            "Select your answer:",
            current_q['options'],
            key=f"q_{st.session_state.question_number}"
        )
        
        if st.button("Submit Answer"):
            st.session_state.show_answer = True
            st.session_state.answer_time = time.time()
            
            if current_q['options'].index(answer) == current_q['correct']:
                st.success("CORRECT!")
                st.session_state.score += 1
            else:
                st.error("Incorrect. The correct answer was: " + current_q['options'][current_q['correct']])
            
            # Wait for 3 seconds before proceeding
            time.sleep(3)
            
            st.session_state.question_number += 1
            st.session_state.show_answer = False
            st.rerun()
            
    else:
        # Game completed
        final_score = st.session_state.score
        st.markdown(f"### Game Complete!")
        st.markdown(f"Your final score: {final_score} out of {len(trivia_questions)}")
        
        # Display performance message
        if final_score == len(trivia_questions):
            st.balloons()
            st.markdown("### Perfect Score! You're a Unicorn Expert!")
        elif final_score >= len(trivia_questions) * 0.7:
            st.markdown("### Great Work! You're a Rising Star!")
        else:
            st.markdown("### Keep Learning! Every Expert Started Somewhere!")
        
        # Reset button
        if st.button("Play Again"):
            st.session_state.score = 0
            st.session_state.question_number = 0
            st.session_state.show_answer = False
            st.rerun()

# Predictor Tab Content
with tab7:
    st.markdown("<h1 style='text-align: center; color: #9370DB;'>✨ Startup Valuation Predictor ✨</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Discover your startup's potential valuation</p>", unsafe_allow_html=True)

    # Load and prepare data
    try:
        df = pd.read_csv('/Users/alexanderlange/Desktop/Final_Dataset_Modified.csv')
    except FileNotFoundError:
        st.error("Error: Could not find the dataset file. Please check the file path.")
        st.stop()

    # Clean up Industry column by replacing investor lists with city values
    investor_lists = [
        'Sequoia Capital, Thoma Bravo, Softbank',
        'Tiger Global Management, Tiger Brokers, DCM Ventures',
        'Jungle Ventures, Accel, Venture Highway',
        'Vision Plus Capital, GSR Ventures, ZhenFund',
        'Hopu Investment Management, Boyu Capital, DC Thomson Ventures',
        '500 Global, Rakuten Ventures, Golden Gate Ventures',
        'Sequoia Capital China, ING, Alibaba Entrepreneurs Fund',
        'Sequoia Capital China, Shunwei Capital Partners, Qualgro',
        'Dragonfly Captial, Qiming Venture Partners, DST Global',
        'SingTel Innov8, Alpha JWC Ventures, Golden Gate Ventures',
        'Mundi Ventures, Doqling Capital Partners, Activant Capital',
        'Vertex Ventures SE Asia, Global Founders Capital, Visa Ventures',
        'Andreessen Horowitz, DST Global, IDG Capital',
        'B Capital Group, Monk\'s Hill Ventures, Dynamic Parcel Distribution',
        'Temasek, Guggenheim Investments, Qatar Investment Authority',
        'Kuang-Chi'
    ]

    for investor_list in investor_lists:
        df.loc[df['Industry'] == investor_list, 'Industry'] = df.loc[df['Industry'] == investor_list, 'City']

    # Fix spelling errors in Industry column
    df['Industry'] = df['Industry'].replace('Finttech', 'Fintech')
    df['Industry'] = df['Industry'].replace('Artificial intelligence', 'Artificial Intelligence')

    # Prepare data for model
    valuation_column = [col for col in df.columns if 'valuation' in col.lower()][0]
    df[valuation_column] = df[valuation_column].str.replace('$', '').str.replace('B', '').astype(float)
    df = df[df['Industry'] != '500 Global']

    # Prepare features
    X = pd.get_dummies(df[['Industry', 'Country']])
    y = df[valuation_column]

    # Train model
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X, y)

    # Create input form
    with st.container():
        st.markdown("<div class='prediction-input'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_industry = st.selectbox(
                "What industry is your startup in?",
                options=sorted(df['Industry'].unique())
            )
            
            selected_country = st.selectbox(
                "Where is your startup based?",
                options=sorted(df['Country'].unique())
            )
            
            annual_revenue = st.number_input(
                "Annual Revenue (in billions $)",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                step=0.1
            )
            
        with col2:
            funding_stage = st.selectbox(
                "What's your current funding stage?",
                options=['Seed', 'Series A', 'Series B', 'Series C', 'Series D+']
            )
            
            is_profitable = st.selectbox(
                "Is your startup profitable?",
                options=['Yes', 'No']
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Predict Valuation", type="primary"):
            # Create prediction input
            pred_input = pd.DataFrame(columns=X.columns, data=np.zeros((1, len(X.columns))))
            pred_input[f"Industry_{selected_industry}"] = 1
            pred_input[f"Country_{selected_country}"] = 1
            
            # Make prediction
            predicted_valuation = model.predict(pred_input)[0]
            
            # Adjust prediction based on additional factors
            # Apply revenue multiplier (typical valuation multiples range from 5x to 15x revenue)
            revenue_multiplier = 10  # Using average multiple of 10x
            revenue_based_valuation = annual_revenue * revenue_multiplier
            
            # Blend model prediction with revenue-based valuation (equal weighting)
            predicted_valuation = (predicted_valuation + revenue_based_valuation) / 2
            
            # Apply profitability adjustment
            if is_profitable == 'Yes':
                predicted_valuation *= 1.2
                
            # Apply funding stage multiplier
            funding_multipliers = {
                'Seed': 1.0,
                'Series A': 1.02,
                'Series B': 1.04,
                'Series C': 1.06,
                'Series D+': 1.08
            }
            
            predicted_valuation *= funding_multipliers[funding_stage]
            
            # Display results with enhanced styling
            st.markdown("<div style='text-align: center; padding: 40px 0;'>" + 
                       "<div style='font-size: 24px; color: #666; margin-bottom: 15px;'>Predicted Valuation</div>" +
                       "<div style='font-size: 72px; font-weight: bold; color: #9370DB; " +
                       "text-shadow: 2px 2px 4px rgba(147, 112, 219, 0.3); " +
                       "animation: pulse 2s infinite;'>" +
                       "${:.2f}B".format(predicted_valuation) +
                       "</div></div>" +
                       "<style>" +
                       "@keyframes pulse {" +
                       "0% { transform: scale(1); }" +
                       "50% { transform: scale(1.05); }" +
                       "100% { transform: scale(1); }" +
                       "}</style>", unsafe_allow_html=True)
            
            # Add sparkles around the prediction 
            st.markdown("<h3 style='text-align: center; color: #9370DB;'>✨ Your Unicorn Potential ✨</h3>", unsafe_allow_html=True)
