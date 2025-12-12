# 🏠 Real Estate Investment Advisor

An AI-powered web application that predicts property investment potential and forecasts future prices using machine learning models trained on 250,000+ Indian real estate properties.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-estate-advisor-9ff4usscvnjkpgdhpo59x8.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo

**Try the app:** [https://real-estate-advisor-9ff4usscvnjkpgdhpo59x8.streamlit.app/](https://real-estate-advisor-9ff4usscvnjkpgdhpo59x8.streamlit.app/)

## 📊 Features

### 🔮 Investment Prediction
- **Binary Classification**: Determines if a property is a "Good Investment" or not
- **Confidence Score**: Provides probability-based confidence levels
- **Multi-factor Analysis**: Evaluates price, location, infrastructure, and property features

### 💰 Price Forecasting
- **5-Year Price Projection**: Predicts property value 5 years into the future
- **Annual Return Calculation**: Estimates expected yearly returns
- **Growth Visualization**: Interactive charts showing price trajectories

### 📈 Market Insights
- **City-wise Price Analysis**: Compare average prices across top cities
- **Property Type Distribution**: Understand market composition
- **BHK Investment Quality**: Analyze returns by bedroom configuration
- **Age vs Investment**: Correlation between property age and investment potential

### 🗺️ Data Explorer
- **Advanced Filtering**: Filter by state, city, BHK, price range, and size
- **Interactive Data Table**: Browse through 5,000+ property records
- **Export Functionality**: Download filtered data as CSV

## 🛠️ Technologies Used

### Machine Learning
- **Scikit-learn**: Logistic Regression for classification
- **Linear Regression**: Price prediction model
- **Random Forest**: Feature importance analysis
- **XGBoost**: High-performance gradient boosting

### Web Framework
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive visualizations and charts

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Python 3.10+**: Core programming language

## 📁 Project Structure

```
real-estate-advisor/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── data/
│   └── processed_housing_data.csv  # Processed property dataset (5,000 records)
└── models/
    ├── classifier_model.pkl        # Investment classification model
    ├── regressor_model.pkl         # Price prediction model
    ├── label_encoders.pkl          # Categorical feature encoders
    ├── scaler.pkl                  # Feature scaling transformer
    └── feature_columns.pkl         # Model feature list
```

## 🎯 Model Performance

### Classification (Good Investment Prediction)
- **Algorithm**: Logistic Regression
- **Accuracy**: ~78%
- **Features**: 30+ engineered features including location, price metrics, infrastructure scores

### Regression (Price Forecasting)
- **Algorithm**: Linear Regression
- **R² Score**: ~0.95
- **MAE**: Low prediction error
- **Features**: Property size, age, location, amenities, market trends

## 💻 Local Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/lalit2244/real-estate-advisor.git
   cd real-estate-advisor
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

## 📖 Usage Guide

### Making a Prediction

1. **Navigate to Prediction Page** (default view)

2. **Enter Property Details**:
   - **Location**: Select state and city
   - **Property Type**: Choose from Apartment, Villa, House, etc.
   - **Specifications**: Enter BHK, size, price, year built
   - **Features**: Add floor details, parking, furnishing status
   - **Amenities**: Specify nearby schools, hospitals, transport access
   - **Facing Direction**: Select property orientation

3. **Click "🔍 Analyze Investment Potential"**

4. **View Results**:
   - Investment rating (Good/Not Recommended)
   - Confidence percentage
   - 5-year price forecast
   - Expected annual returns
   - Detailed analysis and recommendations
   - Interactive price projection chart

### Exploring Market Insights

1. Navigate to **"📊 Market Insights"** page
2. View key metrics: total properties, average prices, good investment percentage
3. Analyze interactive charts:
   - Price distribution by city
   - Property type breakdown
   - Investment quality by BHK
   - Age vs investment correlation

### Using Data Explorer

1. Navigate to **"🗺️ Data Explorer"** page
2. Apply filters:
   - Select states and BHK types
   - Adjust price range slider
   - Set size (sqft) range
3. View filtered results in interactive table
4. Download filtered data using "📥 Download" button

## 🎓 How It Works

### Data Processing Pipeline

1. **Data Collection**: Aggregated from 250,000+ property listings across India
2. **Feature Engineering**:
   - Price per square foot calculation
   - Infrastructure scoring (schools, hospitals, transport)
   - Property age computation
   - Location-based feature encoding
3. **Model Training**:
   - Classification: Trained on multi-factor investment criteria
   - Regression: Trained on historical price appreciation patterns
4. **Prediction**: Real-time inference on user-provided property data

### Investment Classification Logic

A property is classified as "Good Investment" based on:
- ✅ Price below city median (competitive pricing)
- ✅ Optimal BHK configuration (2-3 BHK high liquidity)
- ✅ Strong infrastructure score (>50/100)
- ✅ Relatively new property (<15 years old)
- ✅ Ready-to-move availability status

**Score ≥ 3/5 factors → Good Investment** ✅

### Price Forecasting Model

Future price calculation considers:
- Current market price
- Historical appreciation rate (8% baseline)
- Location growth multipliers
- Infrastructure quality adjustments
- Property-specific features (RERA, amenities)

## 📊 Dataset Information

- **Total Records**: 250,000+ properties (5,000 in deployment version)
- **Coverage**: Major cities across India
- **Features**: 30+ attributes per property
- **Sources**: Real estate listings, market data, infrastructure databases

### Key Features
- Location (State, City, Locality)
- Property specifications (BHK, Size, Price, Type)
- Age and construction details
- Nearby amenities (Schools, Hospitals)
- Infrastructure accessibility
- Furnishing and parking details
- Investment metrics

## 🌟 Key Highlights

- 🚀 **Fast Predictions**: Real-time investment analysis in seconds
- 📊 **Data-Driven**: Trained on 250,000+ real property listings
- 🎯 **High Accuracy**: 78%+ accuracy in investment classification
- 💡 **Explainable AI**: Provides reasoning behind recommendations
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🌐 **Cloud Deployed**: Accessible worldwide via Streamlit Cloud
- 🔄 **Always Available**: 24/7 uptime with auto-scaling

## 🔮 Future Enhancements

### Planned Features
- [ ] Historical price trend analysis
- [ ] Neighborhood crime rate integration
- [ ] School quality ratings
- [ ] Rental yield predictions
- [ ] Property comparison tool
- [ ] Email alerts for good deals
- [ ] Investment portfolio tracker
- [ ] Mobile app version
- [ ] API access for developers
- [ ] Multi-language support

### Model Improvements
- [ ] Deep learning models (Neural Networks)
- [ ] Ensemble methods for better accuracy
- [ ] Time-series forecasting for price trends
- [ ] Sentiment analysis from reviews
- [ ] Image-based property valuation

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Contribution
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- 🧪 Test coverage
- 🌍 Localization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Lalit Kumar**

- GitHub: [@lalit2244](https://github.com/lalit2244)
- Project Link: [https://github.com/lalit2244/real-estate-advisor](https://github.com/lalit2244/real-estate-advisor)
- Live App: [https://real-estate-advisor-9ff4usscvnjkpgdhpo59x8.streamlit.app/](https://real-estate-advisor-9ff4usscvnjkpgdhpo59x8.streamlit.app/)

## 🙏 Acknowledgments

- Dataset sourced from Indian real estate market data
- Streamlit for the amazing web framework
- Scikit-learn for machine learning tools
- Plotly for interactive visualizations
- The open-source community

## 📧 Contact & Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/lalit2244/real-estate-advisor/issues)
- **Questions**: Open a discussion on [GitHub Discussions](https://github.com/lalit2244/real-estate-advisor/discussions)
- **Feedback**: Your feedback is valuable! Please share your experience

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/lalit2244/real-estate-advisor?style=social)
![GitHub forks](https://img.shields.io/github/forks/lalit2244/real-estate-advisor?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/lalit2244/real-estate-advisor?style=social)

---

## 🎯 Use Cases

### For Home Buyers
- Evaluate if a property is worth investing in
- Predict future appreciation
- Compare properties across cities
- Make data-driven purchase decisions

### For Real Estate Agents
- Provide clients with investment analysis
- Price properties competitively
- Identify high-potential listings
- Build trust with data-backed insights

### For Investors
- Portfolio optimization
- Risk assessment
- ROI forecasting
- Market trend analysis

### For Developers
- Market research
- Pricing strategy
- Feature prioritization
- Location analysis

---

## 📚 Related Projects

- [House Price Prediction ML](https://github.com/topics/house-price-prediction)
- [Real Estate Analysis](https://github.com/topics/real-estate)
- [Streamlit ML Apps](https://github.com/topics/streamlit)

---

## ⭐ Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=lalit2244/real-estate-advisor&type=Date)](https://star-history.com/#lalit2244/real-estate-advisor&Date)

---

<div align="center">

### Made with ❤️ using Python and Streamlit

**[⬆ Back to Top](#-real-estate-investment-advisor)**

</div>
